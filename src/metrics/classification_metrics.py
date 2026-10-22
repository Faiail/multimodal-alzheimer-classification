from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """Compute binary or multiclass classification metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    if num_classes == 2:
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    else:
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        prec = prec_macro
        rec = rec_macro
        f1 = f1_macro

    auc_ovr = 0.5
    auc_ovo = 0.5
    aupr = 0.0
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            if num_classes == 2:
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob.reshape(-1)
                if len(np.unique(y_true)) > 1:
                    auc_ovr = float(roc_auc_score(y_true, prob_pos))
                    auc_ovo = auc_ovr
                    aupr = float(average_precision_score(y_true, prob_pos))
            else:
                if len(np.unique(y_true)) == num_classes:
                    auc_ovr = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
                    auc_ovo = float(roc_auc_score(y_true, y_prob, multi_class="ovo", average="macro"))
                aupr_list = []
                for c in range(num_classes):
                    y_bin = (y_true == c).astype(int)
                    if y_bin.sum() > 0:
                        aupr_list.append(average_precision_score(y_bin, y_prob[:, c]))
                aupr = float(np.mean(aupr_list)) if aupr_list else 0.0
        except ValueError:
            pass

    metrics: Dict[str, float] = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc": auc_ovr,
        "auc_ovr": auc_ovr,
        "auc_ovo": auc_ovo,
        "aupr": aupr,
        "aupr_macro": aupr,
        "cohen_kappa": kappa,
        "mcc": mcc,
    }

    p_cls, r_cls, f1_cls, sup_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    for c in range(num_classes):
        metrics[f"precision_c{c}"] = float(p_cls[c])
        metrics[f"recall_c{c}"] = float(r_cls[c])
        metrics[f"f1_c{c}"] = float(f1_cls[c])
        metrics[f"support_c{c}"] = int(sup_cls[c])
        if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] > c:
            y_bin = (y_true == c).astype(int)
            if 0 < y_bin.sum() < len(y_bin):
                try:
                    metrics[f"auc_c{c}"] = float(roc_auc_score(y_bin, y_prob[:, c]))
                except ValueError:
                    metrics[f"auc_c{c}"] = 0.5
            else:
                metrics[f"auc_c{c}"] = 0.5

    return metrics


class ClassificationMetrics:
    """Stateful accumulator for epoch-level binary and multiclass metrics."""

    def __init__(self, num_classes: int = 2):
        self.num_classes = int(num_classes)
        self.reset()

    def reset(self):
        self.y_true = []
        self.y_pred = []
        self.y_prob = []

    def update(self, probs: np.ndarray, preds: np.ndarray, targets: np.ndarray):
        self.y_prob.append(np.asarray(probs))
        self.y_pred.extend(np.asarray(preds).tolist())
        self.y_true.extend(np.asarray(targets).tolist())

    def compute(self) -> Dict[str, float]:
        if not self.y_true:
            return {}
        y_prob_arr = np.concatenate(self.y_prob, axis=0) if self.y_prob else None
        return compute_classification_metrics(
            y_true=np.asarray(self.y_true),
            y_pred=np.asarray(self.y_pred),
            y_prob=y_prob_arr,
            num_classes=self.num_classes,
        )

    def confusion_matrix(self) -> np.ndarray:
        return confusion_matrix(
            np.asarray(self.y_true, dtype=int),
            np.asarray(self.y_pred, dtype=int),
            labels=list(range(self.num_classes)),
        )

