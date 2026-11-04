"""
Script for hyperparameter tuning on a base Multimodal GCN model.
"""
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.dataset import HybridAlzheimerDataset
from src.models.model import GCN_KAN_Classifier
from src.models.layers import FocalLoss
from src.data.batch_utils import (build_graph_dataloader, fma_dataset_kwargs,
                                  get_node_feature_dim, get_node_feature_dims, move_graph_batch_to_device)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment" / "multiclass" / "gcn_multigraph.yaml"
CLINICAL_COLS = ['AGE', 'Gender_Male', 'PTEDUCAT', 'APOE4', 'MMSE', 'ABETA_final', 'TAU_final', 'PTAU_final', 'CSF_Available']

def load_config():
    with open(CONFIG_PATH, 'r') as f: return yaml.safe_load(f)

def prepare_multiclass_data():
    print("Loading multiclass data...")
    cfg = load_config()
    
    df_img = pd.read_csv(PROJECT_ROOT / cfg['data']['features_path'])
    df_img['PTID'] = df_img['subject_id'].apply(lambda x: "_".join(x.split('_')[:3]))
    
    multiclass_path = PROJECT_ROOT / "data" / "02_processed" / "multiclass_cn_mci_ad_features.csv"
    df_clin = pd.read_csv(multiclass_path)
    
    df_merged = pd.merge(df_img, df_clin, on='PTID', how='inner', suffixes=('', '_clin'))
    if 'label_clin' in df_merged.columns: df_merged['label'] = df_merged['label_clin']
    df_merged = df_merged.dropna(subset=['label']).reset_index(drop=True)
    
    img_cols = [c for c in df_merged.columns if 'GM_' in c or 'PET_' in c]
    df_merged[img_cols] = StandardScaler().fit_transform(df_merged[img_cols].fillna(0))
    df_merged[CLINICAL_COLS] = StandardScaler().fit_transform(df_merged[CLINICAL_COLS].fillna(0))
    
    print(f"✅ Dataset: {len(df_merged)} samples")
    return df_merged, cfg

def train_model_with_params(train_df, val_df, test_df, cfg, params, device):
    coords_path = PROJECT_ROOT / cfg['data']['coords_path']
    
    # prepare dataset with zhang statistics
    ds_kwargs = {
        'sigma': cfg['model']['sigma_graph'],
        'k_neighbors': 20,
        'graph_type': 'zhang',
        'graph_mix_alpha': cfg['model'].get('hybrid_alpha', 0.5),
        'node_semantics_mode': cfg['model'].get('node_semantics_mode', 'none'),
        'clinical_cols': CLINICAL_COLS
    }
    
    ds_kwargs.update(fma_dataset_kwargs(cfg))

    train_ds_temp = HybridAlzheimerDataset(train_df, coords_path, precompute_adjacency=False, **ds_kwargs)
    zhang_stats = {'mean': train_ds_temp.global_mean, 'std': train_ds_temp.global_std}
    ds_kwargs['precomputed_stats'] = zhang_stats
    
    train_ds = HybridAlzheimerDataset(train_df, coords_path, **ds_kwargs)
    val_ds = HybridAlzheimerDataset(val_df, coords_path, **ds_kwargs)
    test_ds = HybridAlzheimerDataset(test_df, coords_path, **ds_kwargs)
    n_node_features = get_node_feature_dim(train_ds)
    hetero_metadata = train_ds.metadata() if train_ds.is_heterogeneous else None
    hetero_input_dims = get_node_feature_dims(train_ds)
    
    y_train = train_df['label'].values.astype(int)
    weight_scaling = np.sqrt(len(y_train) / np.bincount(y_train))
    sampler = WeightedRandomSampler(torch.from_numpy(weight_scaling[y_train]).float(), len(y_train), replacement=True)
    
    train_loader = build_graph_dataloader(train_ds, batch_size=params['batch_size'], sampler=sampler, shuffle=False)
    val_loader = build_graph_dataloader(val_ds, batch_size=params['batch_size'], shuffle=False)
    test_loader = build_graph_dataloader(test_ds, batch_size=params['batch_size'], shuffle=False)
    
    model = GCN_KAN_Classifier(
        in_features=n_node_features,
        n_clinical_features=len(CLINICAL_COLS),
        hidden_dim=params['hidden_dim'],
        n_classes=3,
        dropout=params['dropout'],
        use_kan_backbone=False,
        use_kan_head=False,
        graph_type='zhang',
        hetero_metadata=hetero_metadata,
        hetero_input_dims=hetero_input_dims
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = FocalLoss(gamma=1.0)
    
    best_val_acc, patience, counter, best_state = 0.0, 15, 0, None
    for epoch in range(150):
        model.train()
        for batch in train_loader:
            x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type='zhang', has_clinical=True)
            optimizer.zero_grad()
            criterion(model(x_g, adj, x_c), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type='zhang', has_clinical=True)
                val_preds.extend(torch.argmax(model(x_g, adj, x_c), 1).cpu().numpy())
                val_truths.extend(y.cpu().numpy())
        
        acc = accuracy_score(val_truths, val_preds)
        if acc > best_val_acc:
            best_val_acc = acc
            best_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    
    preds, truths, probs_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type='zhang', has_clinical=True)
            out = model(x_g, adj, x_c)
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            truths.extend(y.cpu().numpy())
            probs_list.extend(torch.softmax(out, 1).cpu().numpy())
    
    truths = np.array(truths)
    preds = np.array(preds)
    probs = np.array(probs_list)
    n_classes = probs.shape[1]
    class_names = ['CN', 'MCI', 'AD']

    metrics = {}
    metrics['accuracy'] = accuracy_score(truths, preds)
    metrics['balanced_accuracy'] = balanced_accuracy_score(truths, preds)
    metrics['precision_macro'] = precision_score(truths, preds, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(truths, preds, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(truths, preds, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(truths, preds, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(truths, preds, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(truths, preds, average='weighted', zero_division=0)
    metrics['cohen_kappa'] = cohen_kappa_score(truths, preds)
    metrics['mcc'] = matthews_corrcoef(truths, preds)

    try:
        metrics['aupr_macro'] = average_precision_score(truths, probs, average='macro')
        metrics['auroc_macro'] = roc_auc_score(truths, probs, average='macro', multi_class='ovr')
        metrics['auroc_weighted'] = roc_auc_score(truths, probs, average='weighted', multi_class='ovr')
    except ValueError:
        metrics['aupr_macro'] = 0.0
        metrics['auroc_macro'] = 0.0
        metrics['auroc_weighted'] = 0.0

    prec_per = precision_score(truths, preds, average=None, zero_division=0)
    rec_per = recall_score(truths, preds, average=None, zero_division=0)
    f1_per = f1_score(truths, preds, average=None, zero_division=0)
    try:
        aupr_per = average_precision_score(
            np.eye(n_classes)[truths.astype(int)], probs, average=None)
        auroc_per = roc_auc_score(
            np.eye(n_classes)[truths.astype(int)], probs, average=None)
    except ValueError:
        aupr_per = np.zeros(n_classes)
        auroc_per = np.zeros(n_classes)

    for c in range(n_classes):
        name = class_names[c] if c < len(class_names) else f'class_{c}'
        metrics[f'precision_{name}'] = prec_per[c]
        metrics[f'recall_{name}'] = rec_per[c]
        metrics[f'f1_{name}'] = f1_per[c]
        metrics[f'aupr_{name}'] = aupr_per[c]
        metrics[f'auroc_{name}'] = auroc_per[c]

    cm = confusion_matrix(truths, preds, labels=list(range(n_classes)))
    metrics['confusion_matrix'] = cm.tolist()

    return metrics, best_val_acc

def objective(trial, df, cfg, device, n_folds=3):
    params = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128, 256]),
        'dropout': trial.suggest_float('dropout', 0.2, 0.6, step=0.1),
        'weight_decay': trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64])
    }
    
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for i, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'], df['PTID'])):
        if i >= n_folds:
            break
            
        tr_df = df.iloc[tr_idx]
        te_df = df.iloc[te_idx]
        
        train_idx, val_idx = train_test_split(range(len(tr_df)), test_size=0.15, 
                                               stratify=tr_df['label'], random_state=42)
        train_subset = tr_df.iloc[train_idx].reset_index(drop=True)
        val_subset = tr_df.iloc[val_idx].reset_index(drop=True)
        te_subset = te_df.reset_index(drop=True)
        
        metrics, _ = train_model_with_params(train_subset, val_subset, te_subset, cfg, params, device)
        fold_scores.append(metrics['aupr_macro'])
        
        print(f"  Fold {i+1}/{n_folds}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1_macro']:.3f}, AUPR={metrics['aupr_macro']:.3f}")
    
    mean_score = np.mean(fold_scores)
    return mean_score

def main():
    print("="*80)
    print("🔧 HYPERPARAMETER TUNING - GCN Base + Focal Loss (γ=1.0) + Zhang")
    print("="*80)
    
    df, cfg = prepare_multiclass_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    print("📊 PHASE 1: Hyperparameter Search")
    print("-"*80)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='gcn_multiclass_tuning',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(
        lambda trial: objective(trial, df, cfg, device, n_folds=3),
        n_trials=30,
        timeout=None,
        show_progress_bar=True
    )
    
    print("\n" + "="*80)
    print("✅ BEST HYPERPARAMETERS FOUND")
    print("="*80)
    print(f"Best AUPR: {study.best_value:.4f}")
    print("\nBest params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("📊 PHASE 2: Final Training with Best Params (5-fold CV)")
    print("="*80)
    
    best_params = study.best_params
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []
    for i, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'], df['PTID'])):
        print(f"\n📊 FOLD {i+1}/5")
        
        tr_df = df.iloc[tr_idx]
        te_df = df.iloc[te_idx]
        
        train_idx, val_idx = train_test_split(range(len(tr_df)), test_size=0.15,
                                               stratify=tr_df['label'], random_state=42)
        train_subset = tr_df.iloc[train_idx].reset_index(drop=True)
        val_subset = tr_df.iloc[val_idx].reset_index(drop=True)
        te_subset = te_df.reset_index(drop=True)
        
        metrics, val_acc = train_model_with_params(train_subset, val_subset, te_subset, 
                                                      cfg, best_params, device)
        row = {'Fold': i+1, 'Val_Acc': val_acc}
        for k, v in metrics.items():
            if k != 'confusion_matrix':
                row[k] = v
        row['confusion_matrix'] = str(metrics['confusion_matrix'])
        results.append(row)
        
        m = metrics
        print(f"  Val Acc: {val_acc:.3f}")
        print(f"  --- Macro ---")
        print(f"  Accuracy:     {m['accuracy']:.3f}  |  Balanced Acc: {m['balanced_accuracy']:.3f}")
        print(f"  Precision:    {m['precision_macro']:.3f}  |  Recall:       {m['recall_macro']:.3f}  |  F1: {m['f1_macro']:.3f}")
        print(f"  AUPR:         {m['aupr_macro']:.3f}  |  AUROC:        {m['auroc_macro']:.3f}")
        print(f"  Cohen κ:      {m['cohen_kappa']:.3f}  |  MCC:          {m['mcc']:.3f}")
        print(f"  --- Per-class ---")
        for cls in ['CN', 'MCI', 'AD']:
            print(f"  {cls:>3s}  Prec={m[f'precision_{cls}']:.3f}  Rec={m[f'recall_{cls}']:.3f}  "
                  f"F1={m[f'f1_{cls}']:.3f}  AUPR={m[f'aupr_{cls}']:.3f}  AUROC={m[f'auroc_{cls}']:.3f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    
    print("\n" + "="*80)
    print("🏆 FINAL RESULTS WITH TUNED HYPERPARAMETERS")
    print("="*80)
    
    df_results = pd.DataFrame(results)

    metric_keys = ['accuracy', 'balanced_accuracy',
                   'precision_macro', 'recall_macro', 'f1_macro',
                   'precision_weighted', 'recall_weighted', 'f1_weighted',
                   'aupr_macro', 'auroc_macro', 'auroc_weighted',
                   'cohen_kappa', 'mcc']
    per_class_keys = []
    for cls in ['CN', 'MCI', 'AD']:
        per_class_keys += [f'precision_{cls}', f'recall_{cls}', f'f1_{cls}',
                           f'aupr_{cls}', f'auroc_{cls}']

    print("\n--- MACRO / GLOBAL METRICS (mean ± std across folds) ---")
    for k in metric_keys:
        if k in df_results.columns:
            print(f"  {k:<25s}: {df_results[k].mean():.4f} ± {df_results[k].std():.4f}")

    print("\n--- PER-CLASS METRICS (mean ± std across folds) ---")
    for cls in ['CN', 'MCI', 'AD']:
        print(f"  [{cls}]")
        for suffix in ['precision', 'recall', 'f1', 'aupr', 'auroc']:
            k = f'{suffix}_{cls}'
            if k in df_results.columns:
                print(f"    {suffix:<12s}: {df_results[k].mean():.4f} ± {df_results[k].std():.4f}")

    print("\nPer-fold summary:")
    summary_cols = ['Fold', 'Val_Acc', 'accuracy', 'f1_macro', 'aupr_macro', 'auroc_macro', 'cohen_kappa', 'mcc']
    print(df_results[[c for c in summary_cols if c in df_results.columns]].to_string(index=False))
    
    output_dir = PROJECT_ROOT / "reports" / "tuning_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(output_dir / "best_params.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    df_results.to_csv(output_dir / "tuned_results.csv", index=False)
    
    import pickle
    with open(output_dir / "optuna_study.pkl", 'wb') as f:
        pickle.dump(study, f)
    
    print(f"\n✅ Results saved to: {output_dir}")
    
    baseline_acc = 0.6773
    baseline_aupr = 0.7304
    
    improvement_acc = (df_results['accuracy'].mean() - baseline_acc) * 100
    improvement_aupr = (df_results['aupr_macro'].mean() - baseline_aupr) * 100
    
    print("\n" + "="*80)
    print("📈 IMPROVEMENT vs BASELINE (Focal γ=1.0 without tuning)")
    print("="*80)
    print(f"Accuracy:  {improvement_acc:+.2f}% ({baseline_acc:.2%} → {df_results['accuracy'].mean():.2%})")
    print(f"AUPR:      {improvement_aupr:+.2f}% ({baseline_aupr:.2%} → {df_results['aupr_macro'].mean():.2%})")
    
    if improvement_aupr > 1.0:
        print("\n🎉 Tuning SUCCESSFUL! Significant improvement achieved.")
    elif improvement_aupr > 0:
        print("\n✅ Tuning provided modest improvement.")
    else:
        print("\n⚠️  No improvement - baseline params were already near-optimal.")

if __name__ == "__main__":
    main()
