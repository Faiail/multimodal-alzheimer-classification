import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef, average_precision_score)
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parents[1]))
from data.dataset import AlzheimerGraphDataset, HybridAlzheimerDataset 
from data.batch_utils import (build_graph_dataloader, fma_dataset_kwargs,
                              get_node_feature_dim, get_node_feature_dims, move_graph_batch_to_device)
from models.model import GCN_KAN_Classifier, GAT_KAN_Classifier, DualGraph_KAN_Classifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
CLINICAL_COLS = ['AGE', 'Gender_Male', 'PTEDUCAT', 'APOE4', 'MMSE', 'ABETA_final', 'TAU_final', 'PTAU_final', 'CSF_Available']

def load_config():
    with open(CONFIG_PATH, 'r') as f: return yaml.safe_load(f)

def prepare_data_cn_ad(cfg):
    print("--- Data Preparation (CN vs AD) ---")
    
    df_img = pd.read_csv(PROJECT_ROOT / cfg['data']['features_path'])
    df_img['PTID'] = df_img['subject_id'].apply(lambda x: "_".join(x.split('_')[:3]))
    
    clinical_path = PROJECT_ROOT / "data/02_processed/cn_ad_features.csv"
    if not clinical_path.exists():
        raise FileNotFoundError(f"{clinical_path} not found. Run src/data/create_cn_ad_dataset.py first!")
        
    df_clin = pd.read_csv(clinical_path)
    
    df_merged = pd.merge(df_img, df_clin, on='PTID', how='inner', suffixes=('', '_clin'))
    
    if 'label_clin' in df_merged.columns: df_merged['label'] = df_merged['label_clin']
    df_merged = df_merged.dropna(subset=['label']).reset_index(drop=True)
    
    img_cols = [c for c in df_merged.columns if 'GM_' in c or 'PET_' in c]
    df_merged[img_cols] = StandardScaler().fit_transform(df_merged[img_cols].fillna(0))
    df_merged[CLINICAL_COLS] = StandardScaler().fit_transform(df_merged[CLINICAL_COLS].fillna(0))
    
    print(f"Final Dataset CN vs AD: {len(df_merged)} subjects.")
    return df_merged

def run_xgboost(train_df, test_df, mode='multimodal'):
    img_cols = [c for c in train_df.columns if 'GM_' in c or 'PET_' in c]
    features = img_cols + CLINICAL_COLS if mode == 'multimodal' else img_cols
    clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
    clf.fit(train_df[features], train_df['label'])
    
    preds = clf.predict(test_df[features])
    probs = clf.predict_proba(test_df[features])
    truths = test_df['label'].values
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(truths, preds)
    metrics['balanced_accuracy'] = balanced_accuracy_score(truths, preds)
    metrics['precision'] = precision_score(truths, preds, zero_division=0)
    metrics['recall'] = recall_score(truths, preds, zero_division=0)
    metrics['f1'] = f1_score(truths, preds, zero_division=0)
    metrics['cohen_kappa'] = cohen_kappa_score(truths, preds)
    metrics['mcc'] = matthews_corrcoef(truths, preds)
    
    try:
        metrics['auroc'] = roc_auc_score(truths, probs[:, 1])
        metrics['aupr'] = average_precision_score(truths, probs[:, 1])
    except ValueError:
        metrics['auroc'] = 0.0
        metrics['aupr'] = 0.0
    
    prec_per = precision_score(truths, preds, average=None, zero_division=0)
    rec_per = recall_score(truths, preds, average=None, zero_division=0)
    f1_per = f1_score(truths, preds, average=None, zero_division=0)
    
    for c in range(2):
        metrics[f'precision_class_{c}'] = prec_per[c]
        metrics[f'recall_class_{c}'] = rec_per[c]
        metrics[f'f1_class_{c}'] = f1_per[c]
    
    cm = confusion_matrix(truths, preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

def run_deep_model(model_type, cfg, train_df_full, test_df, device, mode='multimodal', graph_type='static'):
    coords_path = PROJECT_ROOT / cfg['data']['coords_path']
    train_idx, val_idx = train_test_split(range(len(train_df_full)), test_size=0.15, stratify=train_df_full['label'], random_state=42)
    train_df = train_df_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_df_full.iloc[val_idx].reset_index(drop=True)

    ds_kwargs = {
        'sigma': cfg['model']['sigma_graph'],
        'k_neighbors': 20,
        'graph_type': graph_type,
        'graph_mix_alpha': cfg['model'].get('hybrid_alpha', 0.5),
        'node_semantics_mode': cfg['model'].get('node_semantics_mode', 'none')
    }

    ds_kwargs.update(fma_dataset_kwargs(cfg))

    if mode == 'multimodal':
        DatasetClass = HybridAlzheimerDataset
        ds_kwargs['clinical_cols'] = CLINICAL_COLS
        n_clin_feats = len(CLINICAL_COLS)
    else:
        DatasetClass = AlzheimerGraphDataset
        n_clin_feats = 0

    train_ds = DatasetClass(train_df, coords_path, **ds_kwargs)
    if graph_type in ('zhang', 'hybrid', 'multigraph'):
        zhang_stats = {'mean': train_ds.global_mean, 'std': train_ds.global_std}
        ds_kwargs['precomputed_stats'] = zhang_stats
    val_ds = DatasetClass(val_df, coords_path, **ds_kwargs)
    test_ds = DatasetClass(test_df, coords_path, **ds_kwargs)
    n_node_features = get_node_feature_dim(train_ds)
    hetero_metadata = train_ds.metadata() if train_ds.is_heterogeneous else None
    hetero_input_dims = get_node_feature_dims(train_ds)
    
    # option 2: calc on full dataset
    # training
    # if graph_type == 'zhang':
    # training
    # zhang_stats = {'mean': temp_ds.global_mean
    # ds_kwargs['precomputed_stats'] = zhang_stats
    # training
    # training
    # test_ds = datasetclass(test_df, coords_path, **ds_kwargs)
    
    y_train = train_df['label'].values.astype(int)
    weight_scaling = np.sqrt(len(y_train) / np.bincount(y_train))
    sampler = WeightedRandomSampler(torch.from_numpy(weight_scaling[y_train]).float(), len(y_train), replacement=True)
    
    train_loader = build_graph_dataloader(train_ds, batch_size=cfg['data']['batch_size'], sampler=sampler, shuffle=False)
    val_loader = build_graph_dataloader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False)
    test_loader = build_graph_dataloader(test_ds, batch_size=cfg['data']['batch_size'], shuffle=False)
    
    if model_type == 'DualGraph' and mode == 'imaging': return 0.0
    
    if model_type == 'GCN':
        model = GCN_KAN_Classifier(
            in_features=n_node_features,
            n_clinical_features=n_clin_feats,
            hidden_dim=64,
            dropout=0.4,
            graph_type=graph_type,
            hetero_metadata=hetero_metadata,
            hetero_input_dims=hetero_input_dims
        ).to(device)
    elif model_type == 'GAT':
        model = GAT_KAN_Classifier(
            in_features=n_node_features,
            n_clinical_features=n_clin_feats,
            hidden_dim=32,
            dropout=0.5,
            graph_type=graph_type,
            hetero_metadata=hetero_metadata,
            hetero_input_dims=hetero_input_dims
        ).to(device)
    elif model_type == 'DualGraph':
        model = DualGraph_KAN_Classifier(
            in_features=n_node_features,
            n_clinical_features=n_clin_feats,
            hidden_dim=32,
            dropout=0.5,
            graph_type=graph_type,
            hetero_metadata=hetero_metadata,
            hetero_input_dims=hetero_input_dims
        ).to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val_acc, patience, counter, best_state = 0.0, 15, 0, None
    for epoch in range(150):
        model.train()
        for batch in train_loader:
            if mode == 'multimodal':
                x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=True)
                out = model(x_g, adj, x_c)
            else:
                x_g, adj, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=False)
                out = model(x_g, adj, None)
            optimizer.zero_grad(); criterion(out, y).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            
        model.eval()
        val_preds, val_truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                if mode == 'multimodal':
                    x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=True)
                    out = model(x_g, adj, x_c)
                else:
                    x_g, adj, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=False)
                    out = model(x_g, adj, None)
                val_preds.extend(torch.argmax(out, dim=1).cpu().numpy()); val_truths.extend(y.cpu().numpy())
        
        acc = accuracy_score(val_truths, val_preds)
        if acc > best_val_acc: best_val_acc, best_state, counter = acc, model.state_dict().copy(), 0
        else: counter += 1
        if counter >= patience: break
    
    if best_state: model.load_state_dict(best_state)
    model.eval()
    preds, truths, probs_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            if mode == 'multimodal':
                x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=True)
                out = model(x_g, adj, x_c)
            else:
                x_g, adj, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=False)
                out = model(x_g, adj, None)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            truths.extend(y.cpu().numpy())
            probs_list.extend(torch.softmax(out, dim=1).cpu().numpy())
    
    truths = np.array(truths)
    preds = np.array(preds)
    probs = np.array(probs_list)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(truths, preds)
    metrics['balanced_accuracy'] = balanced_accuracy_score(truths, preds)
    metrics['precision'] = precision_score(truths, preds, zero_division=0)
    metrics['recall'] = recall_score(truths, preds, zero_division=0)
    metrics['f1'] = f1_score(truths, preds, zero_division=0)
    metrics['cohen_kappa'] = cohen_kappa_score(truths, preds)
    metrics['mcc'] = matthews_corrcoef(truths, preds)
    
    try:
        metrics['auroc'] = roc_auc_score(truths, probs[:, 1])
        metrics['aupr'] = average_precision_score(truths, probs[:, 1])
    except ValueError:
        metrics['auroc'] = 0.0
        metrics['aupr'] = 0.0
    
    prec_per = precision_score(truths, preds, average=None, zero_division=0)
    rec_per = recall_score(truths, preds, average=None, zero_division=0)
    f1_per = f1_score(truths, preds, average=None, zero_division=0)
    
    for c in range(2):
        metrics[f'precision_class_{c}'] = prec_per[c]
        metrics[f'recall_class_{c}'] = rec_per[c]
        metrics[f'f1_class_{c}'] = f1_per[c]
    
    cm = confusion_matrix(truths, preds)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

def main():
    cfg = load_config()
    try:
        df = prepare_data_cn_ad(cfg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedGroupKFold(n_splits=5)
    
    res = {'imaging': {'XGB': [], 'GCN': [], 'GAT': [], 'GAT-KNN': []},
           'multimodal': {'XGB': [], 'GCN': [], 'GAT': [], 'DualGraph': [], 'GAT-KNN': []}}
    
    print(f"\n--- BENCHMARK CN vs AD ---")
    for i, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'], df['PTID'])):
        print(f"\n📊 FOLD {i+1}/5")
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]
        
        print("[Imaging Only]")
        
        m_xgb = run_xgboost(tr_df, te_df, 'imaging')
        res['imaging']['XGB'].append(m_xgb)
        print(f"  XGBoost:   Acc={m_xgb['accuracy']:.3f}, F1={m_xgb['f1']:.3f}, AUPR={m_xgb['aupr']:.3f}")
        
        m_gcn = run_deep_model('GCN', cfg, tr_df, te_df, device, 'imaging', 'static')
        res['imaging']['GCN'].append(m_gcn)
        print(f"  GCN-KAN:   Acc={m_gcn['accuracy']:.3f}, F1={m_gcn['f1']:.3f}, AUPR={m_gcn['aupr']:.3f}")
        
        m_gat = run_deep_model('GAT', cfg, tr_df, te_df, device, 'imaging', 'static')
        res['imaging']['GAT'].append(m_gat)
        print(f"  GAT-KAN:   Acc={m_gat['accuracy']:.3f}, F1={m_gat['f1']:.3f}, AUPR={m_gat['aupr']:.3f}")
        
        m_gat_knn = run_deep_model('GAT', cfg, tr_df, te_df, device, 'imaging', 'knn')
        res['imaging']['GAT-KNN'].append(m_gat_knn)
        print(f"  GAT-KNN:   Acc={m_gat_knn['accuracy']:.3f}, F1={m_gat_knn['f1']:.3f}, AUPR={m_gat_knn['aupr']:.3f}")
        
        print("[Multimodal]")
        
        m_xgb = run_xgboost(tr_df, te_df, 'multimodal')
        res['multimodal']['XGB'].append(m_xgb)
        print(f"  XGBoost:   Acc={m_xgb['accuracy']:.3f}, F1={m_xgb['f1']:.3f}, AUPR={m_xgb['aupr']:.3f}")
        
        m_gat = run_deep_model('GAT', cfg, tr_df, te_df, device, 'multimodal', 'static')
        res['multimodal']['GAT'].append(m_gat)
        print(f"  GAT-KAN:   Acc={m_gat['accuracy']:.3f}, F1={m_gat['f1']:.3f}, AUPR={m_gat['aupr']:.3f}")
        
        m_gat_knn = run_deep_model('GAT', cfg, tr_df, te_df, device, 'multimodal', 'knn')
        res['multimodal']['GAT-KNN'].append(m_gat_knn)
        print(f"  GAT-KNN:   Acc={m_gat_knn['accuracy']:.3f}, F1={m_gat_knn['f1']:.3f}, AUPR={m_gat_knn['aupr']:.3f}")
        
        m_dual = run_deep_model('DualGraph', cfg, tr_df, te_df, device, 'multimodal', 'static')
        res['multimodal']['DualGraph'].append(m_dual)
        print(f"  DualGraph: Acc={m_dual['accuracy']:.3f}, F1={m_dual['f1']:.3f}, AUPR={m_dual['aupr']:.3f}")

    print("\n" + "="*60)
    print("🏆 RESULTS CN vs AD")
    print("="*60)
    
    def avg_metrics(metrics_list):
        if not metrics_list:
            return None
        keys = metrics_list[0].keys()
        avg = {}
        for k in keys:
            if k != 'confusion_matrix':
                vals = [m[k] for m in metrics_list if k in m]
                avg[k] = np.mean(vals) if vals else 0.0
        return avg
    
    print(f"{'Model':<12} | {'Mode':<12} | {'Acc':<12} | {'F1':<12} | {'AUPR':<12} | {'AUROC':<12} | {'κ':<12}")
    print("-" * 80)
    
    for m in ['XGB', 'GCN', 'GAT', 'GAT-KNN', 'DualGraph']:
        if m in res['imaging'] and len(res['imaging'][m]) > 0:
            avg = avg_metrics(res['imaging'][m])
            if avg:
                print(f"{m:<12} | {'Imaging':<12} | {avg['accuracy']:.3f}    | {avg['f1']:.3f}    | {avg['aupr']:.3f}    | {avg.get('auroc', 0):.3f}    | {avg['cohen_kappa']:.3f}")
        
        if m in res['multimodal'] and len(res['multimodal'][m]) > 0:
            avg = avg_metrics(res['multimodal'][m])
            if avg:
                print(f"{m:<12} | {'Multimodal':<12} | {avg['accuracy']:.3f}    | {avg['f1']:.3f}    | {avg['aupr']:.3f}    | {avg.get('auroc', 0):.3f}    | {avg['cohen_kappa']:.3f}")
    
    # best model detailed metrics
    print("\n" + "="*60)
    print("📊 BEST MODEL - PER-CLASS METRICS (Class 0=CN, Class 1=AD)")
    print("="*60)
    
    best_model, best_f1 = None, 0.0
    for m in res['multimodal']:
        if len(res['multimodal'][m]) > 0:
            avg = avg_metrics(res['multimodal'][m])
            if avg and avg['f1'] > best_f1:
                best_f1 = avg['f1']
                best_model = m
    
    if best_model:
        avg = avg_metrics(res['multimodal'][best_model])
        print(f"Model: {best_model} (Multimodal)")
        print(f"\n{'Class':<12} | {'Precision':<12} | {'Recall':<12} | {'F1':<12}")
        print("-" * 50)
        for c in range(2):
            class_label = 'CN' if c == 0 else 'AD'
            print(f"{class_label:<12} | {avg[f'precision_class_{c}']:.3f}      | {avg[f'recall_class_{c}']:.3f}      | {avg[f'f1_class_{c}']:.3f}")
    
    print("="*60)

if __name__ == "__main__":
    main()