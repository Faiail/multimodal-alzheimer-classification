import sys
import yaml
import torch
import numpy as np
import pandas as pd
import datetime
import wandb
import os
import re
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).parents[1]))
from data.dataset import AlzheimerGraphDataset, HybridAlzheimerDataset 
from data.batch_utils import (build_graph_dataloader, fma_dataset_kwargs,
                              get_node_feature_dim, get_node_feature_dims, move_graph_batch_to_device)
from models.model import GCN_KAN_Classifier, GAT_KAN_Classifier, DualGraph_KAN_Classifier
from experiment_config import get_experiment_schedule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
CLINICAL_COLS = ['AGE', 'Gender_Male', 'PTEDUCAT', 'APOE4', 'MMSE', 'ABETA_final', 'TAU_final', 'PTAU_final', 'CSF_Available']

def load_config():
    with open(CONFIG_PATH, 'r') as f: return yaml.safe_load(f)

def sanitize_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))

def save_predictions(ids, truths, probs, meta, run_dir):
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    fname = f"{sanitize_filename(meta['Model'])}_{sanitize_filename(meta.get('Arch', 'Tree'))}_{sanitize_filename(meta['Modality'])}_{sanitize_filename(meta['Graph'])}_{sanitize_filename(meta['Data'])}_F{meta['Fold']}.csv"
    
    df = pd.DataFrame({
        'PTID': ids,
        'True_Label': truths,
        'Pred_Prob': probs
    })
    df.to_csv(pred_dir / fname, index=False)

def update_summary_csv(results_list, run_dir):
    df = pd.DataFrame(results_list)
    df.to_csv(run_dir / "detailed_results_partial.csv", index=False)

def prepare_standard_data(cfg):
    print("Loading Standard ROI Data...")
    df_img = pd.read_csv(PROJECT_ROOT / cfg['data']['features_path'])
    df_img['PTID'] = df_img['subject_id'].apply(lambda x: "_".join(x.split('_')[:3]))
    
    clin_path = PROJECT_ROOT / "data/processed/clinical_features.csv"
    if not clin_path.exists(): clin_path = PROJECT_ROOT / "data/02_processed/clinical_features.csv"
    df_clin = pd.read_csv(clin_path)
    
    df_merged = pd.merge(df_img, df_clin, on='PTID', how='inner', suffixes=('', '_clin'))
    if 'label_clin' in df_merged.columns: df_merged['label'] = df_merged['label_clin']
    df_merged = df_merged.dropna(subset=['label']).reset_index(drop=True)
    
    img_cols = [c for c in df_merged.columns if 'GM_' in c or 'PET_' in c]
    scaler = StandardScaler()
    df_merged[img_cols] = scaler.fit_transform(df_merged[img_cols].fillna(0))
    df_merged[CLINICAL_COLS] = scaler.fit_transform(df_merged[CLINICAL_COLS].fillna(0))
    return df_merged

def prepare_zhang_data():
    print("Loading Zhang 54k Data (for XGB)...")
    paths = [PROJECT_ROOT / "data/02_processed/zhang_features.csv.gz", PROJECT_ROOT / "data/processed/zhang_features.csv.gz"]
    zhang_path = next((p for p in paths if p.exists()), None)
    
    if not zhang_path: 
        print("⚠️  Zhang features file not found - skipping")
        return None, None
    
    try:
        df_zhang = pd.read_csv(zhang_path)
        df_zhang['PTID'] = df_zhang['subject_id'].apply(lambda x: "_".join(str(x).split('_')[:3]))
        
        clin_path = PROJECT_ROOT / "data/processed/clinical_features.csv"
        if not clin_path.exists(): clin_path = PROJECT_ROOT / "data/02_processed/clinical_features.csv"
        df_clin = pd.read_csv(clin_path)
        
        df_merged = pd.merge(df_zhang, df_clin[['PTID', 'label'] + CLINICAL_COLS], on='PTID', how='inner')
        df_merged = df_merged.dropna(subset=['label']).reset_index(drop=True)
        
        cols = [c for c in df_merged.columns if c.startswith('IBN_')]
        return df_merged, cols
    except Exception as e:
        print(f"⚠️  Error loading Zhang features: {e}")
        print("⚠️  Skipping Zhang features (only affects XGBoost experiments)")
        return None, None

def run_xgboost_exp(tr_df, te_df, exp_config, fold_idx, run_dir, zhang_cols=None):
    if exp_config['Data'] == 'Zhang':
        if zhang_cols is None: return 0.0, 0.0
        feature_cols = zhang_cols
    else:
        feature_cols = [c for c in tr_df.columns if 'GM_' in c or 'PET_' in c]
        
    if exp_config['Modality'] == 'Multimodal':
        feature_cols = feature_cols + CLINICAL_COLS
    
    run_name = f"XGB_{exp_config['Data']}_{exp_config['Modality']}_F{fold_idx}"
    wandb.init(project="Alzheimer-KAN-Benchmark", group="XGBoost", name=run_name, reinit=True, config=exp_config)

    clf = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1, tree_method='hist')
    clf.fit(tr_df[feature_cols], tr_df['label'])
    
    preds = clf.predict(te_df[feature_cols])
    probs = clf.predict_proba(te_df[feature_cols])[:, 1]
    truths = te_df['label'].values
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(truths, preds)
    metrics['balanced_accuracy'] = balanced_accuracy_score(truths, preds)
    metrics['precision'] = precision_score(truths, preds, zero_division=0)
    metrics['recall'] = recall_score(truths, preds, zero_division=0)
    metrics['f1'] = f1_score(truths, preds, zero_division=0)
    metrics['cohen_kappa'] = cohen_kappa_score(truths, preds)
    metrics['mcc'] = matthews_corrcoef(truths, preds)
    
    try:
        metrics['auroc'] = roc_auc_score(truths, probs)
        metrics['aupr'] = average_precision_score(truths, probs)
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
    
    wandb.log({"test_acc": metrics['accuracy'], "test_aupr": metrics['aupr'], "test_f1": metrics['f1']})
    wandb.finish()
    
    meta = {'Model': 'XGB', 'Arch': 'Tree', 'Modality': exp_config['Modality'], 
            'Graph': exp_config['Graph'], 'Fold': fold_idx, 'Data': exp_config['Data']}
    save_predictions(te_df['PTID'].values, truths, probs, meta, run_dir)
    
    return metrics

def run_dl_model(m_name, config_type, graph_type, modality, cfg, train_df, test_df, device, fold_idx, run_dir):
    coords_path = PROJECT_ROOT / cfg['data']['coords_path']
    ds_kwargs = {
        'sigma': cfg['model']['sigma_graph'],
        'k_neighbors': 20,
        'graph_type': graph_type,
        'graph_mix_alpha': cfg['model'].get('hybrid_alpha', 0.5),
        'node_semantics_mode': cfg['model'].get('node_semantics_mode', 'none')
    }
    
    ds_kwargs.update(fma_dataset_kwargs(cfg))

    if modality == 'Multimodal':
        DatasetClass = HybridAlzheimerDataset
        ds_kwargs['clinical_cols'] = CLINICAL_COLS
        n_clin = len(CLINICAL_COLS)
    else:
        DatasetClass = AlzheimerGraphDataset
        n_clin = 0

    tr_idx, val_idx = train_test_split(range(len(train_df)), test_size=0.15, stratify=train_df['label'], random_state=42)
    tr_sub = train_df.iloc[tr_idx].reset_index(drop=True)
    val_sub = train_df.iloc[val_idx].reset_index(drop=True)

    if graph_type in ('zhang', 'hybrid', 'multigraph'):
        temp_ds = DatasetClass(train_df, coords_path, precompute_adjacency=False, **ds_kwargs)
        zhang_stats = {'mean': temp_ds.global_mean, 'std': temp_ds.global_std}
        ds_kwargs['precomputed_stats'] = zhang_stats
        print(f"📊 Zhang statistics calculated on the training set")
    train_ds = DatasetClass(tr_sub, coords_path, **ds_kwargs)
    val_ds = DatasetClass(val_sub, coords_path, **ds_kwargs)
    test_ds = DatasetClass(test_df, coords_path, **ds_kwargs)
    n_node_features = get_node_feature_dim(train_ds)
    hetero_metadata = train_ds.metadata() if train_ds.is_heterogeneous else None
    hetero_input_dims = get_node_feature_dims(train_ds)
    
    y_tr = tr_sub['label'].values.astype(int)
    w = np.sqrt(len(y_tr) / (np.bincount(y_tr) + 1e-6))
    sampler = WeightedRandomSampler(torch.from_numpy(w[y_tr]).float(), len(y_tr), replacement=True)
    
    tr_loader = build_graph_dataloader(train_ds, batch_size=32, sampler=sampler)
    val_loader = build_graph_dataloader(val_ds, batch_size=32, shuffle=False)
    te_loader = build_graph_dataloader(test_ds, batch_size=32, shuffle=False)
    
    if config_type == 'Base':         use_bb, use_hd = False, False
    elif config_type == 'PartialKAN': use_bb, use_hd = False, True
    elif config_type == 'FullKAN':    use_bb, use_hd = True, True
    else:
        # fallback di sicurezza
        print(f"⚠️ Warning: Unknown config {config_type}, defaulting to Base.")
        use_bb, use_hd = False, False
    
    # model init
    if m_name == 'GCN': model = GCN_KAN_Classifier(n_node_features, n_clin, 32, 2, 0.4, use_bb, use_hd, graph_type=graph_type, hetero_metadata=hetero_metadata, hetero_input_dims=hetero_input_dims)
    elif m_name == 'GAT': model = GAT_KAN_Classifier(n_node_features, n_clin, 32, 2, 0.5, use_bb, use_hd, graph_type=graph_type, hetero_metadata=hetero_metadata, hetero_input_dims=hetero_input_dims)
    elif m_name == 'DualGraph': model = DualGraph_KAN_Classifier(n_node_features, n_clin, 32, 2, 0.5, use_bb, use_hd, graph_type=graph_type, hetero_metadata=hetero_metadata, hetero_input_dims=hetero_input_dims)
    
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    
    run_name = f"{m_name}_{config_type}_{modality}_{graph_type}_F{fold_idx}"
    wandb.init(project="Alzheimer-KAN-Benchmark", group=f"{m_name}-{config_type}", name=run_name, reinit=True,
               config={"model": m_name, "arch": config_type, "mod": modality, "graph": graph_type, "fold": fold_idx})
    
    best_acc, patience, counter, best_state = 0.0, 15, 0, None
    for ep in range(100):
        model.train()
        t_loss = 0
        for batch in tr_loader:
            if modality == 'Multimodal':
                x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=True)
            else:
                x_g, adj, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=False)
                x_c = None
            optim.zero_grad(); loss = crit(model(x_g, adj, x_c), y); loss.backward(); optim.step()
            t_loss += loss.item()
            
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in val_loader:
                if modality == 'Multimodal':
                    x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=True)
                else:
                    x_g, adj, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=False)
                    x_c = None
                preds.extend(torch.argmax(model(x_g, adj, x_c), 1).cpu().numpy())
                truths.extend(y.cpu().numpy())
        
        val_acc = accuracy_score(truths, preds)
        wandb.log({"epoch": ep, "train_loss": t_loss/len(tr_loader), "val_acc": val_acc})
        
        if val_acc > best_acc: best_acc = val_acc; best_state = model.state_dict().copy(); counter = 0
        else: counter += 1; 
        if counter >= patience: break
            
    if best_state: model.load_state_dict(best_state)
    model.eval()
    preds, probs_list, truths = [], [], []
    with torch.no_grad():
        for batch in te_loader:
            if modality == 'Multimodal':
                x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=True)
            else:
                x_g, adj, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=False)
                x_c = None
            out = model(x_g, adj, x_c)
            probs_full = torch.softmax(out, 1)
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            probs_list.extend(probs_full[:, 1].cpu().numpy())
            truths.extend(y.cpu().numpy())
    
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
        metrics['auroc'] = roc_auc_score(truths, probs)
        metrics['aupr'] = average_precision_score(truths, probs)
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
    
    wandb.log({"test_acc": metrics['accuracy'], "test_aupr": metrics['aupr'], "test_f1": metrics['f1']})
    wandb.finish()
    
    meta = {'Model': m_name, 'Arch': config_type, 'Modality': modality, 'Graph': graph_type, 'Fold': fold_idx, 'Data': 'Std'}
    save_predictions(test_df['PTID'].values, truths, probs, meta, run_dir)
    
    return metrics

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    skf = StratifiedGroupKFold(n_splits=5)
    
    df_std = prepare_standard_data(cfg)
    df_zhang, zhang_cols = prepare_zhang_data()
    
    experiment_list = get_experiment_schedule()
    
    TEST_MODE = False
    n_folds = 1 if TEST_MODE else 5
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "reports" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Results & Predictions will be saved to: {run_dir}")
    if TEST_MODE:
        print(f"⚡ TEST MODE: {len(experiment_list)} Configs x {n_folds} Fold (fast validation)")
    else:
        print(f"🚀 STARTING BENCHMARK: {len(experiment_list)} Configs x {n_folds} Folds 🚀")
    
    results_log = []
    
    try:
        for i, (tr_idx, te_idx) in enumerate(skf.split(df_std, df_std['label'], df_std['PTID'])):
            if i >= n_folds:
                break
            print(f"\n📊 FOLD {i+1}/{n_folds}")
            tr_std, te_std = df_std.iloc[tr_idx], df_std.iloc[te_idx]
            
            tr_z, te_z = None, None
            if df_zhang is not None:
                tr_z = df_zhang[df_zhang['PTID'].isin(tr_std['PTID'])]
                te_z = df_zhang[df_zhang['PTID'].isin(te_std['PTID'])]

            for exp in experiment_list:
                try:
                    if exp['Family'] == 'XGB':
                        if exp['Data'] == 'Zhang' and tr_z is not None:
                            metrics = run_xgboost_exp(tr_z, te_z, exp, i+1, run_dir, zhang_cols)
                        elif exp['Data'] == 'Std':
                            metrics = run_xgboost_exp(tr_std, te_std, exp, i+1, run_dir)
                        else:
                            continue
                    elif exp['Family'] == 'DL':
                        metrics = run_dl_model(exp['Model'], exp['Arch'], exp['Graph'], exp['Modality'], cfg, tr_std, te_std, device, i+1, run_dir)
                    
                    m = metrics
                    print(f"  ✅ {exp['Model']}-{exp.get('Arch', '')} ({exp['Graph']}): Acc={m['accuracy']:.3f}, F1={m['f1']:.3f}, AUPR={m['aupr']:.3f}, κ={m['cohen_kappa']:.3f}")
                    
                    res_entry = exp.copy()
                    if 'Flags' in res_entry: del res_entry['Flags']
                    res_entry['Fold'] = i+1
                    for k, v in metrics.items():
                        if k != 'confusion_matrix':
                            res_entry[k] = v
                    res_entry['confusion_matrix'] = str(metrics['confusion_matrix'])
                    results_log.append(res_entry)
                    update_summary_csv(results_log, run_dir)
                    
                except Exception as e:
                    print(f"❌ Error in {exp['Model']}: {e}")
                    import traceback
                    traceback.print_exc()
                    wandb.finish()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Saving current progress...")
    
    print("\n\n" + "="*100)
    print("🏆 FINAL BENCHMARK SUMMARY")
    print("="*100)
    
    if results_log:
        df_res = pd.DataFrame(results_log)
        df_res.to_csv(run_dir / "detailed_results_final.csv", index=False)
        
        if 'Status' in df_res.columns:
            df_success = df_res[df_res['Status'] == 'Success'].copy()
        else:
            df_success = df_res.dropna(subset=['accuracy', 'aupr']).copy()
        
        if len(df_success) > 0:
            group_cols = ['Model', 'Arch', 'Modality', 'Graph', 'Data']
            metric_cols = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1',
                          'auroc', 'aupr', 'cohen_kappa', 'mcc']
            for c in range(2):
                metric_cols += [f'precision_class_{c}', f'recall_class_{c}', f'f1_class_{c}']
            metric_cols = [c for c in metric_cols if c in df_success.columns]
            
            summary = df_success.groupby(group_cols)[metric_cols].agg(['mean', 'std', 'count']).reset_index()
            summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
            summary = summary.sort_values('accuracy_mean', ascending=False)
            summary.to_csv(run_dir / "summary_results.csv", index=False)
        else:
            print("⚠️ No successful experiments to summarize")
            summary = pd.DataFrame()
        
        print(f"{'Model':<12} | {'Arch':<12} | {'Modal':<12} | {'Graph':<12} | {'Acc':<12} | {'F1':<12} | {'AUPR':<12} | {'AUROC':<12} | {'κ':<12} | {'N':<3}")
        print("-" * 120)
        for _, row in summary.iterrows():
            acc_str = f"{row['accuracy_mean']:.3f}±{row.get('accuracy_std', 0):.2f}"
            f1_str = f"{row['f1_mean']:.3f}±{row.get('f1_std', 0):.2f}"
            aupr_str = f"{row['aupr_mean']:.3f}±{row.get('aupr_std', 0):.2f}"
            auroc_str = f"{row.get('auroc_mean', 0):.3f}±{row.get('auroc_std', 0):.2f}"
            kappa_str = f"{row.get('cohen_kappa_mean', 0):.3f}"
            n = int(row.get('accuracy_count', 0))
            print(f"{row['Model']:<12} | {row['Arch']:<12} | {row['Modality']:<12} | {row['Graph']:<12} | {acc_str:<12} | {f1_str:<12} | {aupr_str:<12} | {auroc_str:<12} | {kappa_str:<7} | {n:<3}")
    else:
        print("No results collected.")
    print("="*100)

if __name__ == "__main__":
    main()