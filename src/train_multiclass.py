import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import (accuracy_score, classification_report, average_precision_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef)

sys.path.append(str(Path(__file__).parents[1]))
from data.dataset import AlzheimerGraphDataset, HybridAlzheimerDataset
from data.preprocessing import FoldFeatureNormalizer, imaging_feature_columns
from data.batch_utils import (build_graph_dataloader, fma_dataset_kwargs,
                              get_node_feature_dim, get_node_feature_dims, move_graph_batch_to_device)
from models.model import GCN_KAN_Classifier, GAT_KAN_Classifier, DualGraph_KAN_Classifier
from models.layers import FocalLoss
from experiment_config import get_experiment_schedule, LOSS_CONFIGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model_config.yaml"
CLINICAL_COLS = ['AGE', 'Gender_Male', 'PTEDUCAT', 'APOE4', 'MMSE', 'ABETA_final', 'TAU_final', 'PTAU_final', 'CSF_Available']

def load_config():
    with open(CONFIG_PATH, 'r') as f: return yaml.safe_load(f)

def prepare_multiclass_data():
    print("--- Data Preparation (Multiclass CN vs MCI vs AD) ---")

    cfg = load_config()
    df_img = pd.read_csv(PROJECT_ROOT / cfg['data']['features_path'])
    df_img['PTID'] = df_img['subject_id'].apply(lambda x: "_".join(x.split('_')[:3]))

    multiclass_path = PROJECT_ROOT / "data" / "02_processed" / "multiclass_cn_mci_ad_features.csv"
    if not multiclass_path.exists():
        raise FileNotFoundError(f"{multiclass_path} not found. Run src/data/create_multiclass_cn_mci_ad_dataset.py first!")

    df_clin = pd.read_csv(multiclass_path)

    df_merged = pd.merge(df_img, df_clin, on='PTID', how='inner', suffixes=('', '_clin'))
    if 'label_clin' in df_merged.columns: df_merged['label'] = df_merged['label_clin']
    df_merged = df_merged.dropna(subset=['label']).reset_index(drop=True)

    print(f"✅ Multiclass dataset loaded: {len(df_merged)} patients")

    class_dist = df_merged['label'].value_counts().sort_index()
    print("📊 Class distribution:")
    for label, count in class_dist.items():
        class_name = {0: 'CN', 1: 'MCI', 2: 'AD'}[label]
        percentage = count / len(df_merged) * 100
        print(f"      {class_name}: {count} patients ({percentage:.1f}%)")

    img_cols = imaging_feature_columns(df_merged)
    for col in img_cols:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    clin_cols_present = [c for c in CLINICAL_COLS if c in df_merged.columns]
    for col in clin_cols_present:
        df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce')

    print(
        f"✅ Prepared {len(img_cols)} imaging and {len(clin_cols_present)} clinical "
        "columns; normalization will be fitted inside each training fold"
    )

    print(f"Final Multiclass Dataset: {len(df_merged)} subjects.")
    return df_merged

def run_deep_model(model_type, cfg, train_df_full, test_df, device, mode='multimodal', graph_type='static', arch_flags=None, zhang_reference='all_train', loss_config='cross_entropy', n_layers=2, multigraph_fusion='hetero_sum'):
    if multigraph_fusion != 'hetero_sum':
        raise ValueError("multigraph_fusion must be 'hetero_sum'")

    if arch_flags is None:
        arch_flags = {'use_kan_backbone': False, 'use_kan_head': False}

    model_aliases = {
        'gcn': 'GCN',
        'gat': 'GAT',
        'dualgraph': 'DualGraph',
    }
    model_key = str(model_type).strip().lower()
    if model_key not in model_aliases:
        raise ValueError(
            f"Invalid model_type '{model_type}'. "
            f"Must be one of {list(model_aliases.values())}"
        )
    model_type = model_aliases[model_key]
    
    valid_zhang_refs = ['all_train', 'cn_only', 'mci_only']
    if zhang_reference not in valid_zhang_refs:
        raise ValueError(f"Invalid zhang_reference '{zhang_reference}'. Must be one of {valid_zhang_refs}")
    
    if zhang_reference != 'all_train' and graph_type not in ('zhang', 'hybrid', 'multigraph'):
        raise ValueError(
            f"zhang_reference '{zhang_reference}' can only be used with graph_type in ('zhang', 'hybrid', 'multigraph'), but got '{graph_type}'"
        )
    
    coords_path = PROJECT_ROOT / cfg['data']['coords_path']
    ds_kwargs = {
        'sigma': cfg['model']['sigma_graph'],
        'k_neighbors': 20,
        'graph_type': graph_type,
        'graph_mix_alpha': cfg['model'].get('hybrid_alpha', 0.5),
        'node_semantics_mode': cfg['model'].get('node_semantics_mode', 'none'),
        'codeviation_threshold': cfg['model'].get('codeviation_threshold', 0.9),
        'codeviation_kernel_sigma': cfg['model'].get(
            'codeviation_kernel_sigma', np.sqrt(2.0)
        ),
    }

    ds_kwargs.update(fma_dataset_kwargs(cfg))

    if mode == 'multimodal':
        DatasetClass = HybridAlzheimerDataset
        ds_kwargs['clinical_cols'] = CLINICAL_COLS
        n_clin_feats = len(CLINICAL_COLS)
    else:
        DatasetClass = AlzheimerGraphDataset
        n_clin_feats = 0

    train_idx, val_idx = train_test_split(range(len(train_df_full)), test_size=0.15, stratify=train_df_full['label'], random_state=42)
    tr_sub = train_df_full.iloc[train_idx].reset_index(drop=True)
    val_sub = train_df_full.iloc[val_idx].reset_index(drop=True)

    imaging_cols = imaging_feature_columns(tr_sub)
    clinical_cols = [column for column in CLINICAL_COLS if column in tr_sub.columns]
    normalizer = FoldFeatureNormalizer(imaging_cols, clinical_cols)
    tr_sub, val_sub, test_df = normalizer.fit_transform_many(
        tr_sub, val_sub, test_df.reset_index(drop=True)
    )
    print(
        f"✅ Fold-safe normalization fitted on {len(tr_sub)} training samples: "
        f"{len(imaging_cols)} imaging (GM/WM/CSF/PET), "
        f"{len(clinical_cols)} clinical features"
    )

    if graph_type in ('zhang', 'hybrid', 'multigraph'):
        if zhang_reference == 'all_train':
            reference_df = tr_sub
            ref_label = "whole training set"
        elif zhang_reference == 'cn_only':
            reference_df = tr_sub[tr_sub['label'] == 0].reset_index(drop=True)
            if len(reference_df) == 0:
                raise ValueError("No CN samples found in training set for zhang_reference='cn_only'")
            ref_label = "only CN patients (healthy controls)"
        elif zhang_reference == 'mci_only':
            reference_df = tr_sub[tr_sub['label'] == 1].reset_index(drop=True)
            if len(reference_df) == 0:
                raise ValueError("No MCI samples found in training set for zhang_reference='mci_only'")
            ref_label = "only MCI patients"
        
        reference_ds = DatasetClass(
            reference_df, coords_path, precompute_adjacency=False, **ds_kwargs
        )
        zhang_stats = {'mean': reference_ds.global_mean, 'std': reference_ds.global_std}
        ds_kwargs['precomputed_stats'] = zhang_stats
        print(f"📊 Zhang statistics calculated on {ref_label} ({len(reference_df)} patients)")

    train_ds = DatasetClass(tr_sub, coords_path, **ds_kwargs)
    val_ds = DatasetClass(val_sub, coords_path, **ds_kwargs)
    test_ds = DatasetClass(test_df, coords_path, **ds_kwargs)
    n_node_features = get_node_feature_dim(train_ds)
    hetero_metadata = train_ds.metadata() if train_ds.is_heterogeneous else None
    hetero_input_dims = get_node_feature_dims(train_ds)

    y_train = tr_sub['label'].values.astype(int)
    class_counts = np.bincount(y_train)
    weight_scaling = np.sqrt(len(y_train) / class_counts)
    sampler = WeightedRandomSampler(torch.from_numpy(weight_scaling[y_train]).float(), len(y_train), replacement=True)

    num_workers = 0
    
    loader_kwargs = {
        'batch_size': cfg['data']['batch_size'],
        'num_workers': num_workers,
        'pin_memory': True if device.type == 'cuda' else False,
        'persistent_workers': num_workers > 0
    }
    
    train_loader = build_graph_dataloader(train_ds, sampler=sampler, shuffle=False, **loader_kwargs)
    val_loader = build_graph_dataloader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = build_graph_dataloader(test_ds, shuffle=False, **loader_kwargs)
    
    print(f"    DataLoader: batch_size={loader_kwargs['batch_size']}, num_workers={num_workers}, pin_memory={loader_kwargs['pin_memory']}")

    if model_type == 'DualGraph' and mode == 'imaging': return 0.0, {}

    use_bb = arch_flags.get('use_kan_backbone', False)
    use_hd = arch_flags.get('use_kan_head', False)

    hidden_dim = cfg['model']['hidden_dim']
    dropout    = cfg['model']['dropout']
    lr         = cfg['model']['learning_rate']

    if model_type == 'GCN':
        model = GCN_KAN_Classifier(
            in_features=n_node_features,
            n_clinical_features=n_clin_feats,
            hidden_dim=hidden_dim,
            n_classes=3,
            dropout=dropout,
            use_kan_backbone=use_bb,
            use_kan_head=use_hd,
            n_layers=n_layers,
            graph_type=graph_type,
            hetero_metadata=hetero_metadata,
            hetero_input_dims=hetero_input_dims
        ).to(device)
    elif model_type == 'GAT':
        model = GAT_KAN_Classifier(
            in_features=n_node_features,
            n_clinical_features=n_clin_feats,
            hidden_dim=hidden_dim,
            n_classes=3,
            dropout=dropout,
            use_kan_backbone=use_bb,
            use_kan_head=use_hd,
            n_layers=n_layers,
            graph_type=graph_type,
            hetero_metadata=hetero_metadata,
            hetero_input_dims=hetero_input_dims
        ).to(device)
    elif model_type == 'DualGraph':
        model = DualGraph_KAN_Classifier(
            in_features=n_node_features,
            n_clinical_features=n_clin_feats,
            hidden_dim=hidden_dim,
            n_classes=3,
            dropout=dropout,
            use_kan_backbone=use_bb,
            use_kan_head=use_hd,
            n_layers=n_layers,
            graph_type=graph_type,
            hetero_metadata=hetero_metadata,
            hetero_input_dims=hetero_input_dims
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    depth_info = f", n_layers={n_layers}" if model_type in ('GCN', 'GAT') else ""
    print(f"    Model: {model_type}{depth_info}, hidden_dim={hidden_dim}, KAN_backbone={use_bb}, KAN_head={use_hd}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    
    if loss_config in LOSS_CONFIGS:
        loss_params = LOSS_CONFIGS[loss_config]
        if loss_params['type'] == 'focal':
            alpha_cfg = loss_params.get('alpha', None)
            if alpha_cfg is None:
                alpha_tensor = None
                alpha_info = "None"
            elif alpha_cfg == 'balanced':
                n_classes = len(class_counts)
                alpha_weights = len(y_train) / (n_classes * class_counts)
                print(f"    Class counts: {class_counts}, Alpha weights (balanced): {alpha_weights.round(3)}")
                alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32).to(device)
                alpha_info = f"balanced {alpha_weights.round(3)}"
            elif alpha_cfg == 'sqrt_balanced':
                n_classes = len(class_counts)
                alpha_weights = np.sqrt(len(y_train) / (n_classes * class_counts))
                print(f"    Class counts: {class_counts}, Alpha weights (sqrt_balanced): {alpha_weights.round(3)}")
                alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32).to(device)
                alpha_info = f"sqrt_balanced {alpha_weights.round(3)}"
            elif isinstance(alpha_cfg, (list, tuple)):
                alpha_tensor = torch.tensor(alpha_cfg, dtype=torch.float32).to(device)
                alpha_info = f"manual {alpha_cfg}"
            else:
                raise ValueError(f"Invalid alpha config: {alpha_cfg}. Use None, 'balanced', 'sqrt_balanced', or a list.")
            
            criterion = FocalLoss(alpha=alpha_tensor, gamma=loss_params['gamma'])
            print(f"    Using Focal Loss (γ={loss_params['gamma']}, α={alpha_info})")
        else:
            criterion = torch.nn.CrossEntropyLoss()
            print(f"    Using Cross Entropy Loss")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        print(f"    Using Cross Entropy Loss")

    max_epochs = int(cfg['model'].get('epochs', 150))
    patience = int(cfg['model'].get('early_stopping_patience', 15))
    best_val_acc, counter, best_state = 0.0, 0, None
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            if mode == 'multimodal':
                x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=True)
                out = model(x_g, adj, x_c)
            else:
                x_g, adj, y = move_graph_batch_to_device(batch, device, graph_type=graph_type, has_clinical=False)
                out = model(x_g, adj, None)
            optimizer.zero_grad(); criterion(out, y).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        
        if epoch == 0 and device.type == 'cuda':
            print(f"    [Epoch {epoch+1}] GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB / {torch.cuda.max_memory_allocated(device)/1e9:.2f}GB allocated")
            torch.cuda.reset_peak_memory_stats(device)

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

    return metrics

def main():
    cfg = load_config()
    try:
        df = prepare_multiclass_data()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    print("\n🔍 GPU Diagnostics:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        requested_gpu = 0
        if requested_gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{requested_gpu}")
            print(f"  ✅ Using GPU {requested_gpu}: {torch.cuda.get_device_name(requested_gpu)}")
        else:
            print(f"  ⚠️  GPU {requested_gpu} not available, using GPU 0")
            device = torch.device("cuda:0")
    else:
        print("  ⚠️  CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    if device.type == 'cuda':
        test_tensor = torch.randn(100, 100).to(device)
        _ = test_tensor @ test_tensor
        torch.cuda.synchronize()
        print(f"  ✅ GPU test passed - tensors can be moved to {device}")
        del test_tensor
    
    n_splits = int(cfg['experiment'].get('n_splits', 5))
    max_folds = int(cfg['experiment'].get('max_folds', n_splits))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("\n🔍 Verifying StratifiedGroupKFold:")
    fold_class_distributions = []
    for i, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'], df['PTID'])):
        tr_labels = df.iloc[tr_idx]['label'].values
        te_labels = df.iloc[te_idx]['label'].values

        tr_dist = np.bincount(tr_labels, minlength=3)
        te_dist = np.bincount(te_labels, minlength=3)

        fold_class_distributions.append((tr_dist, te_dist))
        print(f"  Fold {i+1}: Train {tr_dist} | Test {te_dist}")

    all_experiments = get_experiment_schedule(
        test_mode=bool(cfg['experiment'].get('test_mode', False))
    )
    experiments = [exp for exp in all_experiments if exp['Family'] == 'DL']

    print(f"\n📋 Loaded DL experiments: {len(experiments)}")
    for exp in experiments[:5]:  # show first 5
        loss_info = f", Loss={exp.get('Loss', 'cross_entropy')}" if 'Loss' in exp else ""
        print(f"  • {exp['Model']}-{exp['Arch']} ({exp['Modality']}, {exp['Graph']}{loss_info})")

    results_log = []

    print(f"\n--- BENCHMARK MULTICLASSE CN vs MCI vs AD ---")
    for i, (tr_idx, te_idx) in enumerate(skf.split(df, df['label'], df['PTID'])):
        if i >= max_folds:
            break
        print(f"\n📊 FOLD {i+1}/{min(n_splits, max_folds)}")
        tr_df, te_df = df.iloc[tr_idx], df.iloc[te_idx]

        for exp in experiments:
            model_name = exp['Model']
            arch_name = exp['Arch']
            modality = exp['Modality'].lower()
            graph_type = exp['Graph']
            arch_flags = exp['Flags']
            zhang_ref = exp.get('ZhangRef', 'all_train')
            loss_config = exp.get('Loss', 'cross_entropy')
            n_layers = exp.get('NLayers', 2)
            multigraph_fusion = exp.get('MultigraphFusion', 'hetero_sum')

            if model_name == 'DualGraph' and modality == 'imaging':
                continue

            loss_label = LOSS_CONFIGS.get(loss_config, {}).get('type', 'CE')
            layers_info = f", L={n_layers}" if model_name in ('GCN', 'GAT') else ""
            fusion_info = f", fusion={multigraph_fusion}" if graph_type == 'multigraph' else ""
            print(f"  Training {model_name}-{arch_name}{layers_info} ({modality}, {graph_type}{fusion_info}, loss={loss_label})...")

            try:
                metrics = run_deep_model(model_name, cfg, tr_df, te_df, device, modality, graph_type, arch_flags, zhang_ref, loss_config, n_layers, multigraph_fusion)
                
                m = metrics
                print(f"  ✅ {model_name}-{arch_name} ({graph_type}, loss={loss_label})")
                print(f"      Macro: Acc={m['accuracy']:.3f}, BalAcc={m['balanced_accuracy']:.3f}, F1={m['f1_macro']:.3f}, AUPR={m['aupr_macro']:.3f}, AUROC={m['auroc_macro']:.3f}")
                print(f"      Per-class F1: CN={m['f1_CN']:.3f}, MCI={m['f1_MCI']:.3f}, AD={m['f1_AD']:.3f}")
                print(f"      Cohen κ={m['cohen_kappa']:.3f}, MCC={m['mcc']:.3f}")
                
                res_entry = exp.copy()
                if 'Flags' in res_entry: del res_entry['Flags']
                res_entry['Fold'] = i+1
                for k, v in metrics.items():
                    if k != 'confusion_matrix':
                        res_entry[k] = v
                res_entry['confusion_matrix'] = str(metrics['confusion_matrix'])
                results_log.append(res_entry)
            except ValueError as e:
                print(f"  ❌ Configuration error: {e}")
                continue
            except Exception as e:
                print(f"  ❌ Training error: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n\n" + "="*100)
    print("🏆 FINAL BENCHMARK SUMMARY")
    print("="*100)
    
    if results_log:
        df_res = pd.DataFrame(results_log)
        
        df_success = df_res.dropna(subset=['accuracy', 'aupr_macro']).copy()
        
        if len(df_success) > 0:
            group_cols = ['Model', 'Arch', 'Modality', 'Graph']
            if 'NLayers' in df_success.columns:
                group_cols.append('NLayers')
            if 'ZhangRef' in df_success.columns:
                df_success['GroupKey'] = df_success.apply(
                    lambda row: f"{row.get('ZhangRef', 'N/A')}" if row['Graph'] in ('zhang', 'hybrid', 'multigraph') else 'N/A',
                    axis=1
                )
                group_cols.append('GroupKey')
            if 'MultigraphFusion' in df_success.columns:
                # Fill NaN for non-multigraph experiments; pandas groupby drops NaN rows by default
                df_success['MultigraphFusion'] = df_success['MultigraphFusion'].fillna('N/A')
                group_cols.append('MultigraphFusion')
            if 'Loss' in df_success.columns:
                group_cols.append('Loss')
            
            metric_cols = ['accuracy', 'balanced_accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                          'precision_weighted', 'recall_weighted', 'f1_weighted',
                          'aupr_macro', 'auroc_macro', 'auroc_weighted', 'cohen_kappa', 'mcc']
            for cls in ['CN', 'MCI', 'AD']:
                metric_cols += [f'precision_{cls}', f'recall_{cls}', f'f1_{cls}', 
                               f'aupr_{cls}', f'auroc_{cls}']
            metric_cols = [c for c in metric_cols if c in df_success.columns]
            
            summary = df_success.groupby(group_cols)[metric_cols].agg(['mean', 'std', 'count']).reset_index()
            summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
            summary = summary.sort_values('aupr_macro_mean', ascending=False)
            
            print(f"\n{'Model':<12} | {'Arch':<12} | {'Modal':<10} | {'Graph':<12} | {'L':<3} | {'ZhangRef':<10} | {'Fusion':<12} | {'Loss':<20} | {'Acc':<12} | {'F1':<12} | {'AUPR':<12} | {'AUROC':<12} | {'κ':<7} | {'N':<3}")
            print("-" * 184)
            for _, row in summary.iterrows():
                acc_str = f"{row['accuracy_mean']:.3f}±{row.get('accuracy_std', 0):.3f}"
                f1_str = f"{row['f1_macro_mean']:.3f}±{row.get('f1_macro_std', 0):.3f}"
                aupr_str = f"{row['aupr_macro_mean']:.3f}±{row.get('aupr_macro_std', 0):.3f}"
                auroc_str = f"{row['auroc_macro_mean']:.3f}±{row.get('auroc_macro_std', 0):.3f}"
                kappa_str = f"{row['cohen_kappa_mean']:.3f}" if 'cohen_kappa_mean' in row else "N/A"
                n = int(row.get('accuracy_count', 0))
                zhang_ref = row.get('GroupKey', 'N/A')
                fusion_str = row.get('MultigraphFusion', '-')[:10] if 'MultigraphFusion' in row else '-'
                loss_name = row.get('Loss', 'CE')[:18]
                n_layers = int(row.get('NLayers', 0)) if 'NLayers' in row else '-'
                print(f"{row['Model']:<12} | {row['Arch']:<12} | {row['Modality']:<10} | {row['Graph']:<12} | {str(n_layers):<3} | {zhang_ref:<10} | {fusion_str:<12} | {loss_name:<20} | {acc_str:<12} | {f1_str:<12} | {aupr_str:<12} | {auroc_str:<12} | {kappa_str:<7} | {n:<3}")
            
            print("\n" + "="*100)
            print("📊 MACRO METRICS - ALL MODELS (mean ± std)")
            print("="*100)
            
            for idx, row in summary.iterrows():
                n_layers = int(row.get('NLayers', 0)) if 'NLayers' in row else None
                layers_str = f", L={n_layers}" if n_layers is not None else ""
                loss_str = f" | loss={row.get('Loss', 'CE')}" if 'Loss' in row else ""
                model_label = f"{row['Model']}-{row['Arch']}{layers_str} ({row['Modality']}, {row['Graph']}{loss_str})"
                zhang_ref = row.get('GroupKey', 'N/A')
                if zhang_ref != 'N/A':
                    model_label += f" [Zhang:{zhang_ref}]"
                
                print(f"\n{model_label}")
                print("-" * 80)
                
                print(f"  Accuracy:          {row.get('accuracy_mean', 0):.4f} ± {row.get('accuracy_std', 0):.4f}")
                print(f"  Balanced Accuracy: {row.get('balanced_accuracy_mean', 0):.4f} ± {row.get('balanced_accuracy_std', 0):.4f}")
                print(f"  Precision (macro): {row.get('precision_macro_mean', 0):.4f} ± {row.get('precision_macro_std', 0):.4f}")
                print(f"  Recall (macro):    {row.get('recall_macro_mean', 0):.4f} ± {row.get('recall_macro_std', 0):.4f}")
                print(f"  F1 (macro):        {row.get('f1_macro_mean', 0):.4f} ± {row.get('f1_macro_std', 0):.4f}")
                print(f"  Precision (wgt):   {row.get('precision_weighted_mean', 0):.4f} ± {row.get('precision_weighted_std', 0):.4f}")
                print(f"  Recall (wgt):      {row.get('recall_weighted_mean', 0):.4f} ± {row.get('recall_weighted_std', 0):.4f}")
                print(f"  F1 (weighted):     {row.get('f1_weighted_mean', 0):.4f} ± {row.get('f1_weighted_std', 0):.4f}")
                print(f"  AUPR (macro):      {row.get('aupr_macro_mean', 0):.4f} ± {row.get('aupr_macro_std', 0):.4f}")
                print(f"  AUROC (macro):     {row.get('auroc_macro_mean', 0):.4f} ± {row.get('auroc_macro_std', 0):.4f}")
                print(f"  AUROC (weighted):  {row.get('auroc_weighted_mean', 0):.4f} ± {row.get('auroc_weighted_std', 0):.4f}")
                print(f"  Cohen κ:           {row.get('cohen_kappa_mean', 0):.4f} ± {row.get('cohen_kappa_std', 0):.4f}")
                print(f"  MCC:               {row.get('mcc_mean', 0):.4f} ± {row.get('mcc_std', 0):.4f}")
            
            print("\n" + "="*100)
            print("📊 PER-CLASS METRICS - ALL MODELS (mean ± std)")
            print("="*100)
            
            for idx, row in summary.iterrows():
                n_layers = int(row.get('NLayers', 0)) if 'NLayers' in row else None
                layers_str = f", L={n_layers}" if n_layers is not None else ""
                loss_str = f" | loss={row.get('Loss', 'CE')}" if 'Loss' in row else ""
                model_label = f"{row['Model']}-{row['Arch']}{layers_str} ({row['Modality']}, {row['Graph']}{loss_str})"
                zhang_ref = row.get('GroupKey', 'N/A')
                if zhang_ref != 'N/A':
                    model_label += f" [Zhang:{zhang_ref}]"
                
                print(f"\n{model_label}")
                print("-" * 100)
                print(f"{'Class':<12} | {'Precision':<16} | {'Recall':<16} | {'F1':<16} | {'AUPR':<16} | {'AUROC':<16}")
                print("-" * 100)
                
                for cls in ['CN', 'MCI', 'AD']:
                    prec = f"{row.get(f'precision_{cls}_mean', 0):.4f}±{row.get(f'precision_{cls}_std', 0):.4f}"
                    rec = f"{row.get(f'recall_{cls}_mean', 0):.4f}±{row.get(f'recall_{cls}_std', 0):.4f}"
                    f1 = f"{row.get(f'f1_{cls}_mean', 0):.4f}±{row.get(f'f1_{cls}_std', 0):.4f}"
                    aupr = f"{row.get(f'aupr_{cls}_mean', 0):.4f}±{row.get(f'aupr_{cls}_std', 0):.4f}"
                    auroc = f"{row.get(f'auroc_{cls}_mean', 0):.4f}±{row.get(f'auroc_{cls}_std', 0):.4f}"
                    print(f"{cls:<12} | {prec:<16} | {rec:<16} | {f1:<16} | {aupr:<16} | {auroc:<16}")
        else:
            print("⚠️ No successful experiments to summarize")
    else:
        print("No results collected.")
    print("="*100)

if __name__ == "__main__":
    main()