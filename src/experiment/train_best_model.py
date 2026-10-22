"""
Script for training the GAT PartialKAN Multimodal Zhang model
for final report generation.
"""
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import datetime
import pickle
import json
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            confusion_matrix, roc_curve, auc, 
                            precision_recall_curve, average_precision_score,
                            precision_score, recall_score, f1_score,
                            roc_auc_score, balanced_accuracy_score,
                            cohen_kappa_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.data.dataset import HybridAlzheimerDataset
from src.data.batch_utils import (build_graph_dataloader, fma_dataset_kwargs,
                                  get_node_feature_dim, get_node_feature_dims, move_graph_batch_to_device)
from src.models.model import GAT_KAN_Classifier

# configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment" / "mci" / "gat_multigraph.yaml"
CLINICAL_COLS = ['AGE', 'Gender_Male', 'PTEDUCAT', 'APOE4', 'MMSE', 
                 'ABETA_final', 'TAU_final', 'PTAU_final', 'CSF_Available']

# model config
MODEL_NAME = "GAT_PartialKAN"
GRAPH_TYPE = "zhang"
MODALITY = "Multimodal"
USE_BOTTLENECK = False  # partialkan
USE_HEAD = True         # partialkan

HIDDEN_CHANNELS = 32
MAX_EPOCHS = 100
PATIENCE = 20
LEARNING_RATE = 0.0005
DROPOUT = 0.65
WEIGHT_DECAY = 5e-3

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def prepare_data(cfg):
    print("📊 Loading data...")
    df_img = pd.read_csv(PROJECT_ROOT / cfg['data']['features_path'])
    df_img['PTID'] = df_img['subject_id'].apply(lambda x: "_".join(x.split('_')[:3]))
    
    clin_path = PROJECT_ROOT / "data/processed/clinical_features.csv"
    if not clin_path.exists():
        clin_path = PROJECT_ROOT / "data/02_processed/clinical_features.csv"
    df_clin = pd.read_csv(clin_path, on_bad_lines='skip')
    
    df_merged = pd.merge(df_img, df_clin, on='PTID', how='inner', suffixes=('', '_clin'))
    if 'label_clin' in df_merged.columns:
        df_merged['label'] = df_merged['label_clin']
    
    df_merged = df_merged.dropna(subset=['label']).reset_index(drop=True)
    
    img_cols = [c for c in df_merged.columns if 'GM_' in c or 'PET_' in c]
    scaler_img = StandardScaler()
    df_merged[img_cols] = scaler_img.fit_transform(df_merged[img_cols].fillna(0))
    
    scaler_clin = StandardScaler()
    df_merged[CLINICAL_COLS] = scaler_clin.fit_transform(df_merged[CLINICAL_COLS].fillna(0))
    
    print(f"  ✅ Samples: {len(df_merged)}")
    print(f"  - Class 0: {(df_merged['label']==0).sum()}")
    print(f"  - Class 1: {(df_merged['label']==1).sum()}")
    
    return df_merged, scaler_img, scaler_clin

def create_dataloaders(df, coords_path, cfg, batch_size=32):
    print("\n🔄 Creating dataloaders...")
    
    unique_ptids = df['PTID'].unique()
    labels_per_ptid = df.groupby('PTID')['label'].first()
    
    train_ptids, test_ptids = train_test_split(
        unique_ptids, test_size=0.2, stratify=labels_per_ptid, random_state=42
    )
    
    train_df = df[df['PTID'].isin(train_ptids)].reset_index(drop=True)
    test_df = df[df['PTID'].isin(test_ptids)].reset_index(drop=True)
    
    train_labels = train_df.groupby('PTID')['label'].first()
    tr_ptids, val_ptids = train_test_split(
        train_df['PTID'].unique(), test_size=0.15, 
        stratify=train_labels, random_state=42
    )
    
    tr_df = train_df[train_df['PTID'].isin(tr_ptids)].reset_index(drop=True)
    val_df = train_df[train_df['PTID'].isin(val_ptids)].reset_index(drop=True)
    
    print(f"  - Train: {len(tr_df)} samples ({len(tr_ptids)} subjects)")
    print(f"  - Val: {len(val_df)} samples ({len(val_ptids)} subjects)")
    print(f"  - Test: {len(test_df)} samples ({len(test_ptids)} subjects)")
    
    ds_kwargs = {
        'sigma': cfg['model']['sigma_graph'],
        'k_neighbors': 20,
        'graph_type': GRAPH_TYPE,
        'graph_mix_alpha': cfg['model'].get('hybrid_alpha', 0.5),
        'node_semantics_mode': cfg['model'].get('node_semantics_mode', 'none'),
        'clinical_cols': CLINICAL_COLS
    }
    
    ds_kwargs.update(fma_dataset_kwargs(cfg))

    train_ds = HybridAlzheimerDataset(tr_df, coords_path, **ds_kwargs)
    val_ds = HybridAlzheimerDataset(val_df, coords_path, **ds_kwargs)
    test_ds = HybridAlzheimerDataset(test_df, coords_path, **ds_kwargs)
    
    y_tr = tr_df['label'].values.astype(int)
    class_weights = np.sqrt(len(y_tr) / (np.bincount(y_tr) + 1e-6))
    sample_weights = torch.from_numpy(class_weights[y_tr]).float()
    sampler = WeightedRandomSampler(sample_weights, len(y_tr), replacement=True)
    
    train_loader = build_graph_dataloader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = build_graph_dataloader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = build_graph_dataloader(test_ds, batch_size=batch_size, shuffle=False)
    
    n_node_features = get_node_feature_dim(train_ds)
    hetero_metadata = train_ds.metadata() if train_ds.is_heterogeneous else None
    hetero_input_dims = get_node_feature_dims(train_ds)
    return train_loader, val_loader, test_loader, test_df, n_node_features, hetero_metadata, hetero_input_dims

def train_model(model, train_loader, val_loader, device, output_dir, epochs=100, patience=20):
    print(f"\n🚀 Training {MODEL_NAME} con {GRAPH_TYPE} graph...")
    print(f"  📋 Config: hidden={HIDDEN_CHANNELS}, dropout={DROPOUT}, weight_decay={WEIGHT_DECAY}")
    print(f"  📋 Training: epochs={epochs}, patience={patience}, lr={LEARNING_RATE}")
    print(f"  ✅ Early Stopping: ATTIVO")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    counter = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_truths = [], []
        
        for batch in train_loader:
            x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=GRAPH_TYPE, has_clinical=True)
            
            optimizer.zero_grad()
            outputs = model(x_g, adj, x_c)
            loss = criterion(outputs, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            train_truths.extend(y.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_truths, train_preds)
        
        model.eval()
        val_loss = 0.0
        val_preds, val_truths = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=GRAPH_TYPE, has_clinical=True)
                outputs = model(x_g, adj, x_c)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
                val_truths.extend(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_truths, val_preds)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = model.state_dict().copy()
            counter = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"\n  ⏹️  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    print(f"\n  🏆 Best model: Epoch {best_epoch}, Val Acc: {best_val_acc:.2%}")
    
    return model, history, best_val_acc

def evaluate_model(model, test_loader, device):
    print("\n📈 Evaluating on test set...")
    
    model.eval()
    all_preds = []
    all_probs = []
    all_truths = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_g, adj, x_c, y = move_graph_batch_to_device(batch, device, graph_type=GRAPH_TYPE, has_clinical=True)
            outputs = model(x_g, adj, x_c)
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_truths.extend(y.cpu().numpy())
    
    acc = accuracy_score(all_truths, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_truths, all_preds, average='binary'
    )
    aupr = average_precision_score(all_truths, all_probs)
    
    fpr, tpr, _ = roc_curve(all_truths, all_probs)
    roc_auc = auc(fpr, tpr)
    
    cm = confusion_matrix(all_truths, all_preds)
    
    balanced_acc = balanced_accuracy_score(all_truths, all_preds)
    cohen_k = cohen_kappa_score(all_truths, all_preds)
    mcc = matthews_corrcoef(all_truths, all_preds)
    
    prec_per = precision_score(all_truths, all_preds, average=None, zero_division=0)
    rec_per = recall_score(all_truths, all_preds, average=None, zero_division=0)
    f1_per = f1_score(all_truths, all_preds, average=None, zero_division=0)
    
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'aupr': aupr,
        'roc_auc': roc_auc,
        'cohen_kappa': cohen_k,
        'mcc': mcc,
        'confusion_matrix': cm.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'precision_class_0': float(prec_per[0]),
        'precision_class_1': float(prec_per[1]),
        'recall_class_0': float(rec_per[0]),
        'recall_class_1': float(rec_per[1]),
        'f1_class_0': float(f1_per[0]),
        'f1_class_1': float(f1_per[1]),
    }
    
    print(f"  ✅ Accuracy:     {acc:.2%}")
    print(f"  ✅ Balanced Acc: {balanced_acc:.2%}")
    print(f"  ✅ Precision:    {prec:.2%}")
    print(f"  ✅ Recall:       {rec:.2%}")
    print(f"  ✅ F1-Score:     {f1:.2%}")
    print(f"  ✅ AUPR:         {aupr:.2%}")
    print(f"  ✅ ROC-AUC:      {roc_auc:.2%}")
    print(f"  ✅ Cohen κ:      {cohen_k:.3f}")
    print(f"  ✅ MCC:          {mcc:.3f}")
    print(f"\n  Per-class metrics:")
    print(f"    Class 0: Prec={prec_per[0]:.3f}, Rec={rec_per[0]:.3f}, F1={f1_per[0]:.3f}")
    print(f"    Class 1: Prec={prec_per[1]:.3f}, Rec={rec_per[1]:.3f}, F1={f1_per[1]:.3f}")
    
    return all_preds, all_probs, all_truths, metrics

def plot_training_curves(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: training_curves.png")

def plot_confusion_matrix(cm, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'],
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: confusion_matrix.png")

def plot_roc_curve(fpr, tpr, roc_auc, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: roc_curve.png")

def plot_precision_recall_curve(truths, probs, aupr, output_dir):
    precision, recall, _ = precision_recall_curve(truths, probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR (AUPR = {aupr:.3f})')
    ax.axhline(y=sum(truths)/len(truths), color='r', linestyle='--', 
               linewidth=2, label=f'Baseline = {sum(truths)/len(truths):.3f}')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: precision_recall_curve.png")

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "models" / f"best_model_v3_antioverfit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {output_dir}")
    
    df, scaler_img, scaler_clin = prepare_data(cfg)
    coords_path = PROJECT_ROOT / cfg['data']['coords_path']
    
    train_loader, val_loader, test_loader, test_df, n_node_features, hetero_metadata, hetero_input_dims = create_dataloaders(
        df, coords_path, cfg, batch_size=32
    )
    
    model = GAT_KAN_Classifier(
        n_node_features,
        len(CLINICAL_COLS),
        HIDDEN_CHANNELS,
        2,
        DROPOUT,
        USE_BOTTLENECK,
        USE_HEAD,
        graph_type=GRAPH_TYPE,
        hetero_metadata=hetero_metadata,
        hetero_input_dims=hetero_input_dims
    ).to(device)
    
    print(f"\n🏗️  Model: {MODEL_NAME}")
    print(f"  - Graph: {GRAPH_TYPE}")
    print(f"  - Modality: {MODALITY}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model, history, best_val_acc = train_model(
        model, train_loader, val_loader, device, output_dir,
        epochs=MAX_EPOCHS, patience=PATIENCE
    )
    
    preds, probs, truths, metrics = evaluate_model(model, test_loader, device)
    
    print("\n💾 Saving results...")
    
    torch.save(model.state_dict(), output_dir / 'model_final.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'name': MODEL_NAME,
            'graph_type': GRAPH_TYPE,
            'modality': MODALITY,
            'use_bottleneck_kan': USE_BOTTLENECK,
            'use_head_kan': USE_HEAD,
            'in_channels': 2,
            'n_clinical': len(CLINICAL_COLS),
            'hidden_channels': HIDDEN_CHANNELS,
            'out_channels': 2,
            'dropout': DROPOUT,
            'version': 'v3_antioverfit',
            'max_epochs': MAX_EPOCHS,
            'patience': PATIENCE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY
        },
        'clinical_cols': CLINICAL_COLS,
        'metrics': metrics
    }, output_dir / 'model_complete.pth')
    
    with open(output_dir / 'scalers.pkl', 'wb') as f:
        pickle.dump({
            'scaler_img': scaler_img,
            'scaler_clin': scaler_clin
        }, f)
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    pred_df = pd.DataFrame({
        'PTID': test_df['PTID'].values,
        'True_Label': truths,
        'Predicted_Label': preds,
        'Predicted_Prob': probs
    })
    pred_df.to_csv(output_dir / 'test_predictions.csv', index=False)
    
    print(f"  ✅ Model saved: model_final.pth")
    print(f"  ✅ Checkpoint saved: model_complete.pth")
    print(f"  ✅ Scalers saved: scalers.pkl")
    print(f"  ✅ Metrics saved: metrics.json")
    
    # plots
    print("\n🎨 Generating visualizations...")
    plot_training_curves(history, output_dir)
    plot_confusion_matrix(metrics['confusion_matrix'], output_dir)
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], output_dir)
    plot_precision_recall_curve(truths, probs, metrics['aupr'], output_dir)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\n📊 Final Results:")
    print(f"  --- Global Metrics ---")
    print(f"  Accuracy:       {metrics['accuracy']:.2%}")
    print(f"  Balanced Acc:   {metrics['balanced_accuracy']:.2%}")
    print(f"  Precision:      {metrics['precision']:.2%}")
    print(f"  Recall:         {metrics['recall']:.2%}")
    print(f"  F1-Score:       {metrics['f1_score']:.2%}")
    print(f"  AUPR:           {metrics['aupr']:.2%}")
    print(f"  ROC-AUC:        {metrics['roc_auc']:.2%}")
    print(f"  Cohen κ:        {metrics['cohen_kappa']:.3f}")
    print(f"  MCC:            {metrics['mcc']:.3f}")
    print(f"\n  --- Per-Class Metrics ---")
    print(f"  Class 0 (CN):   Prec={metrics['precision_class_0']:.3f}, Rec={metrics['recall_class_0']:.3f}, F1={metrics['f1_class_0']:.3f}")
    print(f"  Class 1 (AD):   Prec={metrics['precision_class_1']:.3f}, Rec={metrics['recall_class_1']:.3f}, F1={metrics['f1_class_1']:.3f}")
    print(f"\n📁 All files saved in: {output_dir}")

if __name__ == "__main__":
    main()
