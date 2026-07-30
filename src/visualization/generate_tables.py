"""
Script to generate comparison tables from existing results
Usage: python generate_tables.py <path_to_run_folder>
"""
import pandas as pd
import sys
from pathlib import Path

def generate_comparison_tables(run_dir):
    run_dir = Path(run_dir)
    
    df_res = pd.read_csv(run_dir / "detailed_results.csv")
    
    print(f"📊 Generating tables from {len(df_res)} results...")
    
    pivot_acc = df_res.pivot_table(
        index=['Model', 'Arch'], 
        columns=['Modality', 'Graph'], 
        values='Acc', 
        aggfunc=['mean', 'std']
    )
    pivot_acc.to_csv(run_dir / "table_accuracy_pivot.csv")
    print(f"✅ Saved: table_accuracy_pivot.csv")
    
    pivot_aupr = df_res.pivot_table(
        index=['Model', 'Arch'], 
        columns=['Modality', 'Graph'], 
        values='AUPR', 
        aggfunc=['mean', 'std']
    )
    pivot_aupr.to_csv(run_dir / "table_aupr_pivot.csv")
    print(f"✅ Saved: table_aupr_pivot.csv")
    
    summary = df_res.groupby(['Model', 'Arch', 'Modality', 'Graph', 'Data']).agg({
        'Acc': ['mean', 'std', 'count'],
        'AUPR': ['mean', 'std']
    }).reset_index()
    
    summary['Acc_formatted'] = summary.apply(
        lambda x: f"{x[('Acc', 'mean')]:.2%} ± {x[('Acc', 'std')]:.2%}", axis=1
    )
    summary['AUPR_formatted'] = summary.apply(
        lambda x: f"{x[('AUPR', 'mean')]:.2%} ± {x[('AUPR', 'std')]:.2%}", axis=1
    )
    
    summary_clean = summary[['Model', 'Arch', 'Modality', 'Graph', 'Data', 
                             'Acc_formatted', 'AUPR_formatted', ('Acc', 'count')]]
    summary_clean.columns = ['Model', 'Arch', 'Modality', 'Graph', 'Data', 
                             'Accuracy', 'AUPR', 'N_Folds']
    summary_clean.to_csv(run_dir / "table_summary_formatted.csv", index=False)
    print(f"✅ Saved: table_summary_formatted.csv")
    
    top10 = summary.nlargest(10, ('Acc', 'mean'))
    top10_clean = top10[['Model', 'Arch', 'Modality', 'Graph']].copy()
    top10_clean['Accuracy'] = top10.apply(
        lambda x: f"${x[('Acc', 'mean')]*100:.2f} \\pm {x[('Acc', 'std')]*100:.2f}$", axis=1
    )
    top10_clean['AUPR'] = top10.apply(
        lambda x: f"${x[('AUPR', 'mean')]*100:.2f} \\pm {x[('AUPR', 'std')]*100:.2f}$", axis=1
    )
    
    latex = top10_clean.to_latex(
        index=False,
        escape=False,
        caption="Top 10 Model Configurations",
        label="tab:top10",
        column_format='llllrr'
    )
    with open(run_dir / "table_top10.tex", 'w') as f:
        f.write(latex)
    print(f"✅ Saved: table_top10.tex")
    
    categories = [
        ('XGB', 'Std', 'Imaging'),
        ('XGB', 'Zhang', 'Multimodal'),
        ('GAT', 'KAN-Full', 'Multimodal'),
        ('GCN', 'Standard', 'Imaging')
    ]
    
    best_per_cat = []
    for model, arch, modality in categories:
        subset = df_res[
            (df_res['Model'] == model) & 
            (df_res['Arch'] == arch) & 
            (df_res['Modality'] == modality)
        ]
        if len(subset) > 0:
            best_row = subset.groupby(['Model', 'Arch', 'Modality', 'Graph']).agg({
                'Acc': 'mean', 'AUPR': 'mean'
            }).reset_index().nlargest(1, 'Acc')
            best_per_cat.append(best_row)
    
    if best_per_cat:
        best_df = pd.concat(best_per_cat)
        best_df.to_csv(run_dir / "table_best_per_category.csv", index=False)
        print(f"✅ Saved: table_best_per_category.csv")
    
    if df_res['Fold'].nunique() >= 5:
        from scipy import stats
        
        xgb_scores = df_res[df_res['Model'] == 'XGB']['Acc'].values
        best_gnn_model = summary.nlargest(1, ('Acc', 'mean')).iloc[0]['Model']
        best_gnn_arch = summary.nlargest(1, ('Acc', 'mean')).iloc[0]['Arch']
        
        gnn_scores = df_res[
            (df_res['Model'] == best_gnn_model) & 
            (df_res['Arch'] == best_gnn_arch)
        ]['Acc'].values
        
        if len(xgb_scores) > 0 and len(gnn_scores) > 0:
            t_stat, p_val = stats.ttest_ind(gnn_scores, xgb_scores)
            
            with open(run_dir / "statistical_test.txt", 'w') as f:
                f.write(f"Paired t-test: Best GNN vs XGB Baseline\n")
                f.write(f"Best GNN: {best_gnn_model}-{best_gnn_arch}\n")
                f.write(f"XGB Mean: {xgb_scores.mean():.4f}\n")
                f.write(f"GNN Mean: {gnn_scores.mean():.4f}\n")
                f.write(f"t-statistic: {t_stat:.4f}\n")
                f.write(f"p-value: {p_val:.4f}\n")
                f.write(f"Significant (p<0.05): {'Yes' if p_val < 0.05 else 'No'}\n")
            
            print(f"✅ Saved: statistical_test.txt (p={p_val:.4f})")
    
    print(f"\n🎉 All tables generated in: {run_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        reports_dir = Path(__file__).parents[1] / "reports"
        runs = sorted([d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
        if not runs:
            print("❌ No run folders found in reports/")
            sys.exit(1)
        run_dir = runs[-1]
        print(f"📂 Using latest run: {run_dir.name}")
    else:
        run_dir = Path(sys.argv[1])
    
    generate_comparison_tables(run_dir)
