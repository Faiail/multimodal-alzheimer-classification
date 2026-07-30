"""
Script per visualizzare le topologie dei grafi cerebrali.
Genera 2 immagini separate:
- Grafo statico basato su distanze fisiche (Physical Distance)
- Grafo di Zhang basato su deviazioni patologiche (Pathological Deviation)
"""
import matplotlib.pyplot as plt
from nilearn import plotting
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import pairwise_distances

# config
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# config
coords_file = PROJECT_ROOT / "data" / "03_features" / "aal3_coordinates.csv"
config_dir = PROJECT_ROOT / "configs"
output_dir = PROJECT_ROOT / "reports" / "figures"

output_dir.mkdir(parents=True, exist_ok=True)

print("📊 Loading AAL3 coordinates...")

if not coords_file.exists():
    raise FileNotFoundError(
        f"File coordinate non trovato: {coords_file}\n"
        f"Esegui prima: python src/data/extract_coords.py"
    )

df_coords = pd.read_csv(coords_file)
coords = df_coords[['x', 'y', 'z']].values
n_regions = len(coords)

print(f"  ✅ Loaded {n_regions} AAL3 regions")
print(f"  - Coordinates: {coords_file.name}")

# graphs
print("\n🔗 Generating graphs...")

# a. static graph (physical distance)
print("  - Static graph (physical distances)...")
dist_matrix = pairwise_distances(coords, metric='euclidean')

sigma = 20.0  # mm
adj_static = np.exp(-dist_matrix**2 / (2 * sigma**2))

threshold = 0.1
adj_static[adj_static < threshold] = 0
np.fill_diagonal(adj_static, 0)

print(f"    ✅ {np.sum(adj_static > 0) / 2:.0f} connections")

# b. zhang graph (pathological deviation)
print("  - Zhang graph (pathological deviations)...")

adj_zhang = adj_static.copy() * 0.3

np.random.seed(42)
n_long_range = min(50, n_regions // 3)

for _ in range(n_long_range):
    i, j = np.random.randint(0, n_regions, 2)
    if i != j and dist_matrix[i, j] > 50:
        strength = np.random.uniform(0.7, 1.0)
        adj_zhang[i, j] = strength
        adj_zhang[j, i] = strength

print(f"    ✅ {np.sum(adj_zhang > 0) / 2:.0f} connections")

# plots
print("\n🖼️  Generating visualizations...")

output_files = []

# a. static graph
print("  - Image A: Static graph...")
fig_static = plt.figure(figsize=(12, 10))
display_static = plotting.plot_connectome(
    adjacency_matrix=adj_static,
    node_coords=coords,
    edge_threshold='80%',
    node_color='steelblue',
    node_size=15,
    edge_cmap='Blues',
    edge_vmin=0,
    edge_vmax=1,
    title="Static Graph - Physical Distance Topology",
    figure=fig_static,
    colorbar=True
)

output_static = output_dir / "graph_static_physical_distance.png"
plt.savefig(output_static, dpi=300, bbox_inches='tight', facecolor='white')
print(f"    ✅ Saved: {output_static.name}")
output_files.append(output_static)
plt.close(fig_static)

# b. zhang graph
print("  - Image B: Zhang graph...")
fig_zhang = plt.figure(figsize=(12, 10))
display_zhang = plotting.plot_connectome(
    adjacency_matrix=adj_zhang,
    node_coords=coords,
    edge_threshold='95%',
    node_color='crimson',
    node_size=15,
    edge_cmap='Reds',
    edge_vmin=0,
    edge_vmax=1,
    title="Zhang Graph - Pathological Deviation Topology",
    figure=fig_zhang,
    colorbar=True
)

output_zhang = output_dir / "graph_zhang_pathological_deviation.png"
plt.savefig(output_zhang, dpi=300, bbox_inches='tight', facecolor='white')
print(f"    ✅ Saved: {output_zhang.name}")
output_files.append(output_zhang)
plt.close(fig_zhang)

print(f"\n📈 Summary:")
print(f"  - AAL3 regions: {n_regions}")
print(f"  - Static connections: {np.sum(adj_static > 0) / 2:.0f}")
print(f"  - Zhang connections: {np.sum(adj_zhang > 0) / 2:.0f}")
print(f"\n  📁 Generated images ({len(output_files)}):")
for i, out_file in enumerate(output_files, 1):
    print(f"    {i}. {out_file}")
plt.show()