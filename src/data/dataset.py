import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
from pathlib import Path
from tqdm import tqdm

from .fma_graph import FMAKnowledgeGraph

class AlzheimerGraphDataset(Dataset):
    NODE_TYPE = 'brain_region'
    FMA_NODE_TYPE = 'fma_entity'

    def __init__(self, features_df, coords_path, sigma=15.0, k_neighbors=20, graph_type='static', graph_mix_alpha=0.5,
                 node_semantics_mode='none', precomputed_stats=None, precompute_adjacency=True,
                 codeviation_threshold=0.9, codeviation_kernel_sigma=np.sqrt(2.0),
                 fma_graph_path=None, fma_embedding_backend='pubmedbert',
                 fma_embedding_model='NeuML/pubmedbert-base-embeddings',
                 fma_embedding_dim=128, fma_embedding_cache_dir=None):
        self.df = features_df.reset_index(drop=True)
        self.labels = self.df['label'].values.astype(np.int64)
        self.graph_type = graph_type 
        self.k_neighbors = k_neighbors
        self.sigma = sigma
        self.graph_mix_alpha = float(np.clip(graph_mix_alpha, 0.0, 1.0))
        self.codeviation_threshold = float(codeviation_threshold)
        self.codeviation_kernel_sigma = float(codeviation_kernel_sigma)
        if not 0.0 < self.codeviation_threshold <= 1.0:
            raise ValueError("codeviation_threshold must be in (0, 1]")
        if self.codeviation_kernel_sigma <= 0.0:
            raise ValueError("codeviation_kernel_sigma must be positive")
        self.node_semantics_mode = node_semantics_mode
        valid_semantics = {'none', 'spatial_basic', 'fma_ontology', 'fma_spatial'}
        if node_semantics_mode not in valid_semantics:
            raise ValueError(f"Unsupported node_semantics_mode={node_semantics_mode!r}; use {sorted(valid_semantics)}")
        self.use_fma_graph = node_semantics_mode in {'fma_ontology', 'fma_spatial'}
        self.is_heterogeneous = graph_type == 'multigraph' or self.use_fma_graph
        self.fma_graph = None
        self._fma_config = {
            'graph_path': fma_graph_path,
            'embedding_backend': fma_embedding_backend,
            'embedding_model': fma_embedding_model,
            'embedding_dim': fma_embedding_dim,
            'cache_dir': fma_embedding_cache_dir,
        }
        
        coords_df = pd.read_csv(coords_path)
        feature_cols = set(self.df.columns)
        available_ids = set()
        for col in feature_cols:
            if col.startswith('GM_vol_Region_'):
                try:
                    rid = int(col.split('_')[-1])
                    available_ids.add(rid)
                except ValueError: continue
        
        coords_df = coords_df[coords_df['label_id'].isin(available_ids)].sort_values('label_id').reset_index(drop=True)
        if len(coords_df) == 0: raise ValueError("Errore: Nessuna regione comune!")
            
        self.valid_ids = coords_df['label_id'].values
        self.coords_xyz = coords_df[['x', 'y', 'z']].values 
        self.semantic_feats = self._build_semantic_features(coords_df)
        
        self.gm_cols = [f'GM_vol_Region_{rid}' for rid in self.valid_ids]
        self.wm_cols = [f'WM_vol_Region_{rid}' for rid in self.valid_ids]
        self.csf_cols = [f'CSF_vol_Region_{rid}' for rid in self.valid_ids]
        self.pet_cols = [f'PET_suvr_Region_{rid}' for rid in self.valid_ids]
        
        gm_data = self.df[self.gm_cols].values
        
        wm_data_list = []
        for col in self.wm_cols:
            if col in self.df.columns:
                wm_data_list.append(self.df[col].fillna(0).values)
            else:
                wm_data_list.append(np.zeros(len(self.df)))
        wm_data = np.stack(wm_data_list, axis=1)
        
        csf_data_list = []
        for col in self.csf_cols:
            if col in self.df.columns:
                csf_data_list.append(self.df[col].fillna(0).values)
            else:
                csf_data_list.append(np.zeros(len(self.df)))
        csf_data = np.stack(csf_data_list, axis=1)
        
        pet_data_list = []
        for col in self.pet_cols:
            if col in self.df.columns:
                pet_data_list.append(self.df[col].fillna(0).values)
            else:
                pet_data_list.append(np.zeros(len(self.df)))
        pet_data = np.stack(pet_data_list, axis=1)
        
        self.x_gm = torch.tensor(gm_data, dtype=torch.float32)
        self.x_wm = torch.tensor(wm_data, dtype=torch.float32)
        self.x_csf = torch.tensor(csf_data, dtype=torch.float32)
        self.x_pet = torch.tensor(pet_data, dtype=torch.float32)
        
        sem_dim = self.semantic_feats.shape[1] if self.semantic_feats is not None else 0
        print(f"    Node features: GM={gm_data.shape}, WM={wm_data.shape}, CSF={csf_data.shape}, PET={pet_data.shape}, SEM={sem_dim}")
        
        if self.graph_type in ('zhang', 'hybrid', 'multigraph'):
            if precomputed_stats is not None:
                self.global_mean = precomputed_stats['mean']
                self.global_std = precomputed_stats['std']
                print(f"✅ Usando statistiche precompute dal training set")
            else:
                all_feats = np.concatenate([gm_data, wm_data, csf_data, pet_data], axis=1)
                self.global_mean = np.mean(all_feats, axis=0)
                self.global_std = np.std(all_feats, axis=0) + 1e-6
                print(f"📊 Riferimento Zhang calcolato sul training set ({len(self.df)} pazienti)")

        if not precompute_adjacency:
            self.adj_matrix = None
            self.adj_matrices = None
            return

        if self.graph_type == 'static':
            self.adj_matrix = self._build_static_adjacency(self.coords_xyz, sigma)
            self.adj_matrices = None
            
        elif self.graph_type in ('zhang', 'knn', 'hybrid', 'multigraph'):
            print(f"    ⏳ Pre-computing {self.graph_type.upper()} adjacency for {len(self.df)} samples...")
            if self.graph_type == 'hybrid':
                print(f"    🔀 Mixing static + zhang with alpha={self.graph_mix_alpha:.2f}")
            elif self.graph_type == 'multigraph':
                print("    🔀 Building one heterogeneous graph: relations=proximity, codeviation")
            elif self.graph_type == 'zhang' and self.use_fma_graph:
                print("    ⚡ Building codeviation directly as PyG edges for heterogeneous Zhang+FMA")
            elif self.graph_type == 'knn' and self.use_fma_graph:
                print("    ⚡ Building KNN directly as PyG edges for heterogeneous KNN+FMA")
            self.adj_matrices = []
            static_adj = self._build_static_adjacency(self.coords_xyz, sigma) if self.graph_type in ('hybrid', 'multigraph') else None
            static_edges = self._dense_to_edges(static_adj) if self.graph_type == 'multigraph' else None
            
            for i in tqdm(range(len(self.df)), desc=f"Computing {self.graph_type.upper()} graphs"):
                if self.graph_type == 'zhang':
                    flat_feats = np.concatenate([gm_data[i], wm_data[i], csf_data[i], pet_data[i]])
                    if self.use_fma_graph:
                        adj = self._build_zhang_edges(flat_feats)
                    else:
                        adj = self._build_zhang_adjacency(flat_feats)
                elif self.graph_type == 'hybrid':
                    flat_feats = np.concatenate([gm_data[i], wm_data[i], csf_data[i], pet_data[i]])
                    zhang_adj = self._build_zhang_adjacency(flat_feats)
                    adj = self.graph_mix_alpha * zhang_adj + (1.0 - self.graph_mix_alpha) * static_adj
                    adj = self._normalize_adjacency_torch(adj)
                elif self.graph_type == 'multigraph':
                    flat_feats = np.concatenate([gm_data[i], wm_data[i], csf_data[i], pet_data[i]])
                    adj = {
                        'proximity': static_edges,
                        'codeviation': self._build_zhang_edges(flat_feats),
                    }
                else:  # knn
                    x_patient = np.stack([gm_data[i], wm_data[i], csf_data[i], pet_data[i]], axis=1)
                    if self.use_fma_graph:
                        adj = self._build_knn_edges(x_patient)
                    else:
                        adj = self._build_knn_adjacency(x_patient)
                
                self.adj_matrices.append(adj)
            
            self.adj_matrix = None
            print(f"    ✅ {self.graph_type.upper()} adjacency pre-computed for all samples")
        else:
            self.adj_matrix = None
            self.adj_matrices = None

        if self.use_fma_graph:
            self.fma_graph = FMAKnowledgeGraph(self.valid_ids, **self._fma_config)
            print(
                f"    FMA graph: entities={len(self.fma_graph.entity_ids)}, "
                f"mapped_ROIs={self.fma_graph.mapped_roi_count}/{len(self.valid_ids)}"
            )

    def _build_static_adjacency(self, coords, sigma):
        dist = cdist(coords, coords, metric='euclidean')
        adj = np.exp(- (dist ** 2) / (2 * sigma ** 2))
        adj[adj < 0.1] = 0
        np.fill_diagonal(adj, 1.0)
        row_sum = adj.sum(1)
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return torch.tensor(d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt), dtype=torch.float32)

    def _build_knn_adjacency(self, features):
        adj_sparse = kneighbors_graph(features, self.k_neighbors, mode='connectivity', include_self=True)
        adj = torch.tensor(adj_sparse.toarray(), dtype=torch.float32)
        adj = torch.max(adj, adj.t())
        row_sum = adj.sum(1) + 1e-6
        d_inv = torch.pow(row_sum, -1.0)
        d_mat = torch.diag(d_inv)
        return torch.matmul(d_mat, adj)

    def _build_knn_edges(self, features):
        """Build normalized PyG KNN edges without creating a dense tensor."""
        adjacency = kneighbors_graph(
            features,
            self.k_neighbors,
            mode='connectivity',
            include_self=True,
        )
        adjacency = adjacency.maximum(adjacency.T).tocsr()
        adjacency.sort_indices()
        row_sum = np.asarray(adjacency.sum(axis=1)).ravel() + 1e-6
        source, target = adjacency.nonzero()
        edge_weight = adjacency[source, target].A1 / row_sum[source]
        edge_index = torch.from_numpy(np.stack((source, target))).long()
        edge_weight = torch.from_numpy(
            edge_weight.astype(np.float32, copy=False)
        )
        return edge_index, edge_weight

    def _build_zhang_affinity(self, features_flat):
        """Build the thresholded, unnormalized patient-specific affinity."""
        scale = np.maximum(self.global_std, 1e-6)
        delta = (features_flat - self.global_mean) / scale

        n_nodes = len(self.valid_ids)
        delta_nodes = delta.reshape(4, n_nodes).T
        dist = cdist(delta_nodes, delta_nodes, metric='euclidean')

        adjacency = np.exp(
            - (dist ** 2) / (2.0 * self.codeviation_kernel_sigma ** 2)
        )
        adjacency[adjacency < self.codeviation_threshold] = 0.0
        np.fill_diagonal(adjacency, 1.0)
        return adjacency

    @staticmethod
    def _symmetric_degree_scale(adjacency):
        row_sum = adjacency.sum(1) + 1e-6
        return np.power(row_sum, -0.5)

    def _build_zhang_adjacency(self, features_flat):
        """Build the dense normalized adjacency used by zhang and hybrid."""
        adjacency = self._build_zhang_affinity(features_flat)
        degree_scale = self._symmetric_degree_scale(adjacency)
        normalized = (
            adjacency
            * degree_scale[:, np.newaxis]
            * degree_scale[np.newaxis, :]
        )
        return torch.from_numpy(normalized.astype(np.float32, copy=False))

    def _build_zhang_edges(self, features_flat):
        """Build normalized PyG edges without creating a dense torch tensor."""
        adjacency = self._build_zhang_affinity(features_flat)
        degree_scale = self._symmetric_degree_scale(adjacency)
        source, target = np.nonzero(adjacency)
        edge_weight = (
            adjacency[source, target]
            * degree_scale[source]
            * degree_scale[target]
        )
        edge_index = torch.from_numpy(np.stack((source, target))).long()
        edge_weight = torch.from_numpy(
            edge_weight.astype(np.float32, copy=False)
        )
        return edge_index, edge_weight

    def _normalize_adjacency_torch(self, adj):
        row_sum = torch.sum(adj, dim=1) + 1e-6
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    @staticmethod
    def _dense_to_edges(adj):
        """Convert a weighted dense adjacency to PyG edge_index/edge_weight."""
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        edge_weight = adj[edge_index[0], edge_index[1]].contiguous()
        return edge_index, edge_weight

    def _topology_relations(self, idx):
        if self.graph_type == 'multigraph':
            return self.adj_matrices[idx]
        if self.adj_matrices is not None:
            adjacency = self.adj_matrices[idx]
        elif self.adj_matrix is not None:
            adjacency = self.adj_matrix
        else:
            raise RuntimeError(f"Adjacency not computed for graph_type={self.graph_type}")
        relation_names = {
            'static': 'proximity',
            'zhang': 'codeviation',
            'hybrid': 'hybrid_connectivity',
            'knn': 'feature_similarity',
        }
        relation = relation_names[self.graph_type]
        if self.graph_type in ('zhang', 'knn') and self.use_fma_graph:
            return {relation: adjacency}
        return {relation: self._dense_to_edges(adjacency)}

    def metadata(self):
        """Metadata passed explicitly to torch_geometric.nn.to_hetero."""
        if not self.is_heterogeneous:
            raise ValueError("Metadata is only available for heterogeneous datasets")
        node_types = [self.NODE_TYPE]
        edge_types = [
            (self.NODE_TYPE, relation, self.NODE_TYPE)
            for relation in self._topology_relation_names()
        ]
        if self.use_fma_graph:
            node_types.append(self.FMA_NODE_TYPE)
            edge_types.extend([
                (self.NODE_TYPE, 'maps_to', self.FMA_NODE_TYPE),
                (self.FMA_NODE_TYPE, 'describes', self.NODE_TYPE),
                (self.FMA_NODE_TYPE, 'is_a', self.FMA_NODE_TYPE),
                (self.FMA_NODE_TYPE, 'has_subclass', self.FMA_NODE_TYPE),
                (self.FMA_NODE_TYPE, 'part_of', self.FMA_NODE_TYPE),
                (self.FMA_NODE_TYPE, 'has_part', self.FMA_NODE_TYPE),
            ])
        return node_types, edge_types

    def _topology_relation_names(self):
        if self.graph_type == 'multigraph':
            return ('proximity', 'codeviation')
        return ({
            'static': 'proximity',
            'zhang': 'codeviation',
            'hybrid': 'hybrid_connectivity',
            'knn': 'feature_similarity',
        }[self.graph_type],)

    @staticmethod
    def _spatial_features(coords_df):
        xyz = coords_df[['x', 'y', 'z']].values.astype(np.float32)
        mean = np.mean(xyz, axis=0, keepdims=True)
        std = np.std(xyz, axis=0, keepdims=True) + 1e-6
        xyz_norm = (xyz - mean) / std
        radial = np.linalg.norm(xyz_norm, axis=1, keepdims=True)
        hemisphere = (xyz[:, 0:1] > 0).astype(np.float32)
        return torch.tensor(
            np.concatenate([xyz_norm, radial, hemisphere], axis=1),
            dtype=torch.float32,
        )

    def _build_semantic_features(self, coords_df):
        # FMA knowledge is represented by entity nodes and edges, never by ROI features.
        if self.node_semantics_mode in {'none', 'fma_ontology'}:
            return None
        if self.node_semantics_mode in {'spatial_basic', 'fma_spatial'}:
            return self._spatial_features(coords_df)
        raise AssertionError("node_semantics_mode was validated in __init__")

    def _heterogeneous_sample(self, idx, x):
        data = HeteroData()
        data[self.NODE_TYPE].x = x
        for relation, (edge_index, edge_weight) in self._topology_relations(idx).items():
            edge_type = (self.NODE_TYPE, relation, self.NODE_TYPE)
            data[edge_type].edge_index = edge_index
            data[edge_type].edge_weight = edge_weight

        if self.use_fma_graph:
            fma = self.fma_graph
            data[self.FMA_NODE_TYPE].x = fma.x
            relation_edges = {
                (self.NODE_TYPE, 'maps_to', self.FMA_NODE_TYPE): fma.roi_to_entity,
                (self.FMA_NODE_TYPE, 'describes', self.NODE_TYPE): fma.entity_to_roi,
                (self.FMA_NODE_TYPE, 'is_a', self.FMA_NODE_TYPE): fma.edges['is_a'],
                (self.FMA_NODE_TYPE, 'has_subclass', self.FMA_NODE_TYPE): fma.edges['has_subclass'],
                (self.FMA_NODE_TYPE, 'part_of', self.FMA_NODE_TYPE): fma.edges['part_of'],
                (self.FMA_NODE_TYPE, 'has_part', self.FMA_NODE_TYPE): fma.edges['has_part'],
            }
            for edge_type, edge_index in relation_edges.items():
                data[edge_type].edge_index = edge_index
                data[edge_type].edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

        data.y = torch.tensor([self.labels[idx]], dtype=torch.long)
        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.cat([
            self.x_gm[idx].unsqueeze(1),
            self.x_wm[idx].unsqueeze(1),
            self.x_csf[idx].unsqueeze(1),
            self.x_pet[idx].unsqueeze(1)
        ], dim=1)

        if self.semantic_feats is not None:
            x = torch.cat([x, self.semantic_feats], dim=1)
        
        if self.is_heterogeneous:
            return self._heterogeneous_sample(idx, x)

        if self.adj_matrices is not None:
            adj = self.adj_matrices[idx]
        elif self.adj_matrix is not None:
            adj = self.adj_matrix
        else:
            raise RuntimeError(f"Adjacency not computed for graph_type={self.graph_type}")
            
        return x, adj, self.labels[idx]

class HybridAlzheimerDataset(AlzheimerGraphDataset):
    def __init__(self, features_df, coords_path, clinical_cols, sigma=15.0, k_neighbors=20, graph_type='static', graph_mix_alpha=0.5,
                 node_semantics_mode='none', precomputed_stats=None, precompute_adjacency=True,
                 codeviation_threshold=0.9, codeviation_kernel_sigma=np.sqrt(2.0),
                 fma_graph_path=None, fma_embedding_backend='pubmedbert',
                 fma_embedding_model='NeuML/pubmedbert-base-embeddings',
                 fma_embedding_dim=128, fma_embedding_cache_dir=None):
        super().__init__(
            features_df=features_df,
            coords_path=coords_path,
            sigma=sigma,
            k_neighbors=k_neighbors,
            graph_type=graph_type,
            graph_mix_alpha=graph_mix_alpha,
            node_semantics_mode=node_semantics_mode,
            precomputed_stats=precomputed_stats,
            precompute_adjacency=precompute_adjacency,
            codeviation_threshold=codeviation_threshold,
            codeviation_kernel_sigma=codeviation_kernel_sigma,
            fma_graph_path=fma_graph_path,
            fma_embedding_backend=fma_embedding_backend,
            fma_embedding_model=fma_embedding_model,
            fma_embedding_dim=fma_embedding_dim,
            fma_embedding_cache_dir=fma_embedding_cache_dir,
        )
        self.clinical_cols = clinical_cols
        self.x_clin = torch.tensor(features_df[clinical_cols].values, dtype=torch.float32)

    def __getitem__(self, idx):
        sample = super(HybridAlzheimerDataset, self).__getitem__(idx)
        x_clin = self.x_clin[idx]
        if self.is_heterogeneous:
            sample.x_clin = x_clin.unsqueeze(0)
            return sample
        x_graph, adj, y = sample
        return x_graph, adj, x_clin, y