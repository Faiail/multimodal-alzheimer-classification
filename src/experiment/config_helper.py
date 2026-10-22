from typing import Dict, Tuple

from src.utils.general import resolve_data_path

VALID_GRAPH_MODES = ("static", "zhang", "hybrid", "knn", "multigraph")


def build_model_and_data_configs(cfg: Dict) -> Tuple[Dict, Dict, str]:
    data_cfg = dict(cfg.get("data", {}))
    model_cfg = dict(cfg.get("model", {}))

    graph_mode = str(data_cfg.get("graph_mode", "static")).lower()
    if graph_mode not in VALID_GRAPH_MODES:
        raise ValueError(
            f"Unsupported data.graph_mode='{graph_mode}'. "
            f"Supported modes: {VALID_GRAPH_MODES}"
        )

    use_semantic_features = bool(data_cfg.get("use_semantic_features", True))
    use_fma_graph = bool(data_cfg.get("use_fma_graph", True))

    dataset_kwargs = {
        "csv_path": resolve_data_path(data_cfg.get("features_file") or data_cfg.get("csv_path")),
        "coords_path": resolve_data_path(data_cfg.get("coords_file", "configs/aal3_coordinates.csv")),
        "target_col": data_cfg.get("target_col", "target"),
        "sigma": float(data_cfg.get("sigma", 25.0)),
        "graph_type": graph_mode,
        "knn_k": int(data_cfg.get("knn_k", 10)),
        "zhang_threshold": float(data_cfg.get("zhang_threshold", 0.5)),
        "graph_mix_alpha": float(data_cfg.get("graph_mix_alpha", 0.5)),
        "use_semantic_features": use_semantic_features,
        "ontology_lookup_path": resolve_data_path(
            data_cfg.get("ontology_lookup_file", "configs/aal3_ontology_lookup.csv")
        ),
        "use_fma_graph": use_fma_graph,
        "ontology_kg_path": resolve_data_path(
            data_cfg.get("ontology_kg_file", "configs/fma_knowledge_graph.json")
        ),
        "fma_Reverse_edges": bool(data_cfg.get("fma_reverse_edges", True)),
        "fma_include_part_of": bool(data_cfg.get("fma_include_part_of", True)),
        "fma_include_is_a": bool(data_cfg.get("fma_include_is_a", True)),
        "fma_include_continuous_with": bool(data_cfg.get("fma_include_continuous_with", True)),
        "fma_include_Tributary_of": bool(data_cfg.get("fma_include_tributary_of", True)),
        "fma_include_homologous_to": bool(data_cfg.get("fma_include_homologous_to", True)),
        "fma_include_adjacent_to": bool(data_cfg.get("fma_include_adjacent_to", False)),
        "fma_include_attaches_to": bool(data_cfg.get("fma_include_attaches_to", True)),
        "fma_include_receives_input_from": bool(data_cfg.get("fma_include_receives_input_from", True)),
        "fma_include_sends_output_to": bool(data_cfg.get("fma_include_sends_output_to", True)),
        "fma_include_nerve_supply": bool(data_cfg.get("fma_include_nerve_supply", False)),
        "fma_include_arterial_supply": bool(data_cfg.get("fma_include_arterial_supply", False)),
        "fma_include_venous_drainage": bool(data_cfg.get("fma_include_venous_drainage", False)),
        "fma_include_regional_part_of": bool(data_cfg.get("fma_include_regional_part_of", True)),
        "fma_include_constitutional_part_of": bool(data_cfg.get("fma_include_constitutional_part_of", True)),
        "fma_include_anatomical_axes": bool(data_cfg.get("fma_include_anatomical_axes", False)),
        "fma_include_developmental": bool(data_cfg.get("fma_include_developmental", False)),
        "fma_include_projected_relations": bool(data_cfg.get("fma_include_projected_relations", True)),
        "fma_embedding_model": str(
            data_cfg.get("fma_embedding_model", "pritamdeka/S-PubMedBert-MS-MARCO")
        ),
        "fma_embedding_dim": int(data_cfg.get("fma_embedding_dim", 768)),
        "fma_embedding_cache_dir": str(
            data_cfg.get("fma_embedding_cache_dir", "models/embeddings")
        ),
    }

    semantic_dim = 18 if use_semantic_features else 0
    base_in_features = 4 + semantic_dim
    fma_dim = int(data_cfg.get("fma_embedding_dim", 768)) if use_fma_graph else 0
    in_features = int(model_cfg.get("in_features", base_in_features + fma_dim))

    model_kwargs = {
        "num_nodes": int(model_cfg.get("num_nodes", 166)),
        "in_features": in_features,
        "hidden_dim": int(model_cfg.get("hidden_dim", 64)),
        "out_classes": int(model_cfg.get("out_classes", 2)),
        "dropout": float(model_cfg.get("dropout", 0.3)),
        "grid_size": int(model_cfg.get("grid_size", 5)),
        "spline_order": int(model_cfg.get("spline_order", 3)),
        "num_gnn_layers": int(model_cfg.get("num_gnn_layers", 2)),
        "head_type": str(model_cfg.get("head_type", "kan")),
        "gnn_layer_type": str(model_cfg.get("gnn_layer_type", "standard")),
        "use_hierarchical_pooling": bool(model_cfg.get("use_hierarchical_pooling", True)),
        "pooling_clusters": list(model_cfg.get("pooling_clusters", [50, 15])),
        "pooling_type": str(model_cfg.get("pooling_type", "signature")),
        "tabular_dim": int(model_cfg.get("tabular_dim", 0)),
        "tabular_hidden": int(model_cfg.get("tabular_hidden", 32)),
        "roi_feature_dim": base_in_features,
        "fma_Init_dim": int(data_cfg.get("fma_embedding_dim", 768)),
    }

    return dataset_kwargs, model_kwargs, graph_mode

