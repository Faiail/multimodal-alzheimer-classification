from .brain_graph import plot_brain_graph, plot_brain_graph_with_communities
from .brain_heatmap import plot_brain_heatmap
from .explanation_figures import (
    load_explanation,
    plot_modality_importance,
    plot_patient_comparison,
)
from .fma_figures import (
    has_fma_edge_explanations,
    plot_fma_patient_comparison,
    plot_fma_relation_importance,
)


def run_visualization_task(parameters: dict):
    vis_cfg = parameters.get("visualization", parameters)
    mode = vis_cfg.get("mode", "tables")
    if mode == "tables":
        from .generate_tables import main as generate_tables_main

        return generate_tables_main()
    if mode == "methodology":
        from .methodology_figures import main as methodology_main

        return methodology_main()
    if mode == "images":
        from .plot_images import main as plot_images_main

        return plot_images_main()
    raise ValueError(f"Unsupported visualization mode '{mode}'.")


__all__ = [
    "has_fma_edge_explanations",
    "load_explanation",
    "plot_brain_graph",
    "plot_brain_graph_with_communities",
    "plot_brain_heatmap",
    "plot_fma_patient_comparison",
    "plot_fma_relation_importance",
    "plot_modality_importance",
    "plot_patient_comparison",
    "run_visualization_task",
]

