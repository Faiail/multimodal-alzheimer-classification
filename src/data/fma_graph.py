"""Runtime representation of the compact FMA knowledge graph."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "configs" / "fma_knowledge_graph.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "fma_embeddings"
DEFAULT_BIOMED_MODEL = "NeuML/pubmedbert-base-embeddings"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _resolve_project_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _hash_embeddings(texts: list[str], dimension: int) -> torch.Tensor:
    """Deterministic, dependency-free text vectors used by smoke tests."""
    if dimension <= 0:
        raise ValueError("fma_embedding_dim must be positive")
    result = torch.zeros((len(texts), dimension), dtype=torch.float32)
    for row, text in enumerate(texts):
        tokens = _TOKEN_RE.findall(text.lower()) or ["unknown"]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "little") % dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            result[row, index] += sign
        result[row] /= math.sqrt(len(tokens))
    return F.normalize(result, p=2, dim=-1)


def _biomedical_embeddings(texts: list[str], model_name: str) -> torch.Tensor:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The PubMedBERT FMA embedding backend requires sentence-transformers. "
            "Install requirements.txt or use fma_embedding_backend='hash' for smoke tests."
        ) from exc
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return embeddings.detach().cpu().to(torch.float32)


def _load_or_create_embeddings(
    artifact_path: Path,
    nodes: list[dict],
    backend: str,
    model_name: str,
    hash_dimension: int,
    cache_dir: Path,
) -> torch.Tensor:
    texts = [
        f"{node['label']}. {node.get('definition', '')}".strip()
        for node in nodes
    ]
    fingerprint = hashlib.sha256()
    fingerprint.update(artifact_path.read_bytes())
    fingerprint.update(backend.encode())
    fingerprint.update(model_name.encode())
    fingerprint.update(str(hash_dimension).encode())
    cache_key = fingerprint.hexdigest()[:20]
    cache_path = cache_dir / f"fma_{backend}_{cache_key}.pt"
    if cache_path.exists():
        embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
        if embeddings.shape[0] != len(nodes):
            raise ValueError(f"Stale FMA embedding cache: {cache_path}")
        return embeddings.to(torch.float32)

    if backend == "hash":
        embeddings = _hash_embeddings(texts, hash_dimension)
    elif backend in {"pubmedbert", "sentence_transformer"}:
        embeddings = _biomedical_embeddings(texts, model_name)
    else:
        raise ValueError(
            f"Unsupported fma_embedding_backend={backend!r}; use 'pubmedbert' or 'hash'"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, cache_path)
    return embeddings


class FMAKnowledgeGraph:
    """Patient-independent FMA entity graph, filtered to the active ROI atlas."""

    _embedding_cache: dict[tuple, torch.Tensor] = {}

    def __init__(
        self,
        roi_label_ids,
        graph_path: str | Path | None = None,
        embedding_backend: str = "pubmedbert",
        embedding_model: str = DEFAULT_BIOMED_MODEL,
        embedding_dim: int = 128,
        cache_dir: str | Path | None = None,
    ):
        self.graph_path = _resolve_project_path(graph_path, DEFAULT_GRAPH_PATH)
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"FMA graph artifact not found: {self.graph_path}. "
                "Run python src/data/build_fma_knowledge_graph.py"
            )
        artifact = json.loads(self.graph_path.read_text(encoding="utf-8"))
        if artifact.get("version") != 1:
            raise ValueError(f"Unsupported FMA graph artifact version: {artifact.get('version')}")

        all_nodes = artifact["nodes"]
        all_node_index = {node["id"]: index for index, node in enumerate(all_nodes)}
        forward_edges = {
            relation: [tuple(edge) for edge in artifact["edges"].get(relation, [])]
            for relation in ("is_a", "part_of")
        }
        roi_to_entity = artifact["roi_to_entity"]
        active_roi_ids = [str(int(value)) for value in roi_label_ids]
        mapped_seeds = {
            roi_to_entity[roi_id] for roi_id in active_roi_ids if roi_id in roi_to_entity
        }

        adjacency = {}
        for relation_edges in forward_edges.values():
            for source, target in relation_edges:
                adjacency.setdefault(source, set()).add(target)
        included = set(mapped_seeds)
        frontier = list(mapped_seeds)
        while frontier:
            source = frontier.pop()
            for target in adjacency.get(source, ()):
                if target not in included:
                    included.add(target)
                    frontier.append(target)

        ordered_ids = [node["id"] for node in all_nodes if node["id"] in included]
        self.entity_ids = ordered_ids
        self.entity_labels = [node["label"] for node in all_nodes if node["id"] in included]
        self.entity_index = {entity_id: index for index, entity_id in enumerate(ordered_ids)}
        full_cache_key = (
            str(self.graph_path), self.graph_path.stat().st_mtime_ns,
            embedding_backend, embedding_model, embedding_dim, str(cache_dir),
        )
        if full_cache_key not in self._embedding_cache:
            self._embedding_cache[full_cache_key] = _load_or_create_embeddings(
                self.graph_path,
                all_nodes,
                embedding_backend,
                embedding_model,
                embedding_dim,
                _resolve_project_path(cache_dir, DEFAULT_CACHE_DIR),
            )
        full_embeddings = self._embedding_cache[full_cache_key]
        selected_rows = [all_node_index[entity_id] for entity_id in ordered_ids]
        self.x = full_embeddings[selected_rows].clone()

        self.edges = {}
        reverse_names = {"is_a": "has_subclass", "part_of": "has_part"}
        for relation, relation_edges in forward_edges.items():
            selected = [
                (self.entity_index[source], self.entity_index[target])
                for source, target in relation_edges
                if source in self.entity_index and target in self.entity_index
            ]
            self.edges[relation] = self._edge_tensor(selected)
            self.edges[reverse_names[relation]] = self.edges[relation].flip(0)

        roi_links = []
        for roi_index, roi_id in enumerate(active_roi_ids):
            entity_id = roi_to_entity.get(roi_id)
            if entity_id in self.entity_index:
                roi_links.append((roi_index, self.entity_index[entity_id]))
        self.roi_to_entity = self._edge_tensor(roi_links)
        self.entity_to_roi = self.roi_to_entity.flip(0)
        self.mapped_roi_count = len(roi_links)

    @staticmethod
    def _edge_tensor(edges) -> torch.Tensor:
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
