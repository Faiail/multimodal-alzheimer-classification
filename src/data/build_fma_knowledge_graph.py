"""Build the compact FMA subgraph used by the GNN datasets.

The full FMA OWL is intentionally reduced to the entities reachable from AAL3
mappings through taxonomic (is_a) and structural (*part_of) relations. The
result is a small, deterministic JSON artifact loaded during training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import time
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OWL = PROJECT_ROOT / "configs" / "FMA" / "fma.owl"
DEFAULT_LOOKUP = PROJECT_ROOT / "configs" / "aal3_ontology_lookup.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "configs" / "fma_knowledge_graph.json"
FMA_PREFIX = "http://purl.org/sig/ont/fma/"

LOBE_ENTITY_LABELS = {
    "Frontal": "Frontal lobe",
    "Parietal": "Parietal lobe",
    "Temporal": "Temporal lobe",
    "Occipital": "Occipital lobe",
    "Limbic": "Limbic lobe",
    "Insula": "Insula",
    "Thalamus": "Thalamus",
    "Subcortical": "Subcortical gray matter",
    "Cerebellum": "Cerebellum",
    "Brainstem": "Brainstem",
}
DIVISION_ENTITY_LABELS = {
    "Cerebral_cortex": "Cerebral cortex",
    "Thalamus": "Thalamus",
    "Subcortical": "Subcortical gray matter",
    "Brainstem": "Brainstem",
    "Cerebellum": "Cerebellum",
}

IRI_RE = re.compile(r'<owl:Class rdf:about="([^"]+)"')
LABEL_RE = re.compile(r'<rdfs:label[^>]*>([^<]+)</rdfs:label>')
DEFINITION_RE = re.compile(r'<fma:definition[^>]*>(.*?)</fma:definition>')
RESOURCE_RE = re.compile(r'rdf:resource="([^"]+)"')


def fma_id(iri: str) -> str:
    local = iri.rsplit("/", 1)[-1]
    return local[3:] if local.startswith("fma") else local


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_property(iri: str | None) -> str | None:
    if not iri:
        return None
    local = iri.rsplit("/", 1)[-1]
    if local == "part_of" or local.endswith("_part_of"):
        return "part_of"
    return None


def parse_fma_owl(path: Path):
    """Stream labels, definitions and the hierarchy needed by the GNN."""
    nodes: dict[str, dict] = {}
    edges: dict[str, set[tuple[str, str]]] = {"is_a": set(), "part_of": set()}
    label_index: dict[str, str] = {}
    current = None
    in_subclass = False
    restriction_property = None

    started = time.time()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            class_match = IRI_RE.search(line)
            if class_match:
                current = class_match.group(1)
                nodes.setdefault(current, {"label": None, "definition": None})
                in_subclass = False
                restriction_property = None
                continue
            if current is None:
                continue

            if "<rdfs:label" in line:
                match = LABEL_RE.search(line)
                if match and nodes[current]["label"] is None:
                    label = html.unescape(match.group(1).strip())
                    nodes[current]["label"] = label
                    label_index.setdefault(label.lower(), current)

            if "<fma:definition" in line:
                match = DEFINITION_RE.search(line)
                if match and nodes[current]["definition"] is None:
                    nodes[current]["definition"] = html.unescape(match.group(1).strip())

            if "<rdfs:subClassOf" in line:
                resource = RESOURCE_RE.search(line)
                if resource:
                    target = resource.group(1)
                    if target.startswith(FMA_PREFIX):
                        edges["is_a"].add((current, target))
                    in_subclass = False
                    restriction_property = None
                else:
                    in_subclass = True
                    restriction_property = None
            elif in_subclass and "<owl:onProperty" in line:
                resource = RESOURCE_RE.search(line)
                restriction_property = resource.group(1) if resource else None
            elif in_subclass and "<owl:someValuesFrom" in line:
                resource = RESOURCE_RE.search(line)
                relation = normalize_property(restriction_property)
                if resource and relation:
                    target = resource.group(1)
                    if target.startswith(FMA_PREFIX):
                        edges[relation].add((current, target))

            if "</rdfs:subClassOf>" in line:
                in_subclass = False
                restriction_property = None
            if "</owl:Class>" in line:
                current = None
                in_subclass = False
                restriction_property = None

    print(
        f"Parsed {len(nodes):,} FMA classes and "
        f"{sum(len(v) for v in edges.values()):,} hierarchy edges "
        f"in {time.time() - started:.1f}s"
    )
    return nodes, edges, label_index


def load_roi_mapping(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_category_edges(rows, label_index, roi_to_iri):
    """Link mapped entities to actual FMA lobe/division entities.

    The lookup's lobe/division fields were inferred by walking FMA itself. These
    explicit edges ensure the compact graph retains those high-level concepts.
    """
    category_edges = set()
    category_iris = set()
    for row in rows:
        source = roi_to_iri.get(str(row.get("label_id", "")))
        if source is None:
            continue
        labels = []
        lobe = row.get("fma_lobe")
        division = row.get("fma_division")
        if lobe in LOBE_ENTITY_LABELS:
            labels.append(LOBE_ENTITY_LABELS[lobe])
        if division in DIVISION_ENTITY_LABELS:
            labels.append(DIVISION_ENTITY_LABELS[division])
        for label in labels:
            target = label_index.get(label.lower())
            if target and target != source:
                category_edges.add((source, target))
                category_iris.add(target)
    return category_edges, category_iris


def build_compact_graph(owl_path: Path, lookup_path: Path, max_depth: int):
    nodes, all_edges, label_index = parse_fma_owl(owl_path)
    rows = load_roi_mapping(lookup_path)

    roi_to_iri = {}
    for row in rows:
        raw_id = (row.get("fma_id") or "").strip()
        if not raw_id:
            continue
        iri = FMA_PREFIX + "fma" + raw_id.removesuffix(".0")
        if iri in nodes:
            roi_to_iri[str(row["label_id"])] = iri

    category_edges, category_iris = resolve_category_edges(rows, label_index, roi_to_iri)
    adjacency = {relation: {} for relation in all_edges}
    for relation, relation_edges in all_edges.items():
        for source, target in relation_edges:
            adjacency[relation].setdefault(source, []).append(target)

    included = set(roi_to_iri.values()) | category_iris
    queue = deque((iri, 0) for iri in sorted(included))
    best_depth = {iri: 0 for iri in included}
    while queue:
        source, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for relation in ("is_a", "part_of"):
            for target in adjacency[relation].get(source, ()):
                next_depth = depth + 1
                if target not in best_depth or next_depth < best_depth[target]:
                    best_depth[target] = next_depth
                    included.add(target)
                    queue.append((target, next_depth))

    compact_edges = {}
    for relation, relation_edges in all_edges.items():
        compact_edges[relation] = sorted(
            (fma_id(source), fma_id(target))
            for source, target in relation_edges
            if source in included and target in included
        )
    compact_edges["part_of"] = sorted(set(compact_edges["part_of"]) | {
        (fma_id(source), fma_id(target)) for source, target in category_edges
    })

    compact_nodes = []
    for iri in sorted(included, key=lambda value: (not fma_id(value).isdigit(), fma_id(value))):
        info = nodes.get(iri, {})
        compact_nodes.append({
            "id": fma_id(iri),
            "iri": iri,
            "label": info.get("label") or fma_id(iri),
            "definition": info.get("definition") or "",
        })

    return {
        "version": 1,
        "source_owl": str(owl_path.relative_to(PROJECT_ROOT)),
        "source_sha256": file_sha256(owl_path),
        "max_depth": max_depth,
        "nodes": compact_nodes,
        "edges": compact_edges,
        "roi_to_entity": {
            roi: fma_id(iri) for roi, iri in sorted(roi_to_iri.items(), key=lambda item: int(item[0]))
        },
        "stats": {
            "mapped_rois": len(roi_to_iri),
            "entities": len(compact_nodes),
            "is_a_edges": len(compact_edges["is_a"]),
            "part_of_edges": len(compact_edges["part_of"]),
            "inferred_category_edges": len(category_edges),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", type=Path, default=DEFAULT_OWL)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()
    graph = build_compact_graph(args.owl, args.lookup, args.max_depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")
    print(json.dumps(graph["stats"], indent=2))


if __name__ == "__main__":
    main()
