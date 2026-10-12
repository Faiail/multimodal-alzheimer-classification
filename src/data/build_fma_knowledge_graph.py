"""Build the compact FMA subgraph used by the GNN datasets.

The full FMA OWL is intentionally reduced to the entities reachable from AAL3
mappings through taxonomic (is_a) and structural (*part_of) relations. The
result is a small, deterministic JSON artifact loaded during training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
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

OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_CLASS = f"{{{OWL_NS}}}Class"
OWL_RESTRICTION = f"{{{OWL_NS}}}Restriction"
OWL_ON_PROPERTY = f"{{{OWL_NS}}}onProperty"
OWL_SOME_VALUES_FROM = f"{{{OWL_NS}}}someValuesFrom"
RDFS_LABEL = f"{{{RDFS_NS}}}label"
RDFS_SUBCLASS_OF = f"{{{RDFS_NS}}}subClassOf"
FMA_DEFINITION = f"{{{FMA_PREFIX}}}definition"
RDF_ABOUT = f"{{{RDF_NS}}}about"
RDF_RESOURCE = f"{{{RDF_NS}}}resource"


def fma_id(iri: str) -> str:
    local = iri.rsplit("/", 1)[-1]
    return local[3:] if local.startswith("fma") else local


def canonical_numeric_id(value: str) -> str:
    """Normalize CSV/FMA numeric IDs without assuming fixed zero padding."""
    raw = str(value).strip()
    try:
        numeric = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric FMA ID: {value!r}") from exc
    if not numeric.is_finite() or numeric != numeric.to_integral_value():
        raise ValueError(f"Invalid integral FMA ID: {value!r}")
    return str(int(numeric))


def build_numeric_iri_index(nodes: dict[str, dict]) -> dict[str, tuple[str, ...]]:
    """Index actual OWL IRIs by canonical numeric ID, preserving collisions."""
    grouped: dict[str, list[str]] = {}
    for iri in nodes:
        if not iri.startswith(FMA_PREFIX):
            continue
        local_id = fma_id(iri)
        if not local_id.isdigit():
            continue
        canonical = str(int(local_id))
        grouped.setdefault(canonical, []).append(iri)
    return {
        canonical: tuple(sorted(iris))
        for canonical, iris in grouped.items()
    }


def normalize_label(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()


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


def _element_text(element: ET.Element) -> str:
    return "".join(element.itertext()).strip()


def _parse_named_class(
    element: ET.Element,
    nodes: dict[str, dict],
    edges: dict[str, set[tuple[str, str]]],
    label_index: dict[str, str],
):
    iri = element.get(RDF_ABOUT)
    if not iri:
        return
    info = nodes.setdefault(iri, {"label": None, "definition": None})

    for child in element:
        if child.tag == RDFS_LABEL and info["label"] is None:
            label = _element_text(child)
            if label:
                info["label"] = label
                label_index.setdefault(label.lower(), iri)
        elif child.tag == FMA_DEFINITION and info["definition"] is None:
            definition = _element_text(child)
            if definition:
                info["definition"] = definition
        elif child.tag == RDFS_SUBCLASS_OF:
            target = child.get(RDF_RESOURCE)
            if target and target.startswith(FMA_PREFIX):
                edges["is_a"].add((iri, target))

            for restriction in child.iter(OWL_RESTRICTION):
                on_property = restriction.find(OWL_ON_PROPERTY)
                property_iri = (
                    on_property.get(RDF_RESOURCE)
                    if on_property is not None else None
                )
                relation = normalize_property(property_iri)
                if relation is None:
                    continue
                for value in restriction.iter(OWL_SOME_VALUES_FROM):
                    target = value.get(RDF_RESOURCE)
                    if target and target.startswith(FMA_PREFIX):
                        edges[relation].add((iri, target))


def parse_fma_owl(path: Path):
    """Structurally stream named FMA classes and the selected hierarchy."""
    nodes: dict[str, dict] = {}
    edges: dict[str, set[tuple[str, str]]] = {"is_a": set(), "part_of": set()}
    label_index: dict[str, str] = {}
    stack: list[ET.Element] = []

    started = time.time()
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            stack.append(element)
            continue

        if element.tag == OWL_CLASS and element.get(RDF_ABOUT):
            _parse_named_class(element, nodes, edges, label_index)

        # Clear complete top-level RDF elements only after their nested content
        # has been consumed. This keeps streaming memory bounded without erasing
        # anonymous classes/restrictions before their named parent is processed.
        if len(stack) == 2:
            element.clear()
        stack.pop()

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

    numeric_iri_index = build_numeric_iri_index(nodes)
    roi_to_iri = {}
    unresolved = []
    label_mismatches = []
    seen_roi_ids = set()
    for row in rows:
        roi_id = str(row["label_id"])
        if roi_id in seen_roi_ids:
            raise ValueError(f"Duplicate ROI label_id in lookup: {roi_id}")
        seen_roi_ids.add(roi_id)

        raw_id = (row.get("fma_id") or "").strip()
        if not raw_id:
            continue
        canonical_id = canonical_numeric_id(raw_id)
        candidates = numeric_iri_index.get(canonical_id, ())
        expected_label = normalize_label(row.get("fma_label"))
        if len(candidates) == 1:
            iri = candidates[0]
        elif len(candidates) > 1:
            label_matches = [
                candidate for candidate in candidates
                if normalize_label(nodes[candidate].get("label")) == expected_label
            ]
            if len(label_matches) != 1:
                unresolved.append((roi_id, raw_id, tuple(candidates)))
                continue
            iri = label_matches[0]
        else:
            unresolved.append((roi_id, raw_id))
            continue

        actual_label = normalize_label(nodes[iri].get("label"))
        if expected_label and actual_label != expected_label:
            label_mismatches.append(
                (roi_id, raw_id, row.get("fma_label"), nodes[iri].get("label"))
            )
        roi_to_iri[roi_id] = iri

    if unresolved:
        raise ValueError(f"Unresolved FMA mappings ({len(unresolved)}): {unresolved}")
    if label_mismatches:
        raise ValueError(
            f"FMA mapping label mismatches ({len(label_mismatches)}): "
            f"{label_mismatches}"
        )

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
