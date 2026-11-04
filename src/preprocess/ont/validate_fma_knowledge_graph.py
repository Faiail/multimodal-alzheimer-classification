"""Exhaustively validate the compact FMA artifact against the source OWL."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.sax
from collections import defaultdict, deque
from pathlib import Path
from xml.sax.handler import ContentHandler, feature_namespaces

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.ont.build_fma_knowledge_graph import (  # noqa: E402
    DEFAULT_LOOKUP,
    DEFAULT_OUTPUT,
    DEFAULT_OWL,
    DIVISION_ENTITY_LABELS,
    FMA_PREFIX,
    LOBE_ENTITY_LABELS,
    canonical_numeric_id,
    fma_id,
    normalize_label,
    parse_fma_owl,
)

OWL_NS = "http://www.w3.org/2002/07/owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
FMA_DEFINITION_NS = FMA_PREFIX


def _is_part_of_property(iri: str | None) -> bool:
    if not iri:
        return False
    local = iri.rsplit("/", 1)[-1]
    return local == "part_of" or local.endswith("_part_of")


class IndependentFMAHandler(ContentHandler):
    """SAX implementation independent of the builder's ElementTree parser."""

    def __init__(self):
        super().__init__()
        self.nodes: dict[str, dict] = {}
        self.edges = {"is_a": set(), "part_of": set()}
        self.class_stack = []
        self.subclass_stack = []
        self.restriction_stack = []
        self.capture = None

    @staticmethod
    def _attr(attrs, namespace, local):
        return attrs.get((namespace, local))

    def startElementNS(self, name, qname, attrs):
        if name == (OWL_NS, "Class"):
            iri = self._attr(attrs, RDF_NS, "about")
            context = {"iri": iri, "label": None, "definition": None}
            self.class_stack.append(context)
            if iri:
                self.nodes.setdefault(
                    iri, {"label": None, "definition": None}
                )
            return

        if name == (RDFS_NS, "subClassOf"):
            owner = self.class_stack[-1]["iri"] if self.class_stack else None
            self.subclass_stack.append({"owner": owner})
            target = self._attr(attrs, RDF_NS, "resource")
            if owner and target and target.startswith(FMA_PREFIX):
                self.edges["is_a"].add((owner, target))
            return

        if name == (OWL_NS, "Restriction"):
            owner = (
                self.subclass_stack[-1]["owner"]
                if self.subclass_stack else None
            )
            self.restriction_stack.append({"owner": owner, "property": None})
            return

        if name == (OWL_NS, "onProperty") and self.restriction_stack:
            self.restriction_stack[-1]["property"] = self._attr(
                attrs, RDF_NS, "resource"
            )
            return

        if name == (OWL_NS, "someValuesFrom") and self.restriction_stack:
            restriction = self.restriction_stack[-1]
            target = self._attr(attrs, RDF_NS, "resource")
            if (
                restriction["owner"]
                and target
                and target.startswith(FMA_PREFIX)
                and _is_part_of_property(restriction["property"])
            ):
                self.edges["part_of"].add((restriction["owner"], target))
            return

        if name in {
            (RDFS_NS, "label"),
            (FMA_DEFINITION_NS, "definition"),
        } and self.class_stack and self.class_stack[-1]["iri"]:
            field = "label" if name == (RDFS_NS, "label") else "definition"
            self.capture = {
                "name": name,
                "field": field,
                "iri": self.class_stack[-1]["iri"],
                "text": [],
            }

    def characters(self, content):
        if self.capture is not None:
            self.capture["text"].append(content)

    def endElementNS(self, name, qname):
        if self.capture is not None and name == self.capture["name"]:
            value = "".join(self.capture["text"]).strip()
            iri = self.capture["iri"]
            field = self.capture["field"]
            if value and self.nodes[iri][field] is None:
                self.nodes[iri][field] = value
            self.capture = None

        if name == (OWL_NS, "Restriction"):
            self.restriction_stack.pop()
        elif name == (RDFS_NS, "subClassOf"):
            self.subclass_stack.pop()
        elif name == (OWL_NS, "Class"):
            self.class_stack.pop()


def independently_parse_owl(path: Path):
    parser = xml.sax.make_parser()
    parser.setFeature(feature_namespaces, True)
    handler = IndependentFMAHandler()
    parser.setContentHandler(handler)
    parser.parse(str(path))
    return handler.nodes, handler.edges


def _numeric_index(nodes):
    result = defaultdict(list)
    for iri in nodes:
        if iri.startswith(FMA_PREFIX) and fma_id(iri).isdigit():
            result[str(int(fma_id(iri)))].append(iri)
    return {key: tuple(sorted(values)) for key, values in result.items()}


def _resolve_all_mappings(rows, nodes):
    index = _numeric_index(nodes)
    expected = {}
    errors = []
    for row in rows:
        roi_id = str(row["label_id"])
        canonical = canonical_numeric_id(row["fma_id"])
        candidates = index.get(canonical, ())
        expected_label = normalize_label(row.get("fma_label"))
        matches = [
            iri for iri in candidates
            if normalize_label(nodes[iri].get("label")) == expected_label
        ]
        if len(candidates) == 1 and len(matches) == 1:
            expected[roi_id] = candidates[0]
        elif len(matches) == 1:
            expected[roi_id] = matches[0]
        else:
            errors.append((roi_id, row["fma_id"], row.get("fma_label"), candidates))
    if errors:
        raise AssertionError(f"Unresolvable/ambiguous lookup mappings: {errors}")
    return expected


def _category_edges(rows, label_index, roi_to_iri):
    edges = set()
    targets = set()
    for row in rows:
        source = roi_to_iri[str(row["label_id"])]
        labels = []
        if row.get("fma_lobe") in LOBE_ENTITY_LABELS:
            labels.append(LOBE_ENTITY_LABELS[row["fma_lobe"]])
        if row.get("fma_division") in DIVISION_ENTITY_LABELS:
            labels.append(DIVISION_ENTITY_LABELS[row["fma_division"]])
        for label in labels:
            candidates = label_index.get(normalize_label(label), ())
            if len(candidates) != 1:
                raise AssertionError(
                    f"Category label {label!r} resolves to {candidates}"
                )
            target = candidates[0]
            if target != source:
                edges.add((source, target))
                targets.add(target)
    return edges, targets


def _closure(seeds, full_edges, max_depth):
    adjacency = {relation: defaultdict(list) for relation in full_edges}
    for relation, edges in full_edges.items():
        for source, target in edges:
            adjacency[relation][source].append(target)
    included = set(seeds)
    best_depth = {iri: 0 for iri in seeds}
    queue = deque((iri, 0) for iri in sorted(seeds))
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
    return included


def _assert_equal_sets(name, actual, expected):
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise AssertionError(
            f"{name} mismatch: missing={len(missing)} {sorted(missing)[:10]}, "
            f"extra={len(extra)} {sorted(extra)[:10]}"
        )


def validate(owl_path: Path, lookup_path: Path, artifact_path: Path):
    independent_nodes, independent_edges = independently_parse_owl(owl_path)
    builder_nodes, builder_edges, _ = parse_fma_owl(owl_path)
    _assert_equal_sets("all named OWL classes", set(builder_nodes), set(independent_nodes))
    for relation in ("is_a", "part_of"):
        _assert_equal_sets(
            f"all global {relation} edges",
            builder_edges[relation],
            independent_edges[relation],
        )

    rows = list(csv.DictReader(lookup_path.open(newline="", encoding="utf-8")))
    if len(rows) != 170:
        raise AssertionError(f"Expected 170 ROI lookup rows, found {len(rows)}")
    expected_mapping = _resolve_all_mappings(rows, independent_nodes)
    if len(expected_mapping) != len(rows):
        raise AssertionError("Not every ROI lookup row resolved")

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact_nodes = artifact["nodes"]
    id_to_iri = {node["id"]: node["iri"] for node in artifact_nodes}
    if len(id_to_iri) != len(artifact_nodes):
        raise AssertionError("Duplicate entity IDs in compact artifact")
    for node in artifact_nodes:
        iri = node["iri"]
        if iri not in independent_nodes:
            raise AssertionError(f"Artifact entity absent from FMA: {iri}")
        if node["id"] != fma_id(iri):
            raise AssertionError(f"Artifact ID/IRI mismatch: {node}")
        if normalize_label(node["label"]) != normalize_label(
            independent_nodes[iri].get("label")
        ):
            raise AssertionError(f"Artifact label mismatch for {iri}")

    actual_mapping = {
        roi: id_to_iri[entity_id]
        for roi, entity_id in artifact["roi_to_entity"].items()
    }
    if actual_mapping != expected_mapping:
        missing = set(expected_mapping.items()) - set(actual_mapping.items())
        extra = set(actual_mapping.items()) - set(expected_mapping.items())
        raise AssertionError(f"ROI mapping mismatch: missing={missing}, extra={extra}")

    label_index = defaultdict(list)
    for iri, info in independent_nodes.items():
        if info.get("label"):
            label_index[normalize_label(info["label"])].append(iri)
    label_index = {key: tuple(values) for key, values in label_index.items()}
    category_edges, category_targets = _category_edges(
        rows, label_index, expected_mapping
    )
    included = _closure(
        set(expected_mapping.values()) | category_targets,
        independent_edges,
        int(artifact["max_depth"]),
    )
    _assert_equal_sets(
        "compact entity closure",
        {node["iri"] for node in artifact_nodes},
        included,
    )

    expected_edges = {
        relation: {
            edge for edge in independent_edges[relation]
            if edge[0] in included and edge[1] in included
        }
        for relation in ("is_a", "part_of")
    }
    expected_edges["part_of"] |= category_edges
    actual_edges = {}
    for relation in ("is_a", "part_of"):
        actual_edges[relation] = {
            (id_to_iri[source], id_to_iri[target])
            for source, target in artifact["edges"][relation]
        }
        _assert_equal_sets(
            f"compact {relation} edges",
            actual_edges[relation],
            expected_edges[relation],
        )

    if expected_mapping["131"] != expected_mapping["132"]:
        raise AssertionError("ROI 131 and 132 do not map to the same FMA entity")
    intralaminar_iri = expected_mapping["131"]
    if intralaminar_iri != FMA_PREFIX + "fma0327261":
        raise AssertionError(f"Unexpected intralaminar IRI: {intralaminar_iri}")
    if normalize_label(independent_nodes[intralaminar_iri]["label"]) != normalize_label(
        "Intralaminar nucleus of thalamus"
    ):
        raise AssertionError("Unexpected intralaminar entity label")

    expected_stats = {
        "mapped_rois": len(expected_mapping),
        "entities": len(included),
        "is_a_edges": len(expected_edges["is_a"]),
        "part_of_edges": len(expected_edges["part_of"]),
        "inferred_category_edges": len(category_edges),
    }
    if artifact["stats"] != expected_stats:
        raise AssertionError(
            f"Artifact stats mismatch: {artifact['stats']} != {expected_stats}"
        )

    features_path = PROJECT_ROOT / "data" / "processed" / "all_features.csv"
    coords_path = PROJECT_ROOT / "configs" / "aal3_coordinates.csv"
    if features_path.exists() and coords_path.exists():
        with features_path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        feature_ids = {
            int(column.rsplit("_", 1)[-1])
            for column in header if column.startswith("GM_vol_Region_")
        }
        with coords_path.open(newline="", encoding="utf-8") as handle:
            coordinate_ids = {int(row["label_id"]) for row in csv.DictReader(handle)}
        active_ids = sorted(feature_ids & coordinate_ids)
        unmapped_active = [roi for roi in active_ids if str(roi) not in actual_mapping]
        if len(active_ids) != 165 or unmapped_active:
            raise AssertionError(
                f"Active ROI validation failed: count={len(active_ids)}, "
                f"unmapped={unmapped_active}"
            )
    else:
        active_ids = sorted(int(k) for k in actual_mapping.keys())

    report = {
        "global_named_classes": len(independent_nodes),
        "global_is_a_edges": len(independent_edges["is_a"]),
        "global_part_of_edges": len(independent_edges["part_of"]),
        "lookup_mappings": len(expected_mapping),
        "active_roi_mappings": len(active_ids),
        **expected_stats,
        "roi_131_entity": fma_id(expected_mapping["131"]),
        "roi_132_entity": fma_id(expected_mapping["132"]),
    }
    print(json.dumps(report, indent=2))
    print("PASS exhaustive FMA mapping and relation validation")
    return report


class FMAKnowledgeGraphValidator:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        ont_cfg = parameters.get("ontology", parameters)
        self.owl_path = Path(ont_cfg.get("owl_path", DEFAULT_OWL))
        self.lookup_path = Path(ont_cfg.get("lookup_path", DEFAULT_LOOKUP))
        self.artifact_path = Path(ont_cfg.get("artifact_path", DEFAULT_OUTPUT))

    def launch(self):
        return validate(self.owl_path, self.lookup_path, self.artifact_path)

    def __call__(self):
        return self.launch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", type=Path, default=DEFAULT_OWL)
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    validate(args.owl, args.lookup, args.artifact)


if __name__ == "__main__":
    main()
