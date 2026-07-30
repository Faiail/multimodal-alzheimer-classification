"""
AAL3 → FMA ontology mapping script (v2 — streaming, label-based).

Instead of numeric-ID IRI lookup (which caused substring collisions, e.g.
fma261851 matched before fma61851), this script:
  1. Streams fma.owl once to build:
       label_index : lowercase_label → (canonical_label, IRI)
       graph       : IRI → {label, is_a_iris, part_of_iris}
  2. For each AAL3 region, looks up the correct FMA class by its
     canonical anatomical label (AAL3_TO_FMA_LABEL dict below).
  3. Walks is_a / part_of chains to infer lobe and division.
  4. Saves configs/aal3_ontology_lookup.csv.

Sources:
  - Rolls ET et al. (2020) NeuroImage 206:116189 [AAL3]
  - Rosse C & Mejino JLV (2003) J Biomed Inform 36:478-500 [FMA]

Run with:
    source .venv/bin/activate
    python3 src/data/build_fma_mapping.py
"""
import re
import time
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FMA_PATH     = PROJECT_ROOT / "configs" / "FMA" / "fma.owl"
AAL3_TXT     = PROJECT_ROOT / "configs" / "AAL3" / "AAL3v1.nii.txt"
OUTPUT_CSV   = PROJECT_ROOT / "configs" / "aal3_ontology_lookup.csv"

# ──────────────────────────────────────────────────────────────────────────────
# AAL3 base-name → canonical FMA label (searched case-insensitively).
# None means no reliable FMA equivalent.
# ──────────────────────────────────────────────────────────────────────────────
AAL3_TO_FMA_LABEL = {
    # ── Frontal lobe ──────────────────────────────────────────────────────────
    "Precentral":           "Precentral gyrus",
    "Frontal_Sup_2":        "Superior frontal gyrus",
    "Frontal_Mid_2":        "Middle frontal gyrus",
    "Frontal_Inf_Oper":     "Inferior frontal gyrus",   # pars opercularis — FMA maps all three IFG parts to parent class
    "Frontal_Inf_Tri":      "Inferior frontal gyrus",   # pars triangularis
    "Frontal_Inf_Orb_2":    "Inferior frontal gyrus",   # pars orbitalis
    "Rolandic_Oper":        "Frontoparietal operculum",
    "Supp_Motor_Area":      "Superior frontal gyrus",      # SMA ≈ medial SFG in FMA
    "Olfactory":            "Straight gyrus",
    "Frontal_Sup_Medial":   "Superior frontal gyrus",
    "Frontal_Med_Orb":      "Medial orbital gyrus",
    "Rectus":               "Straight gyrus",
    "OFCmed":               "Medial orbital gyrus",
    "OFCant":               "Anterior orbital gyrus",
    "OFCpost":              "Posterior orbital gyrus",
    "OFClat":               "Lateral orbital gyrus",
    "Insula":               "Insula",
    # ── Cingulate / Limbic ────────────────────────────────────────────────────
    "Cingulate_Ant":        "Cingulate gyrus",
    "Cingulate_Mid":        "Cingulate gyrus",
    "Cingulate_Post":       "Cingulate gyrus",
    "ACC_sub":              "Cingulate gyrus",
    "ACC_pre":              "Cingulate gyrus",
    "ACC_sup":              "Cingulate gyrus",
    # ── Medial temporal / hippocampal ─────────────────────────────────────────
    "Hippocampus":          "Hippocampus",
    "ParaHippocampal":      "Parahippocampal gyrus",
    "Amygdala":             "Amygdala",
    # ── Occipital lobe ────────────────────────────────────────────────────────
    "Calcarine":            "Intracalcarine cortex",    # FMA equivalent: primary visual cortex in calcarine
    "Cuneus":               "Cuneus",
    "Lingual":              "Lingual gyrus",
    "Occipital_Sup":        "Superior occipital gyrus",
    "Occipital_Mid":        "Middle occipital gyrus",
    "Occipital_Inf":        "Inferior occipital gyrus",
    "Fusiform":             "Fusiform gyrus",
    # ── Parietal lobe ─────────────────────────────────────────────────────────
    "Postcentral":          "Postcentral gyrus",
    "Parietal_Sup":         "Superior parietal lobule",
    "Parietal_Inf":         "Inferior parietal lobule",
    "SupraMarginal":        "Supramarginal gyrus",
    "Angular":              "Angular gyrus",
    "Precuneus":            "Precuneus",
    "Paracentral_Lobule":   "Paracentral lobule",
    # ── Temporal lobe ─────────────────────────────────────────────────────────
    "Heschl":               "Transverse temporal gyrus",
    "Temporal_Sup":         "Superior temporal gyrus",
    "Temporal_Pole_Sup":    "Temporal pole",
    "Temporal_Mid":         "Middle temporal gyrus",
    "Temporal_Pole_Mid":    "Temporal pole",
    "Temporal_Inf":         "Inferior temporal gyrus",
    # ── Subcortical / Basal ganglia ───────────────────────────────────────────
    "Caudate":              "Caudate nucleus",
    "Putamen":              "Putamen",
    "Pallidum":             "Globus pallidus",
    "N_Acc":                "Nucleus accumbens",
    # ── Thalamus ──────────────────────────────────────────────────────────────
    "Thalamus":             "Thalamus",
    "Thal_AV":              "Anteroventral nucleus",              # FMA: short form (no 'of thalamus')
    "Thal_LP":              "Lateral posterior nucleus",          # FMA: 'Lateral posterior nucleus'
    "Thal_VA":              "Ventral anterior nucleus",           # FMA: 'Ventral anterior nucleus'
    "Thal_VL":              "Ventral lateral nucleus",            # FMA: 'Ventral lateral nucleus'
    "Thal_VPL":             "Ventral posterolateral nucleus",     # FMA: 'Ventral posterolateral nucleus'
    "Thal_IL":              "Intralaminar nucleus of thalamus",   # FMA: found with 'of thalamus'
    "Thal_Re":              "Thalamic reticular nucleus",         # FMA: 'Thalamic reticular nucleus'
    "Thal_MDm":             "Medial dorsal nucleus of thalamus",  # FMA: 'Medial dorsal nucleus of thalamus'
    "Thal_MDl":             "Medial dorsal nucleus of thalamus",  # FMA: same class (lateral part = subdivision)
    "Thal_LGN":             "Lateral geniculate body",            # FMA: 'body' not 'nucleus'
    "Thal_MGN":             "Medial geniculate body",             # FMA: 'body' not 'nucleus'
    "Thal_PuI":             "Inferior pulvinar nucleus",          # confirmed
    "Thal_PuM":             "Medial pulvinar nucleus",            # confirmed
    "Thal_PuA":             "Oral pulvinar nucleus",              # FMA equivalent of anterior pulvinar
    "Thal_PuL":             "Lateral pulvinar nucleus",           # confirmed
    # ── Cerebellum ────────────────────────────────────────────────────────────
    "Cerebellum_Crus1":     "Lobule VIIA of hemisphere of cerebellum",  # Crus I = FMA Lobule VIIA
    "Cerebellum_Crus2":     "Gracile lobule",                            # Crus II = FMA Gracile lobule (VIIB)
    "Cerebellum_3":         "Lobule III of hemisphere of cerebellum",
    "Cerebellum_4_5":       "Lobule IV of hemisphere of cerebellum",
    "Cerebellum_6":         "Simple lobule of cerebellum",               # Lobule VI = FMA Simple lobule
    "Cerebellum_7b":        "Gracile lobule",                            # Lobule VIIb = FMA Gracile lobule
    "Cerebellum_8":         "Tonsil of cerebellum",                      # Lobule VIII = FMA Tonsil
    "Cerebellum_9":         "Lobule IX of hemisphere of cerebellum",
    "Cerebellum_10":        "Flocculus of cerebellum",                   # Lobule X = FMA Flocculus
    "Vermis_1_2":           "Lingula of vermis of cerebellum",    # FMA: lobulo I-II = Lingula
    "Vermis_3":             "Lobule III of vermis",
    "Vermis_4_5":           "Culmen of vermis of cerebellum",     # FMA: lobuli IV-V = Culmen
    "Vermis_6":             "Declive of vermis of cerebellum",    # FMA: lobulo VI = Declive
    "Vermis_7":             "Lobule VII of vermis",
    "Vermis_8":             "Pyramis of vermis of cerebellum",    # FMA: lobulo VIII = Pyramis
    "Vermis_9":             "Uvula of vermis of cerebellum",      # FMA: lobulo IX = Uvula
    "Vermis_10":            "Nodule of vermis of cerebellum",     # FMA: lobulo X = Noduloe
    # ── Brainstem ─────────────────────────────────────────────────────────────
    "VTA":                  "Ventral tegmental area",
    "SN_pc":                "Substantia nigra",
    "SN_pr":                "Substantia nigra",
    "Red_N":                "Red nucleus",
    "LC":                   "Locus ceruleus",
    "Raphe_D":              "Dorsal raphe nucleus",
    "Raphe_M":              "Median raphe nucleus",
}

# ──────────────────────────────────────────────────────────────────────────────
# Keyword rules to infer lobe and division from hierarchy labels
# ──────────────────────────────────────────────────────────────────────────────
LOBE_RULES = [
    ("frontal",          "Frontal"),
    ("parietal",         "Parietal"),
    ("temporal",         "Temporal"),
    ("occipital",        "Occipital"),
    ("cingulate",        "Limbic"),
    ("hippocamp",        "Limbic"),
    ("parahippoc",       "Limbic"),
    ("amygdala",         "Limbic"),
    ("limbic",           "Limbic"),
    ("insula",           "Insula"),
    ("thalamus",         "Thalamus"),
    ("thalamic",         "Thalamus"),
    ("pulvinar",         "Thalamus"),
    ("geniculate",       "Thalamus"),
    ("cerebellum",       "Cerebellum"),
    ("cerebellar",       "Cerebellum"),
    # FMA cerebellar structures that don't contain 'cerebellum' in the label
    ("gracile lobule",   "Cerebellum"),
    ("simple lobule",    "Cerebellum"),
    ("tonsil of",        "Cerebellum"),
    ("flocculus",        "Cerebellum"),
    ("nodule of",        "Cerebellum"),
    ("lingula of",       "Cerebellum"),
    ("ansiform",         "Cerebellum"),
    ("lobule of vermis", "Cerebellum"),
    ("lobule vii",       "Cerebellum"),
    ("central lobule",   "Cerebellum"),
    ("midbrain",         "Brainstem"),
    ("pons",             "Brainstem"),
    ("medulla",          "Brainstem"),
    ("brainstem",        "Brainstem"),
    ("raphe",            "Brainstem"),
    ("substantia nigra", "Brainstem"),
    ("tegmental",        "Brainstem"),
    ("locus",            "Brainstem"),
    ("caudate",          "Subcortical"),
    ("putamen",          "Subcortical"),
    ("globus pallidus",  "Subcortical"),
    ("accumbens",        "Subcortical"),
    ("basal gangl",      "Subcortical"),
    ("striatum",         "Subcortical"),
    ("straight gyrus",   "Frontal"),
    ("orbital gyrus",    "Frontal"),
    ("precentral",       "Frontal"),
    ("postcentral",      "Parietal"),
    ("precuneus",        "Parietal"),
    ("cuneus",           "Occipital"),
    ("lingual",          "Occipital"),
    ("fusiform",         "Occipital"),
    ("calcarine",        "Occipital"),
    ("operculum",        "Frontal"),
]

DIVISION_RULES = [
    ("operculum",           "Cerebral_cortex"),
    # Cerebellar structures come BEFORE generic 'lobule' to avoid misclassification
    ("cerebellum",          "Cerebellum"),
    ("cerebellar",          "Cerebellum"),
    ("gracile lobule",      "Cerebellum"),
    ("simple lobule",       "Cerebellum"),
    ("tonsil of",           "Cerebellum"),
    ("flocculus",           "Cerebellum"),
    ("nodule of",           "Cerebellum"),
    ("lingula of",          "Cerebellum"),
    ("lobule of vermis",    "Cerebellum"),
    ("central lobule",      "Cerebellum"),
    ("thalamus",            "Thalamus"),
    ("thalamic",            "Thalamus"),
    ("pulvinar",            "Thalamus"),
    ("geniculate",          "Thalamus"),
    ("geniculate body",     "Thalamus"),
    ("caudate",             "Subcortical"),
    ("putamen",             "Subcortical"),
    ("globus pallidus",     "Subcortical"),
    ("accumbens",           "Subcortical"),
    ("basal gangl",         "Subcortical"),
    ("striatum",            "Subcortical"),
    ("hippocamp",           "Subcortical"),
    ("amygdala",            "Subcortical"),
    ("midbrain",            "Brainstem"),
    ("pons",                "Brainstem"),
    ("medulla",             "Brainstem"),
    ("brainstem",           "Brainstem"),
    ("raphe",               "Brainstem"),
    ("substantia nigra",    "Brainstem"),
    ("tegmental",           "Brainstem"),
    ("locus",               "Brainstem"),
    ("red nucleus",         "Brainstem"),
    # ── Cerebral cortex (generic — must come last to avoid false matches) ─────
    ("neocortex",           "Cerebral_cortex"),
    ("cerebral hemisphere", "Cerebral_cortex"),
    ("frontal lobe",        "Cerebral_cortex"),
    ("parietal lobe",       "Cerebral_cortex"),
    ("temporal lobe",       "Cerebral_cortex"),
    ("occipital lobe",      "Cerebral_cortex"),
    ("gyrus",               "Cerebral_cortex"),
    ("lobule",              "Cerebral_cortex"),
    ("sulcus",              "Cerebral_cortex"),
    ("cortex",              "Cerebral_cortex"),
    ("insula",              "Cerebral_cortex"),
    ("cingulate",           "Cerebral_cortex"),
    ("operculum",           "Cerebral_cortex"),
]


def infer_lobe(labels_lower: list) -> str | None:
    text = " | ".join(labels_lower)
    for kw, lobe in LOBE_RULES:
        if kw in text:
            return lobe
    return None


def infer_division(labels_lower: list) -> str | None:
    text = " | ".join(labels_lower)
    for kw, div in DIVISION_RULES:
        if kw in text:
            return div
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Single-pass OWL streaming parser
# ──────────────────────────────────────────────────────────────────────────────
def stream_parse_fma(fma_path: Path):
    """
    One streaming pass over fma.owl.
    Returns:
        label_index : {lowercase_label: (canonical_label, iri)}
        graph       : {iri: {'label': str, 'parents': [iri, ...]}}
                      'parents' collects rdfs:subClassOf / part_of targets
    """
    iri_pat   = re.compile(r'<owl:Class rdf:about="([^"]+)"')
    label_pat = re.compile(r'<rdfs:label[^>]*>([^<]+)</rdfs:label>')
    res_pat   = re.compile(r'rdf:resource="([^"]+)"')

    label_index = {}   # lowercase → (canonical, iri)
    graph       = {}   # iri → {label, parents:[]}

    current_iri    = None
    in_subclass    = False
    lines_in_block = 0

    t0 = time.time()
    size_mb = fma_path.stat().st_size / 1e6
    print(f"Streaming {size_mb:.0f} MB OWL file... (this takes ~60-90 s)")

    with open(fma_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()

            # New class declaration
            m = iri_pat.search(stripped)
            if m:
                current_iri = m.group(1)
                graph.setdefault(current_iri, {'label': None, 'parents': []})
                in_subclass = False
                lines_in_block = 0
                continue

            if current_iri is None:
                continue

            lines_in_block += 1

            # Label
            if '<rdfs:label' in stripped:
                m = label_pat.search(stripped)
                if m:
                    lbl = m.group(1).strip()
                    if graph[current_iri]['label'] is None:
                        graph[current_iri]['label'] = lbl
                        label_index[lbl.lower()] = (lbl, current_iri)

            # Superclass or part_of restriction target
            if '<rdfs:subClassOf' in stripped:
                in_subclass = True
                # Inline: <rdfs:subClassOf rdf:resource="..."/>
                m = res_pat.search(stripped)
                if m:
                    parent_iri = m.group(1)
                    if parent_iri.startswith('http://purl.org/sig/ont/fma/'):
                        graph[current_iri]['parents'].append(parent_iri)
                    in_subclass = False

            elif in_subclass and 'rdf:resource' in stripped:
                m = res_pat.search(stripped)
                if m:
                    parent_iri = m.group(1)
                    if parent_iri.startswith('http://purl.org/sig/ont/fma/'):
                        graph[current_iri]['parents'].append(parent_iri)

            if '</rdfs:subClassOf>' in stripped or '/>' in stripped:
                in_subclass = False

    print(f"  Done in {time.time()-t0:.1f}s  |  classes={len(graph):,}  labels={len(label_index):,}")
    return label_index, graph


def get_hierarchy_labels(iri: str, graph: dict, max_depth: int = 6) -> list:
    """BFS up the parent chain, collecting all ancestor labels."""
    visited = set()
    queue   = [iri]
    labels  = []
    depth   = 0
    while queue and depth < max_depth:
        next_q = []
        for node in queue:
            if node in visited:
                continue
            visited.add(node)
            info = graph.get(node)
            if info and info['label']:
                labels.append(info['label'].lower())
            if info:
                next_q.extend(info['parents'])
        queue = next_q
        depth += 1
    return labels


# ──────────────────────────────────────────────────────────────────────────────
def main():
    label_index, graph = stream_parse_fma(FMA_PATH)

    # Parse AAL3 region list
    aal3_regions = []
    with open(AAL3_TXT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                aal3_regions.append({'label_id': int(parts[0]), 'name': parts[1]})

    records    = []
    not_found  = []
    ambiguous  = []

    print(f"\nBuilding AAL3 → FMA mapping for {len(aal3_regions)} regions...")

    for region in aal3_regions:
        lid  = region['label_id']
        name = region['name']

        if name.endswith('_L'):
            hemisphere, base = 'L', name[:-2]
        elif name.endswith('_R'):
            hemisphere, base = 'R', name[:-2]
        else:
            hemisphere, base = 'bilateral', name

        fma_label_query = AAL3_TO_FMA_LABEL.get(base)

        rec = {
            'label_id':       lid,
            'name':           name,
            'hemisphere':     hemisphere,
            'fma_id':         None,
            'fma_label':      None,
            'fma_superclass': None,
            'fma_lobe':       None,
            'fma_division':   None,
            'match_status':   'not_found',
        }

        if fma_label_query is None:
            not_found.append(name)
            records.append(rec)
            continue

        # Exact label lookup (case-insensitive)
        hit = label_index.get(fma_label_query.lower())
        if hit is None:
            # Fallback: partial match — skip white/gray matter and other non-structural entries
            q_low = fma_label_query.lower()
            skip = ("white matter", "gray matter", "artery", "vein", "nerve",
                    "fiber", "fasciculus", "capsule", "lumen", "tissue")
            partials = [
                (k, v) for k, v in label_index.items()
                if q_low in k and not any(s in k for s in skip)
            ]
            if partials:
                # Prefer shorter labels (more likely to be the primary structure)
                partials.sort(key=lambda x: len(x[0]))
                hit = partials[0][1]
                ambiguous.append(f"{name}: '{fma_label_query}' → '{hit[0]}'")
            else:
                not_found.append(f"{name} (label not found: '{fma_label_query}')")
                records.append(rec)
                continue

        canonical_label, iri = hit
        # Extract numeric FMA id from IRI
        fma_num_str = iri.split('/fma')[-1]
        try:
            fma_num = int(fma_num_str)
        except ValueError:
            fma_num = None

        # Walk hierarchy
        hierarchy_labels = [canonical_label.lower()] + get_hierarchy_labels(iri, graph)

        # Immediate named superclass (first non-self parent with a label)
        fma_superclass = None
        for p_iri in graph.get(iri, {}).get('parents', []):
            p_info = graph.get(p_iri)
            if p_info and p_info['label'] and p_info['label'] != canonical_label:
                fma_superclass = p_info['label']
                break

        fma_lobe     = infer_lobe(hierarchy_labels + [canonical_label.lower()])
        fma_division = infer_division(hierarchy_labels + [canonical_label.lower()])

        rec.update({
            'fma_id':         fma_num,
            'fma_label':      canonical_label,
            'fma_superclass': fma_superclass,
            'fma_lobe':       fma_lobe,
            'fma_division':   fma_division,
            'match_status':   'manual',
        })
        records.append(rec)

    # Save
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved: {OUTPUT_CSV}")

    mapped   = df[df['match_status'] != 'not_found']
    unmapped = df[df['match_status'] == 'not_found']

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total:    {len(df)}")
    print(f"  Mapped:   {len(mapped)}")
    print(f"  Missing:  {len(unmapped)}")

    if ambiguous:
        print(f"\n  Ambiguous (partial match used):")
        for a in ambiguous:
            print(f"    {a}")

    if len(unmapped) > 0:
        print(f"\n  Not found:")
        for n in unmapped['name'].tolist():
            print(f"    - {n}")

    print(f"\n  fma_lobe distribution:")
    print(mapped['fma_lobe'].value_counts(dropna=False).to_string())
    print(f"\n  fma_division distribution:")
    print(mapped['fma_division'].value_counts(dropna=False).to_string())

    print(f"\n  Sample rows:")
    cols = ['label_id','name','fma_id','fma_label','fma_superclass','fma_lobe','fma_division']
    print(df[cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
