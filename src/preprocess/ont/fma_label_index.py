"""
Fast FMA index builder using xml.etree (SAX-style streaming parse).
Builds a label→IRI dictionary from fma.owl without owlready2,
then looks up specific brain region labels.

Run with:
    source .venv/bin/activate
    python3 src/data/fma_label_index.py
"""
import re
import time
from pathlib import Path

FMA_PATH = Path(__file__).resolve().parents[3] / 'configs' / 'FMA' / 'fma.owl'

# Labels we need to look up (exact FMA labels)
QUERY_LABELS = [
    "Precentral gyrus",
    "Middle frontal gyrus",
    "Superior frontal gyrus",
    "Inferior frontal gyrus",
    "Orbital gyrus",
    "Anterior orbital gyrus",
    "Posterior orbital gyrus",
    "Lateral orbital gyrus",
    "Medial orbital gyrus",
    "Inferior occipital gyrus",
    "Paracentral lobule",
    "Crus II of cerebellar hemisphere",
    "Crus I of cerebellar hemisphere",
    "Nucleus accumbens",
    "Reticular nucleus of thalamus",
    "Medial geniculate nucleus",
    "Inferior pulvinar nucleus",
    "Lateral pulvinar nucleus",
    "Anterior pulvinar nucleus",
    "Lateral posterior nucleus of thalamus",
    "Locus ceruleus",
    "Locus coeruleus",
    "Vermis",
    "Median raphe nucleus",
    "Dorsal raphe nucleus",
    "Ventral tegmental area",
    "Hippocampus",
    "Precuneus",
    "Amygdala",
    "Caudate nucleus",
    "Putamen",
    "Globus pallidus",
    "Thalamus",
    "Substantia nigra",
    "Red nucleus",
    "Intralaminar nucleus of thalamus",
    "Anteroventral nucleus of thalamus",
    "Ventral anterior nucleus of thalamus",
    "Ventral lateral nucleus of thalamus",
    "Ventroposterolateral nucleus of thalamus",
    "Ventral posterolateral nucleus of thalamus",
    "Mediodorsal nucleus of thalamus",
    "Lateral geniculate nucleus",
    "Medial pulvinar nucleus",
]

def build_index():
    """
    Stream-parse fma.owl as text to extract (label → IRI) pairs.
    Much faster than loading the full OWL graph.
    Returns dict: label (str) → IRI (str)
    """
    label_to_iri = {}
    current_iri = None
    iri_pat   = re.compile(r'<owl:Class rdf:about="([^"]+)"')
    label_pat = re.compile(r'<rdfs:label[^>]*>([^<]+)</rdfs:label>')

    t0 = time.time()
    print(f"Streaming {FMA_PATH.stat().st_size/1e6:.0f} MB OWL file...")

    pending_labels = []

    with open(FMA_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            m_iri = iri_pat.search(line)
            if m_iri:
                current_iri = m_iri.group(1)
                pending_labels = []

            if current_iri:
                m_lbl = label_pat.search(line)
                if m_lbl:
                    lbl = m_lbl.group(1).strip()
                    label_to_iri[lbl.lower()] = (lbl, current_iri)

    print(f"Index built: {len(label_to_iri):,} labels in {time.time()-t0:.1f}s")
    return label_to_iri


def main():
    idx = build_index()

    print()
    print("="*70)
    print("LABEL LOOKUPS")
    print("="*70)
    for query in QUERY_LABELS:
        result = idx.get(query.lower())
        if result:
            label, iri = result
            fma_num = iri.split("fma")[-1]
            print(f"  ✅ '{query}'")
            print(f"     IRI={iri}  (numeric ID: {fma_num})")
        else:
            # Try partial match
            partials = [(l, v) for l, v in idx.items() if query.lower() in l]
            if partials:
                print(f"  ⚠️  '{query}' not exact. Partials:")
                for l, (orig, iri) in partials[:3]:
                    print(f"       '{orig}'  {iri}")
            else:
                print(f"  ❌ '{query}' NOT FOUND")


if __name__ == "__main__":
    main()
