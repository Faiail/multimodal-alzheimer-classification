"""
Exploration script for the FMA ontology.
Loads fma.owl and probes the structure to understand how brain regions
are organized before building the full AAL3→FMA mapping.

Run with:
    source .venv/bin/activate
    python3 src/data/explore_fma.py
"""
import sys
import time
from pathlib import Path
from owlready2 import get_ontology, default_world

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FMA_PATH = PROJECT_ROOT / "configs" / "FMA" / "fma.owl"

# ─── A few AAL3 region names to probe (base name without hemisphere suffix) ───
PROBE_REGIONS = [
    "Hippocampus",
    "Amygdala",
    "Precuneus",
    "Frontal",        # generic, to see what comes up
    "Putamen",
    "Thalamus",
    "Caudate",
    "Cerebellum",
    "Insula",
    "Cingulate",
    "Locus coeruleus",
    "Substantia nigra",
    "Raphe",
    "Superior frontal gyrus",
    "Inferior frontal gyrus",
    "Superior temporal gyrus",
]

def get_label(cls):
    """Return the preferred English label of a class, falling back to its name."""
    if hasattr(cls, 'label') and cls.label:
        return str(cls.label.first()) if cls.label.first() else cls.name
    return cls.name

def get_part_of_chain(cls, max_depth=6):
    """Walk part_of ancestors up to max_depth levels, returning a list of class names."""
    chain = []
    current = cls
    for _ in range(max_depth):
        # FMA uses both 'part_of' object property and superclasses
        parents = list(current.is_a)
        part_of_parents = []
        for p in parents:
            # Restriction: some_values_from / value on part_of property
            if hasattr(p, 'property') and 'part_of' in str(p.property):
                part_of_parents.append(p.value)
            elif hasattr(p, 'property') and 'regional_part_of' in str(p.property):
                part_of_parents.append(p.value)
        if not part_of_parents:
            break
        current = part_of_parents[0]
        chain.append(get_label(current))
    return chain


def main():
    print(f"Loading FMA from: {FMA_PATH}")
    print("This may take 1–3 minutes for the full ontology (~266 MB)...")
    t0 = time.time()
    fma = get_ontology(FMA_PATH.as_uri()).load()
    print(f"✅ Loaded in {time.time()-t0:.1f}s")

    all_classes = list(fma.classes())
    print(f"   Total classes in FMA: {len(all_classes):,}")

    # ─── Probe each region ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PROBING BRAIN REGIONS")
    print("="*70)

    for region_name in PROBE_REGIONS:
        query = region_name.lower()
        # Search by label (case-insensitive substring match)
        hits = [
            cls for cls in all_classes
            if query in get_label(cls).lower()
        ]
        # Prefer exact matches
        exact = [c for c in hits if get_label(c).lower() == query]
        candidates = exact if exact else hits[:5]

        print(f"\n--- Query: '{region_name}' → {len(hits)} hits ---")
        for c in candidates[:5]:
            iri_num = c.iri.split('/')[-1] if c.iri else 'N/A'
            chain = get_part_of_chain(c)
            chain_str = " → ".join(chain) if chain else "(no part_of chain)"
            superclasses = [get_label(p) for p in c.is_a
                            if hasattr(p, 'name') and p != c][:3]
            print(f"  [{iri_num}] {get_label(c)}")
            print(f"    is_a: {superclasses}")
            print(f"    part_of chain: {chain_str}")

    # ─── Show some known FMA IDs mentioned in the literature ─────────────────
    print("\n" + "="*70)
    print("DIRECT LOOKUP BY KNOWN FMA IDs")
    print("="*70)
    known_ids = {
        "FMA:275020": "Hippocampus",
        "FMA:61900":  "Precuneus",
        "FMA:61857":  "Superior frontal gyrus",
        "FMA:61860":  "Inferior frontal gyrus",
    }
    for fma_id, expected_name in known_ids.items():
        # IRI format: http://purl.org/sig/ont/fma/fma<NUM>
        num = fma_id.split(':')[1]
        iri = f"http://purl.org/sig/ont/fma/fma{num}"
        cls = default_world.search_one(iri=iri)
        if cls:
            print(f"  {fma_id}: found → '{get_label(cls)}' (expected: {expected_name})")
        else:
            # Try alternate IRI format
            iri2 = f"http://sig.biostr.washington.edu/fma#FMA_{num}"
            cls2 = default_world.search_one(iri=iri2)
            status = f"found → '{get_label(cls2)}'" if cls2 else "NOT FOUND"
            print(f"  {fma_id}: {status} (tried both IRI formats)")

    # ─── Print a sample class to understand annotation properties ─────────────
    print("\n" + "="*70)
    print("SAMPLE CLASS STRUCTURE (first 3 classes with 'gyrus' in name)")
    print("="*70)
    gyrus_classes = [c for c in all_classes if 'gyrus' in get_label(c).lower()][:3]
    for c in gyrus_classes:
        print(f"\n  Class: {c}")
        print(f"  IRI:   {c.iri}")
        print(f"  Label: {get_label(c)}")
        print(f"  is_a:  {list(c.is_a)[:4]}")
        if hasattr(c, 'comment') and c.comment:
            print(f"  comment: {c.comment.first()}")


if __name__ == "__main__":
    main()
