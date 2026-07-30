"""
Debug script: finds the real IRI format for our target FMA IDs by streaming fma.owl.
Also extracts the label and first few is_a/part_of lines around each hit.

Run with:
    source .venv/bin/activate
    python3 src/data/debug_fma_iri.py
"""
import re
import time
from pathlib import Path

FMA_PATH = Path(__file__).resolve().parents[2] / 'configs' / 'FMA' / 'fma.owl'

# The FMA numeric IDs from our AAL3_TO_FMA mapping
TARGET_IDS = [
    61851, 61857, 61858, 61853, 224850, 61868, 61869, 61870, 61871, 61872,
    67329, 62434, 275020, 61918, 61841,
    61901, 61899, 61897, 61891, 61893, 61895, 61896,
    61855, 61885, 61887, 61889, 61890, 61900, 61856,
    61905, 61908, 61906, 61907,
    61833, 61834, 61835, 61838,
    62007, 62026, 62034, 62028, 62030, 62032, 62038, 62040, 62035, 62045, 62044, 62042, 62041,
    72249, 72250, 72244, 72245, 72246, 72248, 72251, 72252, 72253,
    67945, 67947, 67946, 67965, 68462, 68461,
    61863, 61862, 61864, 61860,
]

def make_iri_variants(fma_id: int) -> list:
    """Generate all plausible IRI variants for a given FMA numeric ID."""
    base = "http://purl.org/sig/ont/fma/fma"
    return [
        f"{base}{fma_id}",           # no padding:  fma61857
        f"{base}{fma_id:07d}",       # 7-digit pad: fma0061857
        f"{base}{fma_id:08d}",       # 8-digit pad
        f"http://sig.biostr.washington.edu/fma#FMA_{fma_id}",  # old format
    ]

def main():
    target_set = set(TARGET_IDS)
    
    # Build regex patterns: for each target, try all zero-paddings
    # We look for any occurrence of the numeric suffix in IRI lines
    id_patterns = {}
    for tid in TARGET_IDS:
        variants = [
            str(tid),
            f"{tid:07d}",
            f"{tid:08d}",
        ]
        id_patterns[tid] = variants

    print(f"Streaming {FMA_PATH.stat().st_size / 1e6:.0f} MB OWL file...")
    print(f"Looking for {len(TARGET_IDS)} target FMA IDs...\n")

    # Results: fma_id -> {'iri': ..., 'label': ..., 'context': [...]}
    results = {}
    
    iri_line_pat = re.compile(r'owl:Class rdf:about="([^"]+)"')
    label_pat    = re.compile(r'<rdfs:label[^>]*>([^<]+)</rdfs:label>')
    
    current_iri = None
    current_hit_id = None
    context_lines = []
    collecting = False
    collect_count = 0

    t0 = time.time()
    with open(FMA_PATH, encoding="utf-8", errors="replace") as f:
        for line in f:
            # Check if this is a new class declaration
            m_iri = iri_line_pat.search(line)
            if m_iri:
                iri = m_iri.group(1)
                current_iri = iri
                current_hit_id = None
                
                # Check if this IRI contains any of our target IDs
                for tid, variants in id_patterns.items():
                    if tid in results:
                        continue  # already found
                    for v in variants:
                        if v in iri:
                            current_hit_id = tid
                            if tid not in results:
                                results[tid] = {'iri': iri, 'label': None, 'is_a': [], 'part_of': []}
                            collecting = True
                            collect_count = 0
                            break
                    if current_hit_id:
                        break
                else:
                    collecting = False
            
            if collecting and current_hit_id is not None:
                collect_count += 1
                stripped = line.strip()
                
                # Extract label
                m_lbl = label_pat.search(line)
                if m_lbl and results[current_hit_id]['label'] is None:
                    results[current_hit_id]['label'] = m_lbl.group(1).strip()
                
                # Collect is_a / part_of references (first 8 lines after IRI decl)
                if collect_count < 25:
                    if 'rdf:resource' in stripped or 'rdfs:subClassOf' in stripped:
                        results[current_hit_id]['is_a'].append(stripped[:120])
                
                # Stop collecting after 30 lines (one class block)
                if collect_count > 30:
                    collecting = False

    elapsed = time.time() - t0
    print(f"Scan completed in {elapsed:.1f}s\n")
    print(f"Found {len(results)}/{len(TARGET_IDS)} target IDs\n")
    print("=" * 80)

    found_ids = sorted(results.keys())
    missing_ids = sorted(set(TARGET_IDS) - set(results.keys()))

    for tid in found_ids:
        r = results[tid]
        print(f"FMA:{tid:>7} | IRI: {r['iri']}")
        print(f"            Label: {r['label']}")
        if r['is_a']:
            print(f"            Context: {r['is_a'][0]}")
        print()

    print("=" * 80)
    print(f"\nMISSING ({len(missing_ids)}):")
    for mid in missing_ids:
        print(f"  FMA:{mid}")

    # Also: show what IRI format the found ones use
    print("\n" + "=" * 80)
    print("IRI FORMAT ANALYSIS:")
    formats = set()
    for tid, r in results.items():
        iri = r['iri']
        # Extract the suffix after the last 'fma'
        suffix = iri.split('fma')[-1]
        formats.add(f"fma{suffix} (len={len(suffix)})")
    for f in sorted(formats):
        print(f"  {f}")

if __name__ == "__main__":
    main()
