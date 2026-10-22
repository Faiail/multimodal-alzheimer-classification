"""
Round-2 label finder: broader queries for the still-missing regions.

Run with:
    python3 src/data/find_missing_labels.py
"""
from pathlib import Path
import re

LABELS_FILE = Path(__file__).resolve().parents[3] / 'configs' / 'FMA' / 'fma_labels_only.txt'

QUERIES = {
    # Cerebellar lobules with alternative names
    "crus":         ["crus", "ansiform", "semilunar"],
    "lobule_vi":    ["lobule vi", "simple lobule", "lobulus simplex"],
    "lobule_viib":  ["lobule viib", "lobule 7b", "gracile lobule", "biventr"],
    "lobule_viii":  ["lobule viii", "lobule 8 of", "biventral", "tonsil of cerebellum"],
    "lobule_x":     ["lobule x ", "lobule 10", "nodule of cerebellum", "flocculus"],
    # Thalamic nuclei alternative spellings
    "anteroventral":    ["anteroventral", "anterior ventral nucleus", "ventral anterior group",
                         "anterior nucleus of thalamus"],
    "lat_post_th":      ["lateral posterior", "posterolateral nucleus of thal", "lateral-posterior"],
    "reticular_th":     ["reticular nucleus", "thalamic reticular", "nucleus reticularis"],
    "mediodorsal_th":   ["mediodorsal", "dorsomedial nucleus of thal", "dorsal medial nucleus"],
    "lat_geniculate":   ["lateral geniculate", "geniculate body"],
    "med_geniculate":   ["medial geniculate", "medial geniculate body"],
    "ant_pulvinar":     ["anterior pulvinar", "pulvinar anterior", "oral pulvinar"],
}

SKIP_WORDS = ["white matter", "artery", "vein", "nerve", "fiber", "fasciculus", "bone", "muscle"]

label_extract = re.compile(r'>([^<]+)</rdfs:label>')

print(f"Scanning {LABELS_FILE.stat().st_size/1e6:.1f} MB labels file...")
results = {k: [] for k in QUERIES}

with open(LABELS_FILE, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = label_extract.search(line)
        if not m:
            continue
        lbl = m.group(1).strip()
        lbl_low = lbl.lower()
        if any(s in lbl_low for s in SKIP_WORDS):
            continue
        for key, kws in QUERIES.items():
            for kw in kws:
                if kw in lbl_low:
                    results[key].append(lbl)
                    break

print("\n=== RESULTS ===\n")
for key, hits in results.items():
    unique = list(dict.fromkeys(hits))[:8]
    print(f"[{key}]")
    for h in unique:
        print(f"  '{h}'")
    if not unique:
        print("  (NOT FOUND)")
    print()
