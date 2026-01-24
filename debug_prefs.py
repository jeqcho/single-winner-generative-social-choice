import json
from collections import Counter

prefs = json.load(open("outputs/sample_alt_voters/data/abortion/uniform/persona_no_context/rep0/preferences.json"))

bad = []
for v in range(100):
    r = [prefs[rank][v] for rank in range(100)]
    if len(r) != len(set(r)):
        c = Counter(r)
        bad.append((v, {k:n for k,n in c.items() if n>1}))

print(f"Voters with duplicates: {len(bad)}")
for v, d in bad:
    print(f"  Voter {v}: {d}")
