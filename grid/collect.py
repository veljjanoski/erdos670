"""Merge gp_{tag}_*.jsonl files and print the best distinct values.  Usage: python collect.py TAGPATTERN [top]"""
import sys, glob, json
rows = []
for f in glob.glob(f'gp_{sys.argv[1]}_*.jsonl'):
    for l in open(f):
        if l.strip(): rows.append(json.loads(l))
rows.sort(key=lambda r: r['value']); top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
seen = []
for r in rows:
    if all(abs(r['value'] - s) > 1e-4 for s in seen): seen.append(r['value'])
print(f"{len(rows)} results, {len(seen)} distinct values; best:", ' '.join(f'{v:.4f}' for v in seen[:top]))
