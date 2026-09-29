"""Collect distinct good results from gp_{pattern}_*.jsonl into one base file.  Usage: python select_bases.py PATTERN K OUT"""
import sys, glob, json
rows = [json.loads(l) for f in glob.glob(f'gp_{sys.argv[1]}_*.jsonl') for l in open(f) if l.strip()]
rows.sort(key=lambda r: r['value']); out = []
for r in rows:
    if all(abs(r['value'] - s['value']) > 1e-3 for s in out): out.append(r)
    if len(out) >= int(sys.argv[2]): break
open(sys.argv[3], 'w').write(''.join(json.dumps(r) + '\n' for r in out))
print(len(out), 'bases:', ' '.join(f"{r['value']:.3f}" for r in out))
