"""Write the n configurations obtained by deleting one point of a set (for 'remove one point, re-insert by grid').
Usage: python drop_one.py FILE.json|FILE.jsonl:i OUT.jsonl"""
import sys, json
src = sys.argv[1]
if '.jsonl:' in src:
    f, i = src.rsplit(':', 1); x = json.loads(open(f).read().splitlines()[int(i)])['x']
else:
    x = json.load(open(src))['x']
P = [x[2 * k: 2 * k + 2] for k in range(len(x) // 2)]
with open(sys.argv[2], 'w') as out:
    for k in range(len(P)):
        Q = [c for j, p in enumerate(P) if j != k for c in p]; out.write(json.dumps(dict(value=None, x=Q, dropped=k)) + '\n')
print(len(P), 'sets written')
