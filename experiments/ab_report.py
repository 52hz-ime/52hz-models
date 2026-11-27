"""Exact match and round trip of each prompt variant in the result files of the prompt experiments.

"Round trip" means the reply reads exactly like the keys, a character with several readings counting as any of them
(it is what `verify.align` charges nothing for).

usage: python experiments/ab_report.py RESULT.jsonl [MORE_RESULTS.jsonl ...]   (run from the repository root)
"""
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify import align  # noqa: E402


def sign(a, b):
    k, n = min(a, b), a + b
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


rows = []
for path in sys.argv[1:]:
    rows += [json.loads(line) for line in open(path, encoding='utf-8')]
variants = list(dict.fromkeys(r['variant'] for r in rows))
n = {v: sum(1 for r in rows if r['variant'] == v) for v in variants}
exact = {v: {r['id']: r['text'] == r['gold'] for r in rows if r['variant'] == v} for v in variants}
print(f'{"prompt":52s} exact        round trip    either')
for v in variants:
    rs = [r for r in rows if r['variant'] == v]
    e = sum(exact[v].values())
    trip = sum(align(r['keys'], r['text'])['cost'] == 0 for r in rs if r['text'])
    either = sum(r['text'] == r['gold'] or align(r['keys'], r['text'])['cost'] == 0 for r in rs if r['text'])
    print(f'{v:52s} {e:3d} ({e / n[v]:4.0%})   {trip:3d} ({trip / n[v]:4.0%})    {either:3d} ({either / n[v]:4.0%})')
print('\nexact by length of the expected text:')
for v in variants:
    rs = [r for r in rows if r['variant'] == v]
    print(f'   {v:50s} ' + '  '.join(f'{name} {sum(r["text"] == r["gold"] for r in rs if lo <= len(r["gold"]) <= hi)}/{sum(1 for r in rs if lo <= len(r["gold"]) <= hi)}'
                                   for name, lo, hi in (('4-8', 4, 8), ('9-14', 9, 14), ('15-20', 15, 20), ('21-30', 21, 30))))
print('\nexact by source:')
for v in variants:
    rs = [r for r in rows if r['variant'] == v]
    c = collections.Counter((r['source'], r['text'] == r['gold']) for r in rs)
    print(f'   {v:50s} ' + '  '.join(f'{s[:9]} {c[(s, True)]}/{c[(s, True)] + c[(s, False)]}' for s in sorted({r['source'] for r in rs})))
if len(variants) > 1:
    print('\npaired against the first variant listed:')
    first = variants[0]
    for v in variants[1:]:
        better = sum(1 for i in exact[v] if i in exact[first] and exact[v][i] and not exact[first][i])
        worse = sum(1 for i in exact[v] if i in exact[first] and exact[first][i] and not exact[v][i])
        print(f'   {v:50s} fixed {better:3d}, lost {worse:3d}, sign test p = {sign(better, worse):.4f}')
