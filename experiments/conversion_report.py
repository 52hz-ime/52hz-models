"""What each variant of the conversion experiments achieved.

usage: python experiments/conversion_report.py RESULT.jsonl [MORE_RESULTS.jsonl ...] [--taxonomy VARIANT]
       (run from the repository root)

"exact" is the whole clause equal to the expected text, which the quality of the expected text also limits.
"round trip" only asks whether the reply reads exactly like the pinyin shown, a character with several readings
counting as any of them: it measures the conversion alone. "syllables wrong" is the edit distance between the
syllable sequences of the reply and of the expected text over the number of syllables of the expected text.
"""
import collections
import difflib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinyin import CJK  # noqa: E402
from pypinyin import Style, pinyin  # noqa: E402
from tests.benchmark_cases import character_errors  # noqa: E402
from verify import align, distance  # noqa: E402

BUCKETS = ((8, '4-8'), (14, '9-14'), (20, '15-20'), (10 ** 6, '21-30'))


def bucket(size):
    return next(name for limit, name in BUCKETS if size <= limit)


def wilson(k, n, z=1.96):
    p = k / n
    scale = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return middle - half, middle + half


def sign(a, b):
    k, n = min(a, b), a + b
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def edit(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def syllables_of(text):
    """The readings of its Chinese characters only: punctuation, digits and letters are not syllables."""
    import re
    return [reading[0] for run in re.findall(CJK.pattern + '+', text) for reading in pinyin(run, style=Style.NORMAL, errors='default')]


MEASURED = {}


def measure(row):
    """Everything the tables need about one reply, computed once."""
    key = (row['variant'], row['id'])
    if key not in MEASURED:
        text = row['text']
        want, got = syllables_of(row['gold']), syllables_of(text) if text else []
        MEASURED[key] = {'exact': bool(text) and text == row['gold'],
                         'trip': bool(text) and align(row['keys'], text)['cost'] == 0,
                         'syllables_wrong': edit(want, got), 'syllables': len(want),
                         'characters_wrong': character_errors(row['gold'], text) if text else len(row['gold']), 'characters': len(row['gold']),
                         'ms': row['ms'], 'want': want, 'got': got}
    return MEASURED[key]


def trip(row):
    return measure(row)['trip']


def summary(rows):
    measured = [measure(row) for row in rows]
    n = len(rows)
    exact = sum(m['exact'] for m in measured)
    rounded = sum(m['trip'] for m in measured)
    wrong = sum(m['syllables_wrong'] for m in measured)
    syllables = max(sum(m['syllables'] for m in measured), 1)
    chars = sum(m['characters_wrong'] for m in measured) / max(sum(m['characters'] for m in measured), 1)
    ms = sorted(m['ms'] for m in measured)
    return n, exact, rounded, wrong / syllables, chars, ms[len(ms) // 2]


def main():
    rest = sys.argv[1:]
    taxonomy = rest[rest.index('--taxonomy') + 1] if '--taxonomy' in rest else None
    args = [a for a in rest if not a.startswith('--') and a != taxonomy]
    rows = []
    for path in args:
        rows += [json.loads(line) for line in open(path, encoding='utf-8')]
    variants = list(dict.fromkeys(r['variant'] for r in rows))
    print(f'{"variant":30s} n     exact                 round trip             syllables wrong  characters wrong  median ms')
    for v in variants:
        n, exact, rounded, ser, cer, ms = summary([r for r in rows if r['variant'] == v])
        lo, hi = wilson(exact, n)
        print(f'{v:30s} {n:4d}  {exact:4d} ({exact / n:4.1%}, {lo:.0%}..{hi:.0%})  {rounded:4d} ({rounded / n:4.1%})        {ser:5.2%}           {cer:5.2%}          {ms:5d}')
    table = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        table[r['variant']][('length', bucket(len(r['gold'])))].append(r)
        table[r['variant']][('source', r['source'])].append(r)
        table[r['variant']][('context', 'with' if r['context'] else 'without')].append(r)
    for axis in ('length', 'source', 'context'):
        print(f'\nexact / round trip by {axis}:')
        keys = [k[1] for k in next(iter(table.values())) if k[0] == axis]
        print(f'   {"variant":30s} ' + '  '.join(f'{k:>12s}' for k in keys))
        for v in variants:
            cells = []
            for k in keys:
                group = table[v][(axis, k)]
                n, exact, rounded, *_ = summary(group)
                cells.append(f'{exact / n:4.0%} / {rounded / n:4.0%}')
            print(f'   {v:30s} ' + '  '.join(f'{c:>12s}' for c in cells))
    if len(variants) > 1:
        base = variants[0]
        print(f'\npaired against "{base}":')
        by = {v: {r['id']: r for r in rows if r['variant'] == v} for v in variants}
        for v in variants[1:]:
            fixed = sum(1 for i in by[v] if i in by[base] and by[v][i]['text'] == by[v][i]['gold'] != by[base][i]['text'])
            lost = sum(1 for i in by[v] if i in by[base] and by[base][i]['text'] == by[base][i]['gold'] != by[v][i]['text'])
            fixed_trip = sum(1 for i in by[v] if i in by[base] and trip(by[v][i]) and not trip(by[base][i]))
            lost_trip = sum(1 for i in by[v] if i in by[base] and trip(by[base][i]) and not trip(by[v][i]))
            print(f'   {v:30s} exact: fixed {fixed}, lost {lost} (p = {sign(fixed, lost):.4f});  round trip: fixed {fixed_trip}, lost {lost_trip} (p = {sign(fixed_trip, lost_trip):.4f})')
    if taxonomy:
        kinds = collections.Counter()
        for r in (r for r in rows if r['variant'] == taxonomy and not trip(r)):
            want, got = measure(r)['want'], measure(r)['got']
            for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, want, got, autojunk=False).get_opcodes():
                if tag == 'equal':
                    continue
                if tag == 'replace' and i2 - i1 == j2 - j1 == 1:
                    kinds['one syllable read another way, one keystroke off' if distance(want[i1], got[j1]) == 1 else 'one syllable read another way, further off'] += 1
                elif tag == 'replace':
                    kinds['several syllables read as several others'] += 1
                else:
                    kinds['syllable left out' if tag == 'delete' else 'syllable added'] += 1
        print(f'\nround trips that failed under "{taxonomy}", by what differs:')
        for k, v in kinds.most_common():
            print(f'   {k:52s} {v}')


if __name__ == '__main__':
    main()
