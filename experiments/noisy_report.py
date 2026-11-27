"""What a decoder achieved on noisy keys.

usage: python experiments/noisy_report.py CASES.jsonl RESULT.jsonl [MORE_RESULTS.jsonl ...]
       (run from the repository root; no model needed)

On noisy keys the expected text does not read back exactly: it carries the errors the typist made. So besides
exact match, every reply is judged by the verifier against the keys and against the expected text:

  reads as well as the truth   the reply's cost under the verifier is not higher than the expected text's
  within the budget            the reply's cost is within the verifier's budget, so the decoder would commit it
  cost more than the truth     by how much the reply reads worse than the expected text
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.benchmark_cases import character_errors, errors_in, label  # noqa: E402
from verify import align  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n
    scale = 1 + z * z / n
    middle = (p + z * z / (2 * n)) / scale
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / scale
    return middle - half, middle + half


def sign(a, b):
    k, n = min(a, b), a + b
    return 1.0 if not n else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def judged(case, reply):
    text = reply.get('text') or ''
    report = align(case['keys'], text) if text else None
    gold = align(case['keys'], case['text'])
    return {'exact': text == case['text'], 'cost': report['cost'] if report else 10 ** 6, 'budget': report['budget'] if report else 0,
            'gold': gold['cost'], 'as_well': bool(report) and report['cost'] <= gold['cost'], 'within': bool(report) and report['consistent'],
            'chars': character_errors(case['text'], text) if text else len(case['text']), 'length': len(case['text'])}


def line(name, rows):
    n = len(rows)
    verdicts = [judged(case, reply) for case, reply in rows]
    exact = sum(v['exact'] for v in verdicts)
    low, high = wilson(exact, n)
    ms = sorted(reply.get('ms', 0) for _, reply in rows)
    q = lambda p: ms[min(n - 1, round(p * (n - 1)))] if ms else 0
    print(f'{name:52s} exact {exact:3d} ({exact / n:5.1%}, {low:.0%}..{high:.0%})  reads as well as the truth {sum(v["as_well"] for v in verdicts) / n:5.1%}  '
          f'within budget {sum(v["within"] for v in verdicts) / n:5.1%}  worse than truth by 1 {sum(v["cost"] - v["gold"] == 1 for v in verdicts) / n:4.0%}  '
          f'2+ {sum(v["cost"] - v["gold"] >= 2 for v in verdicts) / n:4.0%}  chars wrong {sum(v["chars"] for v in verdicts) / max(sum(v["length"] for v in verdicts), 1):5.1%}  '
          f'median {q(.5) / 1000:5.2f} s  p90 {q(.9) / 1000:5.2f} s  max {ms[-1] / 1000 if ms else 0:6.1f} s')


def main():
    cases_path, results = sys.argv[1], sys.argv[2:]
    cases = {row['id']: row for row in (json.loads(line) for line in open(cases_path, encoding='utf-8'))}
    rows = []
    for path in results:
        rows += [json.loads(line) for line in open(path, encoding='utf-8')]
    variants = list(dict.fromkeys(reply['variant'] for reply in rows))
    for variant in variants:
        selected = [(cases[r['id']], r) for r in rows if r['variant'] == variant and r['id'] in cases]
        line(variant, selected)
    names = ('clean', 'drop', 'duplicate', 'insert', 'neighbor', 'swap', 'hand_swap', 'truncate_suffix', 'nasal', 'n_l', 'retroflex', 'f_h', 'x2 and more')
    print('\nexact match by noise in the input:')
    print(f'   {"variant":52s} ' + ' '.join(f'{name[:7]:>8s}' for name in names))
    for variant in variants:
        selected = [(cases[r['id']], r) for r in rows if r['variant'] == variant and r['id'] in cases]
        cells = []
        for name in names:
            group = [(case, reply) for case, reply in selected if (label(case) if errors_in(case) == 1 else 'clean' if not errors_in(case) else 'x2 and more') == name]
            cells.append(f'{sum(judged(case, reply)["exact"] for case, reply in group) / len(group):8.0%}' if group else f'{"-":>8s}')
        print(f'   {variant[:52]:52s} ' + ' '.join(cells))
    if len(variants) == 2:
        first = {r['id']: r for r in rows if r['variant'] == variants[0]}
        second = {r['id']: r for r in rows if r['variant'] == variants[1]}
        common = [i for i in first if i in second and i in cases]
        better = sum(1 for i in common if judged(cases[i], second[i])['exact'] and not judged(cases[i], first[i])['exact'])
        worse = sum(1 for i in common if judged(cases[i], first[i])['exact'] and not judged(cases[i], second[i])['exact'])
        fixed = sum(1 for i in common if judged(cases[i], second[i])['as_well'] and not judged(cases[i], first[i])['as_well'])
        lost = sum(1 for i in common if judged(cases[i], first[i])['as_well'] and not judged(cases[i], second[i])['as_well'])
        print(f'\npaired, {variants[1]} against {variants[0]}: exact fixed {better}, lost {worse} (p = {sign(better, worse):.4f}); '
              f'reads as well as the truth: fixed {fixed}, lost {lost} (p = {sign(fixed, lost):.4f})')


if __name__ == '__main__':
    main()
