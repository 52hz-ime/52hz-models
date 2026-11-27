"""On noisy keys, does a grammar built from the segmenter's line still allow the expected text?

usage: python experiments/grammar_ceiling.py CASES.jsonl [MORE_CASES.jsonl ...]      (run from the repository root; no model needed)

Constrained decoding can only write what the grammar admits. The grammar has one rule per token of the
segmenter's line, and each rule admits the characters that read the token. What a rule admits decides how
often the text the user meant can be written at all:

  strict        the token as read, and the syllables an accent merges it with (en/eng, n/l, zh/z, ch/c, sh/s, f/h)
  at "?"        strict, and one keystroke off where the segmenter tolerated an error
  one keystroke  strict, and one keystroke off everywhere

A final token that is not a whole syllable (the last letters were never typed) admits every syllable that
starts with it. A line with another number of tokens than the expected text has characters cannot be
written under any of the three.

A grammar can also be built on the keys instead of on the segmenter's one line: at every position of the keys
it admits every syllable that is written there, so the model chooses the boundaries itself ("lattice"). With
"one keystroke" a syllable may differ from the letters by one keystroke, which covers a dropped or neighbouring
letter and a transposition; the last syllable may be cut short. Whether the expected text can be written is
then whether its readings can walk from the start of the keys to the end.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pinyin_conversion import readings  # noqa: E402
from syllables import SYLLABLES, accent_partners, segment  # noqa: E402
from tests.benchmark_cases import errors_in, label  # noqa: E402
from verify import distance  # noqa: E402

POLICIES = ('strict', 'at "?"', 'one keystroke')
NEAR = {syllable: {other for other in SYLLABLES if distance(other, syllable) <= 1} for syllable in SYLLABLES}


def admitted(token, policy):
    """The readings a token's rule admits."""
    unsure, typed = token.endswith('?'), token.rstrip('?')
    if typed in SYLLABLES:
        allowed = {typed, *accent_partners(typed)}
        if policy == 'one keystroke' or (policy == 'at "?"' and unsure):
            allowed |= NEAR[typed]
        return allowed
    return {syllable for syllable in SYLLABLES if syllable.startswith(typed)} if typed.isalpha() else {typed}


def expressible(keys, text, policy):
    tokens, units = segment(keys).split(' '), readings(text, 'spaced')
    return len(tokens) == len(units) and all(unit in admitted(token, policy) for token, unit in zip(tokens, units))


def walkable(keys, units, near):
    """Whether the readings of the text can cover the keys from the first letter to the last."""
    reach = {0}
    for index, unit in enumerate(units):
        following, last = set(), index == len(units) - 1
        for start in reach:
            for size in range(1, len(keys) - start + 1):
                piece = keys[start:start + size]
                if last and unit in SYLLABLES and unit.startswith(piece) and len(piece) < len(unit):
                    following.add(start + size)                       # the last syllable was never finished
                elif unit in SYLLABLES and (unit == piece or (near and distance(unit, piece) <= 1)):
                    following.add(start + size)
                elif unit not in SYLLABLES and unit == piece:
                    following.add(start + size)
        if not following:
            return False
        reach = following
    return len(keys) in reach


def main():
    rows = []
    for path in sys.argv[1:]:
        rows += [json.loads(line) for line in open(path, encoding='utf-8')]
    table = collections.defaultdict(collections.Counter)
    for row in rows:
        name = label(row) if errors_in(row) == 1 else 'clean' if not errors_in(row) else 'x2 and more'
        for policy in POLICIES:
            table[name][policy] += expressible(row['keys'], row['text'], policy)
    order = ['clean', 'drop', 'duplicate', 'insert', 'neighbor', 'swap', 'hand_swap', 'truncate_suffix', 'nasal', 'n_l', 'retroflex', 'f_h', 'x2 and more']
    groups = [name for name in order if name in table] + sorted(name for name in table if name not in order)
    print(f'the expected text can be written under the grammar, share of cases ({len(rows)} cases)')
    print(f'   {"noise in the input":18s} n      ' + '  '.join(f'{policy:>13s}' for policy in POLICIES))
    for name in groups:
        counts = collections.Counter(label(row) if errors_in(row) == 1 else 'clean' if not errors_in(row) else 'x2 and more' for row in rows)
        n = counts[name]
        print(f'   {name:18s} {n:5d}  ' + '  '.join(f'{table[name][policy] / n:13.1%}' for policy in POLICIES))
    n = len(rows)
    print(f'   {"all":18s} {n:5d}  ' + '  '.join(f'{sum(table[name][policy] for name in groups) / n:13.1%}' for policy in POLICIES))
    lattice = collections.defaultdict(collections.Counter)
    for row in rows:
        name = label(row) if errors_in(row) == 1 else 'clean' if not errors_in(row) else 'x2 and more'
        units = readings(row['text'], 'spaced')
        lattice[name]['lattice, exact'] += walkable(row['keys'], units, near=False)
        lattice[name]['lattice, one keystroke'] += walkable(row['keys'], units, near=True)
    policies = ('lattice, exact', 'lattice, one keystroke')
    print(f'\nthe same for a grammar built on the keys (every syllable that can be written at each position)')
    print(f'   {"noise in the input":18s} n      ' + '  '.join(f'{policy:>22s}' for policy in policies))
    for name in groups:
        counts = collections.Counter(label(row) if errors_in(row) == 1 else 'clean' if not errors_in(row) else 'x2 and more' for row in rows)
        print(f'   {name:18s} {counts[name]:5d}  ' + '  '.join(f'{lattice[name][policy] / counts[name]:22.1%}' for policy in policies))
    print(f'   {"all":18s} {n:5d}  ' + '  '.join(f'{sum(lattice[name][policy] for name in groups) / n:22.1%}' for policy in policies))


if __name__ == '__main__':
    main()
