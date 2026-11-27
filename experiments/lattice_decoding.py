"""Constrained decoding with the grammar built on the keys themselves, so that nothing has to be segmented first.

usage: python experiments/lattice_decoding.py CASES.jsonl OUT.jsonl VARIANT [--one-keystroke] [--chars-from LEXICON.jsonl] [--characters N]
       (run from the repository root, with the model server up)

The grammar is a lattice over the keys: one rule per position, and from each position the syllables that can be
written there, each leading on to the position after its letters. Where one syllable ends and the next begins
is left to the model. With --one-keystroke a syllable may also differ from the letters by one keystroke, which
is how a dropped or neighbouring letter and a transposition get in. The last syllable may be left unfinished.
A word with a capital letter or a digit in it (iPhone15, HR, 2002) is passed on as typed. When no walk from the
first letter to the last exists, the request is sent without a grammar and the reply is marked so.
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.constrained_decoding import READERS, frequent  # noqa: E402
from experiments.pinyin_conversion import ENDPOINT, prompt  # noqa: E402
from syllables import LONGEST, SYLLABLES  # noqa: E402
from verify import distance  # noqa: E402

# What is passed on as typed rather than converted: a word led by a capital letter, with at most two lowercase
# letters in front of it (iPhone15, HR, WeChat); or a run of lowercase letters directly followed by digits and
# ending there (yiyi1111116, ss2). A run of pinyin letters never qualifies, so the keys cannot be echoed back.
WORD = re.compile(r'[a-z]{0,2}[A-Z][A-Za-z0-9]*|[a-z]+[0-9]+')
NEAR = {}
KEEP = None            # with --characters: how many of the most frequent characters of a reading are admitted


def rule(syllable):
    """The name of a reading's rule. llama.cpp takes letters, digits and dashes in rule names, not underscores."""
    return 'r' + syllable


def literal(char):
    return '"' + char.replace('\\', '\\\\').replace('"', '\\"') + '"'


def near(keys):
    """For every stretch of lowercase letters in the keys, the syllables within one keystroke of it."""
    table = {}
    for start, char in enumerate(keys):
        if not 'a' <= char <= 'z':
            continue
        for size in range(1, min(LONGEST, len(keys) - start) + 1):
            typed = keys[start:start + size]
            if not typed.islower():
                break
            table[typed] = {syllable for syllable in SYLLABLES if distance(syllable, typed) <= 1}
    return table


def steps(keys, start, one_keystroke):
    """(what is written, how many letters it takes) for everything that may be written at one position."""
    found, rest, char = [], len(keys) - start, keys[start]
    if 'a' <= char <= 'z':
        for size in range(1, min(LONGEST, rest) + 1):
            typed = keys[start:start + size]
            if typed in SYLLABLES:
                found.append((typed, size))
            if one_keystroke:
                found += [(syllable, size) for syllable in NEAR.get(typed, ()) if syllable != typed]
    if '0' <= char <= '9':
        end = start
        while end < len(keys) and '0' <= keys[end] <= '9':
            end += 1
        found.append((keys[start:end], end - start))               # 9, 2002
    if char.isalnum():
        end = start
        while end < len(keys) and keys[end].isalnum():
            end += 1
        if WORD.fullmatch(keys[start:end]):                         # the run as a whole, or nothing
            found.append((keys[start:end], end - start))
    else:
        found.append((char, 1))                                     # punctuation and the like, as typed
    return found


def lattice(keys, one_keystroke=False):
    """The grammar for the keys, or None when nothing can walk from the first letter to the last."""
    NEAR.clear()
    if one_keystroke:
        NEAR.update(near(keys))
    length = len(keys)
    moves = {start: [(item, size) for item, size in steps(keys, start, one_keystroke) if item not in SYLLABLES or item in READERS]
             for start in range(length)}          # a reading with no character to write it is no step at all
    alive = {length}
    for start in range(length - 1, -1, -1):
        if any(start + size in alive for _, size in moves[start]):
            alive.add(start)
    if 0 not in alive:
        return None
    syllables = sorted({item for start in alive if start < length for item, size in moves[start] if item in SYLLABLES})
    lines = ['root ::= n0']
    for start in sorted(alive):
        if start == length:
            continue
        alternatives = []
        for item, size in moves[start]:
            if start + size not in alive:
                continue
            alternatives.append((rule(item) if item in SYLLABLES else literal(item)) + (f' n{start + size}' if start + size < length else ''))
        lines.append(f'n{start} ::= ' + ' | '.join(dict.fromkeys(alternatives)))
    lines += [f'{rule(syllable)} ::= ' + ' | '.join(literal(char) for char in READERS[syllable][:KEEP]) for syllable in syllables]
    return '\n'.join(lines) + '\n'


def post(payload):
    request = Request(ENDPOINT + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    global READERS, KEEP
    rest = sys.argv[1:]
    one_keystroke = '--one-keystroke' in rest
    rest = [argument for argument in rest if argument != '--one-keystroke']
    word_list = rest[rest.index('--chars-from') + 1] if '--chars-from' in rest else None
    rest = [argument for argument in rest if argument != '--chars-from']
    if '--characters' in rest:
        at = rest.index('--characters')
        KEEP = int(rest[at + 1])
        del rest[at:at + 2]
    if word_list:
        READERS = frequent(word_list, KEEP)
    elif KEEP:
        sys.exit('--characters needs --chars-from, the source of the counts')
    cases_path, out, variant = Path(rest[0]), Path(rest[1]), rest[2]
    name = variant + ' + lattice grammar' + (', one keystroke' if one_keystroke else '')
    rows = [json.loads(line) for line in open(cases_path, encoding='utf-8')]
    with out.open('a', encoding='utf-8') as handle:
        for row in rows:
            started = time.perf_counter()
            grammar, failure = lattice(row['keys'], one_keystroke), ''
            payload = {'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256, 'messages': prompt(variant, row)}
            try:
                text = (post(dict(payload, grammar=grammar) if grammar else payload)['choices'][0]['message']['content'] or '').strip()
            except (OSError, HTTPError, KeyError, IndexError) as error:      # a slow or refused grammar must not end the run
                text, failure = '', f'{type(error).__name__}'
            handle.write(json.dumps({'variant': name, 'id': row['id'], 'text': text, 'gold': row['text'], 'keys': row['keys'],
                                     'source': row['source_id'], 'context': bool(row.get('context')), 'ms': round((time.perf_counter() - started) * 1000),
                                     'grammar': failure or ('lattice' if grammar else 'none'), 'grammar_characters': len(grammar) if grammar else 0},
                                    ensure_ascii=False) + '\n')
            handle.flush()


if __name__ == '__main__':
    main()
