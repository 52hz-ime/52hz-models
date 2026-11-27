"""Pinyin to characters with the choice at every position limited to the characters that read that way.

usage: python experiments/constrained_decoding.py CASES.jsonl OUT.jsonl VARIANT [--probe]
       (run from the repository root, with the model server up)

The pinyin is shown to the model exactly as in `pinyin_conversion.py` (same variants), and in addition the
request carries a grammar: one rule per syllable, admitting only the characters that have that reading, so
that whatever the model writes reads back as the pinyin it was given. `--probe` only checks that the server
accepts a grammar on its chat endpoint, with a two-word grammar, and prints what it answered.
`--chars-from LEXICON.jsonl` limits the candidates to the single characters the word list has counted, which
leaves out rare and traditional characters and makes the grammar much smaller.
"""
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pinyin_conversion import ENDPOINT, prompt, readings  # noqa: E402
from pinyin import CJK  # noqa: E402
from pypinyin.contrib.tone_convert import to_normal  # noqa: E402
from pypinyin.pinyin_dict import pinyin_dict  # noqa: E402

PROBE = 'root ::= "你好" | "再见"'


def readers(known=None):
    """For every toneless reading, the characters that have it (the dictionary lists readings with tones)."""
    table = {}
    for code, tones in pinyin_dict.items():
        char = chr(code)
        if known is not None and char not in known:
            continue
        for tone in tones.split(','):
            table.setdefault(to_normal(tone), []).append(char)
    return {reading: ''.join(chars) for reading, chars in table.items()}


def counted(path):
    """The single characters the word list has counted."""
    return {json.loads(line)['word'] for line in open(path, encoding='utf-8') if len(json.loads(line)['word']) == 1}


READERS = readers()


def grammar(text):
    """One rule per character of the expected text: the characters that read the same, or the literal itself."""
    rules, names = [], []
    for index, (char, reading) in enumerate(zip(text, readings(text, 'spaced'))):
        if CJK.match(char) and reading in READERS:
            name = f'r{index}'
            rules.append(f'{name} ::= ' + ' | '.join(f'"{candidate}"' for candidate in READERS[reading]))
        else:
            name = '"' + char.replace('\\', '\\\\').replace('"', '\\"') + '"'
        names.append(name)
    return 'root ::= ' + ' '.join(names) + '\n' + '\n'.join(rules)


def post(payload):
    request = Request(ENDPOINT + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def main():
    global READERS
    rest = sys.argv[1:]
    if '--chars-from' in rest:
        at = rest.index('--chars-from')
        READERS = readers(counted(rest[at + 1]))
        del rest[at:at + 2]
    args = [a for a in rest if not a.startswith('--')]
    if '--probe' in sys.argv:
        reply = post({'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 16, 'grammar': PROBE,
                      'messages': [{'role': 'user', 'content': '随便打个招呼'}]})
        print(json.dumps(reply['choices'][0]['message'], ensure_ascii=False), '| grammar characters:',
              len(grammar('你好吗')), '| readings known:', len(READERS))
        return
    cases_path, out, variant = Path(args[0]), Path(args[1]), args[2]
    rows = [json.loads(line) for line in open(cases_path, encoding='utf-8')]
    with out.open('a', encoding='utf-8') as handle:
        for row in rows:
            fields = {key: row[key] for key in ('keys', 'context') if key in row}
            started = time.perf_counter()
            reply = post({'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256, 'grammar': grammar(row['text']),
                          'messages': prompt(variant, fields)})
            text = (reply['choices'][0]['message']['content'] or '').strip()
            handle.write(json.dumps({'variant': variant + ' +grammar', 'id': row['id'], 'text': text, 'gold': row['text'], 'keys': row['keys'],
                                     'source': row['source_id'], 'context': bool(row.get('context')), 'ms': round((time.perf_counter() - started) * 1000),
                                     'usage': reply.get('usage')}, ensure_ascii=False) + '\n')
            handle.flush()


if __name__ == '__main__':
    main()
