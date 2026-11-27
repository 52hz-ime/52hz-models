"""Ask again, pointing at whatever does not read like the pinyin, until it all does or a round stops helping.

usage: python experiments/reask.py CASES.jsonl FIRST_REPLIES.jsonl VARIANT OUT.jsonl [ROUNDS=3]
       (run from the repository root, with the model server up)

The first replies come from a finished run of `pinyin_conversion.py`, so only the asking again is measured.
The prompt and the acceptance rule are the decoder's own (`infer.REPAIR_SYSTEM`, and a rewrite is kept only
when it reads more like the pinyin, is made of words, and turns no character into letters). What differs from
the decoder's loop is when it runs: for any syllable that does not read like the pinyin, one keystroke off or
one letter left over included. Nothing here needs a grammar.
"""
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pinyin_conversion import ENDPOINT, syllables  # noqa: E402
from infer import PAIRS, REPAIR_SYSTEM, characters, hint_context  # noqa: E402
from syllables import alternatives, segment  # noqa: E402
from verify import align, summary  # noqa: E402

NAMED = 10


def spots(report):
    """Every character that does not read like the pinyin and every unclaimed letter, however small."""
    lines = []
    for index in report['doubts'] + [index for index in report['flagged'] if index not in report['doubts']]:
        unit = report['units'][index]
        line = f"第{index + 1}个字「{unit['char']}」读作 {unit['reading']}，拼音这一处是 {unit['typed'] or '空'}"
        others = alternatives(unit['typed'], (unit['reading'], unit['typed']))
        lines.append(line + (f"，也可以读作 {' / '.join(others)}" if others else ""))
    for gap in report['gaps']:
        before, letters = gap['before'], gap['letters']
        where = '句首' if before == 0 else '句末' if before >= len(report['units']) else f"第{before}个字「{report['units'][before]['char']}」之前"
        lines.append(f"{where}有 {len(letters)} 个字母没有对应上：{letters}，可以读作 {segment(letters)}")
    return lines[:NAMED]


def post(path, payload):
    request = Request(ENDPOINT + path, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def ask(post, fields, shown, text, report):
    lines = [*hint_context(fields), f"按键：{fields['keys']}", f"拼音：{shown}", f"转写：{text}", "对不上：", *spots(report)]
    packet = post('/v1/chat/completions', {'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256,
                                           'messages': [{'role': 'system', 'content': REPAIR_SYSTEM}, {'role': 'user', 'content': '\n'.join(lines)}]})
    content = packet['choices'][0]['message']['content']
    revised = content.strip() if isinstance(content, str) else ''
    turned = set(filter(str.islower, revised)) - set(filter(str.islower, text)) and characters(revised) < characters(text)
    chosen = fields.get('selected', {}).values()
    if not revised or revised == text or any(value not in revised for value in chosen) or turned:
        return text, report
    second = align(fields['keys'], revised)
    if not second or second['cost'] >= report['cost'] or (PAIRS.loaded and PAIRS.novel(revised, text)):
        return text, report
    return revised, second


def main():
    cases_path, first_path, variant, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    cases = {row['id']: row for row in (json.loads(line) for line in open(cases_path, encoding='utf-8'))}
    first = {row['id']: row for row in (json.loads(line) for line in open(first_path, encoding='utf-8')) if row['variant'] == variant}
    with out.open('w', encoding='utf-8') as handle:
        for case_id, reply in first.items():
            row = cases[case_id]
            fields = {key: row[key] for key in ('keys', 'context') if key in row}
            shown = syllables(row['text'], 'spaced')
            text, report = reply['text'], align(row['keys'], reply['text']) if reply['text'] else None
            started, gains = time.perf_counter(), []
            for _ in range(rounds):
                if report is None or report['cost'] == 0 or not spots(report):
                    break
                before = report['cost']
                text, report = ask(post, fields, shown, text, report)
                gains.append(before - report['cost'])
                if report['cost'] == before:
                    break
            asking = round((time.perf_counter() - started) * 1000)
            handle.write(json.dumps({'variant': variant + ' + ask again', 'id': case_id, 'text': text, 'gold': row['text'], 'keys': row['keys'],
                                     'source': row['source_id'], 'context': bool(row.get('context')), 'ms': asking + reply['ms'], 'ask_ms': asking,
                                     'rounds': len(gains), 'gains': gains, 'verify': summary(report) if report else None, 'first': reply['text']},
                                    ensure_ascii=False) + '\n')
            handle.flush()


if __name__ == '__main__':
    main()
