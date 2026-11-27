"""How well the model turns a correct pinyin sequence into characters, apart from the decoder's harness.

usage: python experiments/pinyin_conversion.py CASES.jsonl OUT.jsonl VARIANT [VARIANT ...]
       (run from the repository root, with the model server up; requests go one at a time, so results repeat)

A variant names the form of the pinyin and what else the prompt carries, for example "spaced +context +examples":

  joined     nihaoma            as typed, without spaces or tones
  spaced     ni hao ma          one syllable at a time, cut where the expected text reads
  numbered   ni3 hao3 ma5       with the tone as a digit; information a keyboard does not give
  toned      nǐ hǎo ma          with the tone as a mark; the same
  +context   the lines typed before, as the decoder gets them
  +examples  the decoder's few-shot examples, written in the same form and without their errors
  +strict    the rule the decoder states: every character must read like its syllable, none added or left out

"decoder prompt" sends exactly the messages the decoder sends before a commit: the keys as typed and the
segmenter's line of syllables, under the decoder's system prompt, examples and context.

The syllables come from the expected text, so they are exactly right: this measures what the model makes of a
perfect pinyin sequence, not how well anything segments what was typed. Everything that is not Chinese is
shown as typed, the way benchmark keys are made, and the joined form is the keys.
"""
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer import HINT_EXAMPLES, HINT_SYSTEM, hint_context, hint_user  # noqa: E402
from pinyin import CJK  # noqa: E402
from pypinyin import Style, pinyin  # noqa: E402

ENDPOINT = 'http://127.0.0.1:8089'
STRICT = '输出的每个字都要对得上拼音的读音，不增字不删字，不改写、不概括。'
FORMS = {'joined': '连写', 'spaced': '按音节分开', 'numbered': '带声调数字', 'toned': '带声调符号'}
STYLES = {'joined': Style.NORMAL, 'spaced': Style.NORMAL, 'numbered': Style.TONE3, 'toned': Style.TONE}


RUN = re.compile(CJK.pattern + '+|[^' + CJK.pattern[1:-1] + ']+')        # a run of Chinese characters, or a run of anything else


def readings(text, form):
    """One item per character of the text: its reading, or the character itself when it is not Chinese."""
    items = []
    for run in RUN.findall(text):
        items += [reading[0] for reading in pinyin(run, style=STYLES[form], errors='default', neutral_tone_with_five=True)] if CJK.match(run[0]) else list(run)
    return items


def syllables(text, form):
    """The pinyin of the text, one item per syllable; a run of punctuation, digits or letters is one item, as typed."""
    items = []
    for run in RUN.findall(text):
        items += [reading[0] for reading in pinyin(run, style=STYLES[form], errors='default', neutral_tone_with_five=True)] if CJK.match(run[0]) else [run]
    return ''.join(items) if form == 'joined' else ' '.join(items)


def prompt(variant, row):
    if variant == 'decoder prompt':                  # exactly what the decoder sends for a commit
        examples = [message for example in HINT_EXAMPLES for message in ({'role': 'user', 'content': hint_user(example)}, {'role': 'assistant', 'content': example['text']})]
        fields = {key: row[key] for key in ('keys', 'context') if key in row}
        return [{'role': 'system', 'content': HINT_SYSTEM}, *examples, {'role': 'user', 'content': hint_user(fields)}]
    form = next(name for name in FORMS if variant.split()[0] == name)
    system = f'把{FORMS[form]}的汉语拼音转成用户想打的中文，只输出转出来的文字。' + (STRICT if '+strict' in variant else '')
    messages = [{'role': 'system', 'content': system}]
    if '+examples' in variant:
        for example in HINT_EXAMPLES:
            lines = [f"上文：{example['context']}"] if example.get('context') else []
            messages += [{'role': 'user', 'content': '\n'.join([*lines, syllables(example['text'], form)])},
                         {'role': 'assistant', 'content': example['text']}]
    lines = hint_context(row) if '+context' in variant else []
    messages.append({'role': 'user', 'content': '\n'.join([*lines, syllables(row['text'], form)])})
    return messages


def post(payload):
    request = Request(ENDPOINT + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)['choices'][0]['message']['content']


def main():
    cases_path, out, variants = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    rows = [json.loads(line) for line in open(cases_path, encoding='utf-8')]
    assert not any(row.get('noise_ops') for row in rows), 'this measures conversion without noise: the cases must be clean'
    for variant in variants:
        assert variant == 'decoder prompt' or variant.split()[0] in FORMS, variant
    with out.open('a', encoding='utf-8') as handle:
        for variant in variants:
            for number, row in enumerate(rows, 1):
                started = time.perf_counter()
                text = post({'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256, 'messages': prompt(variant, row)}).strip()
                handle.write(json.dumps({'variant': variant, 'id': row['id'], 'text': text, 'gold': row['text'], 'keys': row['keys'],
                                         'source': row['source_id'], 'context': bool(row.get('context')), 'ms': round((time.perf_counter() - started) * 1000)},
                                        ensure_ascii=False) + '\n')
            handle.flush()
            print(f'{variant}: {len(rows)} cases', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
