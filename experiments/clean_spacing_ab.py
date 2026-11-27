"""The same clean inputs with the syllables shown in different ways: how much of the loss is the joined spelling?

usage: python experiments/clean_spacing_ab.py OUT.jsonl   (run from the repository root, with the model server up; requests go one at a time)
The clean cases are the ones without noise in trial-3000 and holdout-3000.
"""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer import HINT_EXAMPLES, HINT_SYSTEM, hint_context, hint_user  # noqa: E402
from syllables import segment  # noqa: E402

ENDPOINT = 'http://127.0.0.1:8089'
BARE = '把用户输入的连写汉语拼音转成中文，只输出转出来的文字。'
VARIANTS = ('bare, syllables spaced', 'bare, syllables spaced + context', 'pipeline prompt (control)', 'examples, spaced syllables as the input')
SETS = [('/data/52hz-eval/build/benchmark/iteration-003/trial-3000.jsonl', 'runs/round-2026-11-27d/trial-3000/serial/verify-repair-4'),
        ('/data/52hz-eval/build/benchmark/iteration-003/holdout-3000.jsonl', 'runs/round-2026-11-27d/holdout-3000/serial/final-2')]
cases = []
for cases_file, run in SETS:
    fields = {f['id']: f for f in (json.loads(l) for l in open(run + '/requests.jsonl', encoding='utf-8'))}
    cases += [(c, fields[c['id']]) for c in (json.loads(l) for l in open(cases_file, encoding='utf-8')) if not c['noise_ops']]
spaced_user = lambda f: '\n'.join([*hint_context(f), f"拼音（按音节分开）：{segment(f['keys'])}"])
spaced_examples = []
for e in HINT_EXAMPLES:
    lines = [f"上文：{e['context']}"] if e.get('context') else []
    lines += [f"记忆：{m}" for m in e.get('memory') or []] + [f"已选定：{t}（按键 {k}，不可改动）" for k, t in (e.get('selected') or {}).items()]
    lines += [f"拼音（按音节分开，带 ? 的音节可能按错了）：{segment(e['keys'])}"]
    spaced_examples += [{'role': 'user', 'content': '\n'.join(lines)}, {'role': 'assistant', 'content': e['text']}]
pipeline_examples = [m for e in HINT_EXAMPLES for m in ({'role': 'user', 'content': hint_user(e)}, {'role': 'assistant', 'content': e['text']})]


def messages(variant, f):
    if variant == 'bare, syllables spaced':
        return [{'role': 'system', 'content': BARE}, {'role': 'user', 'content': segment(f['keys'])}]
    if variant == 'bare, syllables spaced + context':
        return [{'role': 'system', 'content': BARE}, {'role': 'user', 'content': spaced_user(f)}]
    if variant == 'pipeline prompt (control)':
        return [{'role': 'system', 'content': HINT_SYSTEM}, *pipeline_examples, {'role': 'user', 'content': hint_user(f)}]
    system = '把按音节分开的汉语拼音转成用户想打的中文，只输出转出来的文字。带 ? 的音节可能按错了键，结合上文判断它本来是什么。'
    return [{'role': 'system', 'content': system}, *spaced_examples, {'role': 'user', 'content': spaced_user(f)}]


def post(payload):
    request = Request(ENDPOINT + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)['choices'][0]['message']['content']


with open(sys.argv[1], 'w', encoding='utf-8') as out:
    for variant in VARIANTS:
        for c, f in cases:
            text = post({'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256, 'messages': messages(variant, f)})
            out.write(json.dumps({'variant': variant, 'id': c['id'], 'text': text.strip(), 'gold': c['text'], 'keys': c['keys'], 'source': c['source_id']}, ensure_ascii=False) + '\n')
        out.flush()
        print(f'{variant}: {len(cases)} cases done', file=sys.stderr, flush=True)
