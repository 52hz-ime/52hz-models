"""The same clean inputs under several prompts: is it the model or our prompt that loses the clean cases?

usage: python experiments/clean_prompt_ab.py OUT.jsonl   (run from the repository root, with the model server up; requests go one at a time)
The clean cases are the ones without noise in trial-3000 and holdout-3000.
"""
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from infer import HINT_EXAMPLES, HINT_SYSTEM, hint_context  # noqa: E402

ENDPOINT = 'http://127.0.0.1:8089'
BARE = '把用户输入的连写汉语拼音转成中文，只输出转出来的文字。'
VARIANTS = ('bare', 'bare + context', 'examples, no error warning', 'examples, with error warning')
SETS = [('/data/52hz-eval/build/benchmark/iteration-003/trial-3000.jsonl', 'runs/round-2026-11-27d/trial-3000/serial/verify-repair-4'),
        ('/data/52hz-eval/build/benchmark/iteration-003/holdout-3000.jsonl', 'runs/round-2026-11-27d/holdout-3000/serial/final-2')]
cases = []
for cases_file, run in SETS:
    fields = {f['id']: f for f in (json.loads(l) for l in open(run + '/requests.jsonl', encoding='utf-8'))}
    cases += [(c, fields[c['id']]) for c in (json.loads(l) for l in open(cases_file, encoding='utf-8')) if not c['noise_ops']]
no_error_warning = '\n'.join(line for line in HINT_SYSTEM.split('\n') if not line.startswith('- 「切分」'))
examples = []
for e in HINT_EXAMPLES:
    lines = [f"上文：{e['context']}"] if e.get('context') else []
    lines += [f"记忆：{m}" for m in e.get('memory') or []] + [f"已选定：{t}（按键 {k}，不可改动）" for k, t in (e.get('selected') or {}).items()] + [f"按键：{e['keys']}"]
    examples += [{'role': 'user', 'content': '\n'.join(lines)}, {'role': 'assistant', 'content': e['text']}]


def messages(variant, f):
    typed = '\n'.join([*hint_context(f), f['keys'] if variant.startswith('bare') else f"按键：{f['keys']}"])
    if variant.startswith('bare'):
        return [{'role': 'system', 'content': BARE}, {'role': 'user', 'content': f['keys'] if variant == 'bare' else typed}]
    system = no_error_warning if variant == 'examples, no error warning' else HINT_SYSTEM
    return [{'role': 'system', 'content': system}, *examples, {'role': 'user', 'content': typed}]


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
