"""Sample a reply several times, keep the ones that read back as the pinyin, and take the most common of those.

usage: python experiments/sample_and_vote.py CASES.jsonl OUT.jsonl VARIANT SAMPLES TEMPERATURE
       (run from the repository root, with the model server up; requests go one at a time, each with its own fixed seed)

Nothing here needs a grammar: only candidates whose reading is exactly the pinyin shown can win, and the
expected text is always among those. A fixed seed per sample keeps the run repeatable.
"""
import collections
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.pinyin_conversion import ENDPOINT, prompt  # noqa: E402
from verify import align  # noqa: E402


def post(payload):
    request = Request(ENDPOINT + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=120) as response:
        return json.load(response)['choices'][0]['message']['content']


def main():
    cases_path, out, variant = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    samples, temperature = int(sys.argv[4]), float(sys.argv[5])
    rows = [json.loads(line) for line in open(cases_path, encoding='utf-8')]
    with out.open('a', encoding='utf-8') as handle:
        for row in rows:
            fields = {key: row[key] for key in ('keys', 'context') if key in row}
            messages = prompt(variant, fields)
            started, drawn = time.perf_counter(), []
            for number in range(samples):
                text = post({'model': '52hz-qwen', 'temperature': temperature, 'top_p': 0.95, 'max_tokens': 256, 'seed': number + 1,
                             'messages': messages}).strip()
                drawn.append({'text': text, 'cost': align(row['keys'], text)['cost'] if text else None})
            clean = [sample['text'] for sample in drawn if sample['cost'] == 0]
            votes = collections.Counter(clean)
            best = votes.most_common(1)[0][0] if votes else (min((s for s in drawn if s['cost'] is not None), key=lambda s: s['cost'])['text'] if any(s['cost'] is not None for s in drawn) else '')
            handle.write(json.dumps({'variant': variant, 'id': row['id'], 'text': best, 'gold': row['text'], 'keys': row['keys'], 'source': row['source_id'],
                                     'context': bool(row.get('context')), 'ms': round((time.perf_counter() - started) * 1000), 'samples': samples, 'temperature': temperature,
                                     'clean': len(clean), 'distinct_clean': len(votes), 'votes': votes.most_common(3),
                                     'first': drawn[0]['text'], 'first_cost': drawn[0]['cost']}, ensure_ascii=False) + '\n')
            handle.flush()


if __name__ == '__main__':
    main()
