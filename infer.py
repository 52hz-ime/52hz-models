"""Run blind IME requests against a local llama.cpp chat endpoint."""

import argparse
import json
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from candidate_search import ranked
from fidelity import reliable
from syllables import segment


PROMPT = """任务：把 keys 中的连续汉语拼音按键，逐音节转换成当前要上屏的中文。只转换 keys，不回复聊天，不复述背景。
keys 是唯一需要转换的机械输入；context、memory 和 selected 都不是待上屏文本，尤其不能把 context 原样复述或当成候选答案。
先按 keys 的原始顺序切分并还原每个拼音音节，再用 context 和 memory 解决同音字；不要把整句概括、改写、翻译成标题或替换成近义句。
context 是之前已经写完的文字；memory 是参考信息。它们只帮助在发音相近的答案之间做选择，不属于待转换的输入。
例如 context 是“车票订好了。”，keys 是 wozaihuochezhan，首选应为“我在火车站”，不是“车票订好了。”。
允许漏键、相邻键误触和音节交换，但只能修复这些按键错误；没有按键依据时不得增删汉字、语气词或改写句式。
原样保留 keys 中的英文名、数字、大小写、空格和标点；不要额外补充内容或句末标点。候选必须覆盖 keys 的整段输入，不能只输出标题、摘要或关键词。
selected 中用户选定的译文不可改动。明确时 candidates 给出整段译文，首选在前；需要消歧时 focus 是 keys 中的原始按键子串，options 只给该处的不同中文译法。"""


HINT_SYSTEM = """你是拼音输入法的解码器。把「按键」里的连写拼音转成用户想打的中文，只输出转出来的文字。
- 「切分」是把按键机械切成音节的结果，仅供参考：带 ? 的位置很可能有按键错误（多按、漏按、按成相邻键、字母颠倒），没有 ? 的地方也可能漏了字母。以上文和按键为准，不要照抄切分。
- 按键里的英文、数字、标点原样保留，句末标点也要保留。
- 结合「上文」和「记忆」里的人名、叫法选同音字；「已选定」的文字原样保留。
- 每个字都要对得上按键的读音，不增字不删字，不改写、不概括，不复述上文。"""
HINT_EXAMPLES = [
    {"keys": "jintianwanshangchifan。", "text": "今天晚上吃饭。"},
    {"context": "会议安排：", "keys": "mingtian9diankaizongjiehui", "text": "明天9点开总结会"},
    {"context": "车票订好了。", "keys": "wozaihuochezhan", "text": "我在火车站"},
    {"keys": "woyaomaiyigeiPhone15", "text": "我要买一个iPhone15"},
    {"keys": "jintianwanshangchifann", "text": "今天晚上吃饭。"},
    {"keys": "jintianwanshangcihfan", "text": "今天晚上吃饭。"},
    {"keys": "jintianwanshangchifsn", "text": "今天晚上吃饭。"},
    {"keys": "jintianwannshangchifan", "text": "今天晚上吃饭。"},
    {"keys": "zhegewentiwomenmingtianzaitaolunba", "text": "这个提案我们明天再讨论吧"},
    {"context": "上次聊到开会。", "keys": "mingtianwoyoukaihuile", "text": "明天我又开会了"},
    {"context": "银行资料带齐了。", "keys": "mingtianwoqukaihu", "text": "明天我去开户"},
    {"context": "周报请发给组长。", "memory": ["我们组的组长叫王静。"], "keys": "qingfageiwangjing", "text": "请发给王静"},
    {"context": "银行已经预约好了。", "selected": {"kaihu": "开会"}, "keys": "mingtiankaihu", "text": "明天开会"},
]

def hint_user(fields):
    lines = [f"上文：{fields['context']}"] if fields.get('context') else []
    lines += [f"记忆：{item}" for item in fields.get('memory') or []]
    lines += [f"已选定：{text}（按键 {part}，不可改动）" for part, text in (fields.get('selected') or {}).items()]
    return "\n".join([*lines, f"按键：{fields['keys']}", f"切分：{segment(fields['keys'])}"])


def hinted(post, fields, actions):
    """One plain-text completion with examples and a syllable hint; the reply itself is the only candidate.

    This decoder never asks: a reply that reads like the keys is shown or committed, anything else is kept
    back. It returns None when the reply drops a selected fragment, so the caller falls back to the JSON
    contract, which the `more` event always uses.
    """
    examples = [message for example in HINT_EXAMPLES
                for message in ({'role': 'user', 'content': hint_user(example)}, {'role': 'assistant', 'content': example['text']})]
    packet = post('/v1/chat/completions', {'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256,
                                           'messages': [{'role': 'system', 'content': HINT_SYSTEM}, *examples,
                                                        {'role': 'user', 'content': hint_user(fields)}]})
    content = packet['choices'][0]['message']['content']
    text = content.strip() if isinstance(content, str) else ''
    if not text:
        return {'error': 'invalid model output', 'raw': content}
    if any(chosen not in text for chosen in fields.get('selected', {}).values()):
        return None
    if not reliable(fields['keys'], text):
        return {'error': '首选与按键的读音相差太远，保留原始输入供继续编辑。', 'rejected': text}
    return {'action': 'preview' if actions == ('preview',) else 'commit', 'candidates': [text]}


def decision_contract(row):
    event = row['event']
    if event == 'type' or (event == 'select' and row.get('origin') == 'more'):
        return ('preview',), '正在编辑，返回 preview 和整段候选，不提交。'
    if event == 'more':
        return ('ask',), '用户主动修改 focus；返回 ask，只列该处的译法。'
    if event == 'space' or (event == 'select' and row.get('origin') == 'space'):
        return ('commit', 'ask'), '准备上屏。根据按键和背景能判定就 commit；如仍有实质歧义，只 ask 最有助于消歧的一处。'
    raise ValueError(f'unsupported IME event: {event}')


def output_schema(actions):
    branches = []
    texts = {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1, 'maxItems': 4}
    for action in actions:
        fields = {'action': {'type': 'string', 'enum': [action]}}
        if action == 'ask':
            fields.update(focus={'type': 'string'}, options={**texts, 'minItems': 2})
        else:
            fields['candidates'] = texts
        branches.append({'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False})
    return branches[0] if len(branches) == 1 else {'oneOf': branches}


def predict(endpoint, row):
    actions, instruction = decision_contract(row)
    fields = {key: row[key] for key in ('keys', 'context', 'memory', 'selected', 'focus') if key in row}

    def post(path, payload):
        request = Request(endpoint.rstrip('/') + path, data=json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=120) as response:
            return json.load(response)

    def chat(system, data, schema, name):
        packet = post('/v1/chat/completions', {'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256,
                      'response_format': {'type': 'json_schema', 'json_schema': {'name': name, 'schema': schema}},
                      'messages': [{'role': 'system', 'content': system},
                                   {'role': 'user', 'content': json.dumps(data, ensure_ascii=False)}]})
        content = packet['choices'][0]['message']['content']
        try:
            result = json.loads(content)
            if not isinstance(result, dict):
                raise ValueError('expected JSON object')
            return result
        except (ValueError, TypeError):
            return {'error': 'invalid model output', 'raw': content}

    mode = os.environ.get('IME_DECODER', 'ranked')
    # The hinted decoder handles typing and commits; asking on demand and dropped selections use the JSON contract.
    result = hinted(post, fields, actions) if mode == 'hint' and row['event'] in ('type', 'space') else None
    if result is None:
        if mode in ('direct', 'hint'):
            result = chat(PROMPT + '\n' + instruction, fields, output_schema(actions), 'ime_decision')
        elif mode == 'ranked':
            result = ranked(post, chat, fields, PROMPT, actions, instruction)
        else:
            raise ValueError('IME_DECODER must be hint, ranked or direct')
    return {**result, 'id': row['id']}


def numbered_rows(source):
    seen = set()
    for number, line in enumerate(source, 1):
        try:
            row = json.loads(line)
            case_id = row['id']
            if not isinstance(case_id, str) or case_id in seen or not isinstance(row['keys'], str):
                raise ValueError('invalid id or keys, or duplicate id')
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'第 {number} 条请求或服务响应格式错误：{error}') from error
        seen.add(case_id)
        yield number, row


def infer_rows(endpoint, rows, workers):
    """Infer the rows with at most `workers` requests in flight and yield the replies in request order."""
    def attempt(item):
        number, row = item
        try:
            return predict(endpoint, row)
        except HTTPError as error:
            raise ValueError(f'第 {number} 条请求 HTTP {error.code}: {error.read(500).decode(errors="replace")}') from error
        except (OSError, HTTPException) as error:
            raise ValueError(f'第 {number} 条请求未完成：{error}。请检查 make status / make serve。') from error
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise ValueError(f'第 {number} 条请求或服务响应格式错误：{error}') from error

    if workers <= 1:
        for item in rows:
            yield attempt(item)
        return
    pending, items = deque(), iter(rows)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for item in items:
            pending.append(pool.submit(attempt, item))
            if len(pending) == workers:
                break
        while pending:
            yield pending.popleft().result()
            item = next(items, None)
            if item is not None:
                pending.append(pool.submit(attempt, item))


def run(endpoint, requests, responses, workers=1):
    if requests.resolve() == responses.resolve():
        raise ValueError('requests and responses must use different paths')
    with requests.open(encoding='utf-8') as source, NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=responses.parent, prefix=responses.name + '.',
            suffix='.partial', delete=False) as output:
        print(f'推理服务：{endpoint}；进行中的响应：{output.name}', file=sys.stderr)
        count = 0
        for reply in infer_rows(endpoint, numbered_rows(source), workers):
            print(json.dumps(reply, ensure_ascii=False), file=output, flush=True)
            count += 1
        if not count:
            raise ValueError('empty request file')
    Path(output.name).replace(responses)
    print(f'已写入 {count} 条响应：{responses}', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('endpoint')
    parser.add_argument('requests', type=Path)
    parser.add_argument('responses', type=Path)
    parser.add_argument('--workers', type=int, default=1, help='同时在途的请求数；服务端需要相同的 --parallel')
    args = parser.parse_args()
    try:
        if args.workers < 1:
            raise ValueError('workers must be at least 1')
        run(args.endpoint, args.requests, args.responses, args.workers)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
