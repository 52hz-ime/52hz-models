"""Run blind IME requests against a local llama.cpp chat endpoint."""

import argparse
import json
import os
import sys
from http.client import HTTPException
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from candidate_search import ranked


PROMPT = """任务：把 keys 中的连续汉语拼音按键，逐音节转换成当前要上屏的中文。只转换 keys，不回复聊天，不复述背景。
先按 keys 的原始顺序切分并还原每个拼音音节，再用 context 和 memory 解决同音字；不要把整句概括、改写、翻译成标题或替换成近义句。
context 是之前已经写完的文字；memory 是参考信息。它们只帮助在发音相近的答案之间做选择，不属于待转换的输入。
例如 context 是“车票订好了。”，keys 是 wozaihuochezhan，首选应为“我在火车站”，不是“车票订好了。”。
允许漏键、相邻键误触和音节交换，但只能修复这些按键错误；没有按键依据时不得增删汉字、语气词或改写句式。
原样保留 keys 中的英文名、数字、大小写、空格和标点；不要额外补充内容或句末标点。候选必须覆盖 keys 的整段输入，不能只输出标题、摘要或关键词。
selected 中用户选定的译文不可改动。明确时 candidates 给出整段译文，首选在前；需要消歧时 focus 是 keys 中的原始按键子串，options 只给该处的不同中文译法。"""


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
    fields = {key: row[key] for key in ('context', 'memory', 'selected', 'focus', 'keys') if key in row}

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
    if mode == 'direct':
        result = chat(PROMPT + '\n' + instruction, fields, output_schema(actions), 'ime_decision')
    elif mode == 'ranked':
        result = ranked(post, chat, fields, PROMPT, actions, instruction)
    else:
        raise ValueError('IME_DECODER must be ranked or direct')
    return {**result, 'id': row['id']}


def run(endpoint, requests, responses):
    if requests.resolve() == responses.resolve():
        raise ValueError('requests and responses must use different paths')
    with requests.open(encoding='utf-8') as source, NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=responses.parent, prefix=responses.name + '.',
            suffix='.partial', delete=False) as output:
        print(f'推理服务：{endpoint}；进行中的响应：{output.name}', file=sys.stderr)
        seen = set()
        for number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
                case_id = row['id']
                if not isinstance(case_id, str) or case_id in seen or not isinstance(row['keys'], str):
                    raise ValueError('invalid id or keys, or duplicate id')
                seen.add(case_id)
                reply = predict(endpoint, row)
            except HTTPError as error:
                raise ValueError(f'第 {number} 条请求 HTTP {error.code}: {error.read(500).decode(errors="replace")}') from error
            except (OSError, HTTPException) as error:
                raise ValueError(f'第 {number} 条请求未完成：{error}。请检查 make status / make serve。') from error
            except (ValueError, KeyError, TypeError, IndexError) as error:
                raise ValueError(f'第 {number} 条请求或服务响应格式错误：{error}') from error
            print(json.dumps(reply, ensure_ascii=False), file=output, flush=True)
        if not seen:
            raise ValueError('empty request file')
    Path(output.name).replace(responses)
    print(f'已写入 {len(seen)} 条响应：{responses}', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('endpoint')
    parser.add_argument('requests', type=Path)
    parser.add_argument('responses', type=Path)
    args = parser.parse_args()
    try:
        run(args.endpoint, args.requests, args.responses)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
