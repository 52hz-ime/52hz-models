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
from time import perf_counter
from unicodedata import category
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from candidate_search import ranked
from lexicon import PAIRS
from pinyin import CJK, CJK_RUN
from syllables import alternatives, segment
from verify import align, summary


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
    {"keys": "jintianwanshangchifann", "text": "今天晚上吃饭"},
    {"keys": "jintianwanshangcihfan", "text": "今天晚上吃饭"},
    {"keys": "jintianwanshangchifsn", "text": "今天晚上吃饭"},
    {"keys": "jintianwannshangchifan", "text": "今天晚上吃饭"},
    {"keys": "zhegewentiwomenmingtianzaitaolunba", "text": "这个问题我们明天再讨论吧"},
    {"context": "上次聊到开会。", "keys": "mingtianwoyoukaihuile", "text": "明天我又开会了"},
    {"context": "银行资料带齐了。", "keys": "mingtianwoqukaihu", "text": "明天我去开户"},
    {"context": "周报请发给组长。", "memory": ["我们组的组长叫王静。"], "keys": "qingfageiwangjing", "text": "请发给王静"},
    {"context": "银行已经预约好了。", "selected": {"kaihu": "开会"}, "keys": "mingtiankaihu", "text": "明天开会"},
]

REPAIR_SYSTEM = """用户打了一段带错误的拼音。「转写」是第一次的转换结果，但有几处和按键对不上，列在「对不上」里：每一处给出按键在这一位置的字母，以及它在一步按键错误或口音合并之内还能读成的音节；写着“没有对应上”的地方，是转写在那里漏了字。
只改这些位置，让每个字都对得上按键的读音，漏掉的字补回去；其余文字保持不变，并结合上文选最通顺的读法。只输出改写后的文字。"""
ROUNDS = 2
POSITIONS = 4
# A reply that needs this many errors is looked at again, within its budget or not, when the verifier has
# evidence against it; fewer unclaimed letters than GAP is a stray keystroke, not a word left out.
ENTER = 2
GAP = 2


def hint_context(fields):
    lines = [f"上文：{fields['context']}"] if fields.get('context') else []
    lines += [f"记忆：{item}" for item in fields.get('memory') or []]
    lines += [f"已选定：{text}（按键 {part}，不可改动）" for part, text in (fields.get('selected') or {}).items()]
    return lines


def hint_user(fields):
    return "\n".join([*hint_context(fields), f"按键：{fields['keys']}", f"切分：{segment(fields['keys'])}"])


def problems(report):
    """What a repair should look at.

    The characters the verifier has evidence against, and every stretch of keys that no character claims,
    with where it sits in the sentence and how it reads: that is where the reply left something out. A
    reading one keystroke from the keys is a valid reading; such characters are listed only when the
    sentence is over its budget and there is nothing else to point at.
    """
    lines = []
    gaps = [gap for gap in report["gaps"] if len(gap["letters"]) >= GAP]
    for index in (report["doubts"] or ([] if gaps else report["flagged"]))[:POSITIONS]:
        unit = report["units"][index]
        line = f"第{index + 1}个字「{unit['char']}」读作 {unit['reading']}，按键这一处是 {unit['typed'] or '空'}"
        others = alternatives(unit["typed"], (unit["reading"], unit["typed"]))
        lines.append(line + (f"，也可以读作 {' / '.join(others)}" if others else ""))
    for gap in gaps:
        letters = gap["letters"]
        before = gap["before"]
        if before == 0:
            where = "句首"
        elif before >= len(report["units"]):
            where = "句末"
        else:
            where = f"第{before}个字「{report['units'][before]['char']}」之前"
        lines.append(f"{where}有 {len(letters)} 个字母没有对应上：{letters}，可以读作 {segment(letters)}")
    if not lines:
        over = f"，超过了 {report['budget']} 的额度" if not report['consistent'] else ''
        lines.append(f"整句要 {report['cost']} 个错误才能解释按键{over}；请重新考虑各字的读法")
    return lines


def repair(post, fields, text, report):
    """Ask for the flagged characters alone to be rewritten.

    The rewrite is kept only when it fits the keys better and, when the corpus word list is available,
    introduces no character pair the corpus has never seen: a rewrite can always be made to fit the keys
    by writing nonsense, and that must not reach the user.
    """
    lines = [*hint_context(fields), f"按键：{fields['keys']}", f"切分：{segment(fields['keys'])}", f"转写：{text}",
             "对不上：", *problems(report)]
    packet = post('/v1/chat/completions', {'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256,
                                           'messages': [{'role': 'system', 'content': REPAIR_SYSTEM},
                                                        {'role': 'user', 'content': "\n".join(lines)}]})
    content = packet['choices'][0]['message']['content']
    revised = content.strip() if isinstance(content, str) else ''
    chosen = fields.get('selected', {}).values()
    # A rewrite that turns characters into letters, the keys echoed back for instance, fits the keys only
    # because letters match themselves; the decoder turns pinyin into characters and never the other way.
    # Writing what was typed in capitals in lower case is allowed: it turns no character into letters.
    turned_into_letters = set(filter(str.islower, revised)) - set(filter(str.islower, text)) and characters(revised) < characters(text)
    if not revised or revised == text or any(value not in revised for value in chosen) or turned_into_letters:
        return text, report
    second = align(fields['keys'], revised)
    if not second or second['cost'] >= report['cost'] or (PAIRS.loaded and PAIRS.novel(revised, text)):
        return text, report
    return revised, second


def characters(value):
    """How many Chinese characters a text has."""
    return sum(map(len, CJK_RUN.findall(value)))


def unfinished(report):
    """Whether a reply still needs repair.

    Over its budget, always. Within it, when it needs ENTER or more errors and the verifier has evidence
    against it: a reading two or more keystrokes from the keys, or a stretch of keys no character claims.
    """
    if not report['consistent']:
        return True
    # Only a reading two or more keystrokes away counts; a comma nobody typed is not evidence of a misreading.
    far = [index for index in report['doubts'] if report['units'][index]['cost'] >= ENTER]
    return report['cost'] >= ENTER and bool(far or any(len(gap['letters']) >= GAP for gap in report['gaps']))


def settle(post, fields, text, report):
    """Repair a reply until it explains the keys or a round stops improving it.

    Returns the reply, its report and the number of rounds run.
    """
    rounds = 0
    while unfinished(report) and rounds < ROUNDS:
        earlier = report['cost']
        rounds += 1
        text, report = repair(post, fields, text, report)
        if report['cost'] == earlier:
            break
    return text, report, rounds


def literal(char):
    """Punctuation, a symbol or a space: typed as itself, or not typed at all."""
    return category(char)[0] in "PSZ"


def tidied(fields, text):
    """Give the typed punctuation its due, without the model: keep it as typed, and add none the user did not type.

    Returns the text and its report. Chinese characters and typed letters and digits are never touched.
    """
    report = align(fields['keys'], text)
    unclaimed = lambda before: "".join(gap["letters"] for gap in report["gaps"] if gap["before"] == before)
    pieces = []
    for index, unit in enumerate(report["units"]):
        gap = unclaimed(index)
        if gap and all(map(literal, gap)):
            pieces.append(gap)                       # a comma the reply left out, where the user typed it
        char, typed = unit["char"], unit["typed"]
        if not CJK.match(char) and literal(char) and (not typed or literal(typed)):
            if typed:
                pieces.append(typed)                 # as typed: ? stays ?, not ？
        else:
            pieces.append(char)                      # punctuation nobody typed is left out
    tail = unclaimed(len(report["units"]))
    pieces.append(tail if tail and all(map(literal, tail)) else "")
    revised = "".join(pieces)
    if revised == text:
        return text, report
    second = align(fields["keys"], revised)
    return (revised, second) if second and second["cost"] < report["cost"] else (text, report)


def looked_after(post, fields, text):
    """Everything that follows a first reply before it is shown as final.

    The typed punctuation is put right without the model, the reply is repaired while the verifier has
    evidence against it, and the punctuation is put right again. Returns the text, its report and the rounds.
    """
    text, report = tidied(fields, text)
    text, report, rounds = settle(post, fields, text, report)
    return (*tidied(fields, text), rounds)


def hinted(post, fields, actions):
    """One completion with examples and a syllable hint, checked against the keys, repaired where it fails.

    The reply is the only candidate. It is kept when it reads like the keys within the error budget;
    otherwise the characters that do not fit are rewritten, at most ROUNDS times, and the rewrite is kept
    only if it fits better. What still does not fit is not shown. This decoder never asks.

    Returns None when the reply drops a selected fragment, so the caller falls back to the JSON contract,
    which the `more` event always uses.
    """
    examples = [message for example in HINT_EXAMPLES
                for message in ({'role': 'user', 'content': hint_user(example)}, {'role': 'assistant', 'content': example['text']})]
    packet = post('/v1/chat/completions', {'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 256,
                                           'messages': [{'role': 'system', 'content': HINT_SYSTEM}, *examples,
                                                        {'role': 'user', 'content': hint_user(fields)}]})
    content = packet['choices'][0]['message']['content']
    text = content.strip() if isinstance(content, str) else ''
    report = align(fields['keys'], text) if text else None
    if report is None:
        return {'error': 'invalid model output', 'raw': content}
    if any(chosen not in text for chosen in fields.get('selected', {}).values()):
        return None
    text, report = tidied(fields, text)
    trace = {'rounds': 0, 'before': report['cost'], 'first': text}
    if actions != ('preview',):
        text, report, trace['rounds'] = looked_after(post, fields, text)
    trace['after'] = report['cost']
    if actions != ('preview',) and not report['consistent']:
        return {'error': '首选解释不了这串按键，保留原始输入供继续编辑。', 'rejected': text, 'verify': summary(report), 'repair': trace}
    return {'action': 'preview' if actions == ('preview',) else 'commit', 'candidates': [text], 'verify': summary(report),
            'repair': trace}


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

    mode = os.environ.get('IME_DECODER', 'hint')
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
        started = perf_counter()
        try:
            reply = predict(endpoint, row)
        except HTTPError as error:
            raise ValueError(f'第 {number} 条请求 HTTP {error.code}: {error.read(500).decode(errors="replace")}') from error
        except (OSError, HTTPException) as error:
            raise ValueError(f'第 {number} 条请求未完成：{error}。请检查 make status / make serve。') from error
        except (ValueError, KeyError, TypeError, IndexError) as error:
            raise ValueError(f'第 {number} 条请求或服务响应格式错误：{error}') from error
        # Wall-clock time of the whole request, the verification and any repair calls included.
        return {**reply, 'ms': round((perf_counter() - started) * 1000)}

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
