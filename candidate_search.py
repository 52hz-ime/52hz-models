"""Bounded token branching on llama.cpp, followed by a closed-pool decision."""

import json
from itertools import permutations
from time import perf_counter


POOL_LIMIT = 5
STRING_REST = r'''root ::= char* "\""
char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})'''


def forks(content, token_rows):
    """Prefer a fork before a repeated word, then alternatives near the greedy token."""
    encoded = content.encode()
    offset = 0
    choices = []
    for index, token in enumerate(token_rows):
        data = bytes(token['bytes'])
        if not data or not encoded.startswith(data, offset):
            break
        offset += len(data)
        if not token['token'].isalnum():
            continue
        try:
            before = encoded[:offset - len(data)].decode()
        except UnicodeDecodeError:
            continue
        repeats = index + 1 < len(token_rows) and token_rows[index + 1]['id'] == token['id']
        for other in token['top_logprobs']:
            if other['id'] == token['id'] or not other['token'].isalnum():
                continue
            gap = token['logprob'] - other['logprob']
            if gap <= 4 and bytes(other['bytes']) == other['token'].encode():
                choices.append((not repeats, gap, index, before + other['token']))
    return list(dict.fromkeys(prefix for *_, prefix in sorted(choices)))[:POOL_LIMIT + 1]


def generate(post, messages, locks):
    formatted = post('/apply-template', {'messages': messages, 'chat_template_kwargs': {'enable_thinking': False}})
    prompt = formatted['prompt'] + '{\n  "action": "commit",\n  "candidates": [\n    "'

    def complete(prefix, probabilities):
        return post('/completion', {'prompt': prompt + prefix, 'grammar': STRING_REST, 'temperature': 0,
                                    'n_probs': probabilities, 'n_predict': 256, 'cache_prompt': True})

    pool = []

    def collect(prefix, continuation):
        try:
            sentence = json.loads('"' + prefix + continuation['content'])
        except ValueError:
            return
        if sentence.strip() and sentence not in pool and all(value in sentence for value in locks):
            pool.append(sentence)

    first = complete('', 8)
    collect('', first)
    for prefix in forks(first['content'], first['completion_probabilities']):
        collect(prefix, complete(prefix, 0))
        if len(pool) == POOL_LIMIT:
            break
    return pool


def judge_schema(labels, actions):
    branches = []
    for action in actions:
        fields = {'action': {'type': 'string', 'enum': [action]}, 'choice': {'type': 'string', 'enum': labels}}
        if action == 'ask':
            fields.update(focus={'type': 'string'}, options={'type': 'array', 'items': {'type': 'string'},
                                                             'minItems': 2, 'maxItems': 4})
        branches.append({'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False})
    return branches[0] if len(branches) == 1 else {'oneOf': branches}


def ranked(post, chat, fields, prompt, actions, instruction):
    started = perf_counter()
    locks = [value for part, value in fields.get('selected', {}).items()
             if actions != ('ask',) or part != fields.get('focus')]
    messages = [{'role': 'system', 'content': prompt + '\n' + instruction},
                {'role': 'user', 'content': json.dumps(fields, ensure_ascii=False)}]
    pool = generate(post, messages, locks)
    if actions == ('ask',) and len(pool) < POOL_LIMIT:
        variants = chat('用户要求修改 focus。draft 是完整首选。仅改变 focus 对应的词，生成少量同音或错漏键的替代整句。'
                        '即使首选符合背景，也要考虑其他合理意图。不要复述上文、添加内容、标点或 Markdown。',
                        {**fields, 'draft': pool[0] if pool else ''}, {'type': 'object', 'properties': {'sentences': {'type': 'array', 'items': {'type': 'string'},
                                 'minItems': 2, 'maxItems': 4}}, 'required': ['sentences'], 'additionalProperties': False}, 'ime_more')
        alternatives = variants.get('sentences')
        if isinstance(alternatives, list):
            for item in alternatives:
                if isinstance(item, str) and item.strip() and item not in pool and all(value in item for value in locks):
                    pool.append(item)
                    if len(pool) == POOL_LIMIT:
                        break
    generated = perf_counter()
    trace = {'pool': pool, 'generate_ms': round((generated - started) * 1000)}
    if not pool:
        return {'error': '没有得到满足用户已选片段的完整候选。', 'search': trace}

    labels = dict(zip('ABCDE', sorted(pool)))
    decision = chat('你在比较几句已经写好的输入结果。第一优先级是逐音节忠实于原始 keys：不能增删、概括、改写或替换 keys 没有依据的汉字，不能把“在今天”改成“今天”，也不能把正文改成新闻标题。'
                    '第二优先级才是上下文中的语义自然度；英文、型号、数字、大小写、空格和标点不要擅改。同样保真时再选更自然的一项，不用近义词替换用户打出的词。'
                    'choice 填标签，列表顺序没有含义。仅当有证据区分合理意图才直接提交；上下文同时支持多项或没说明指谁时，'
                    '询问能区分意图的一处，不为生僻错字打断用户。ask 的 focus 填 keys 中的原始子串，options 只填不同候选中该片段的译文。'
                    'selected 片段已确定，不再询问。\n' + instruction,
                    {**fields, 'options': labels}, judge_schema(list(labels), actions), 'ime_judge')
    trace['judge_ms'] = round((perf_counter() - generated) * 1000)
    winner = labels.get(decision.get('choice')) if isinstance(decision.get('choice'), str) else None
    if winner is None or decision.get('action') not in actions:
        return {'error': '裁判未返回合法候选编号或动作。', 'search': trace}
    trace['winner'] = winner
    action = decision['action']
    if action == 'ask':
        focus, options = decision.get('focus'), decision.get('options')
        if (not isinstance(focus, str) or not focus or focus not in fields['keys']
                or ('focus' in fields and fields['focus'] != focus)
                or (focus in fields.get('selected', {}) and focus != fields.get('focus'))
                or not isinstance(options, list) or not all(isinstance(item, str) and item for item in options)):
            return {'error': '裁判没有返回有效的待确认片段。', 'search': trace}
        options = list(dict.fromkeys(item for item in options if any(item in sentence for sentence in pool)))
        if not 2 <= len(options) <= min(4, len(pool)) or not any(
                all(option in sentence for option, sentence in zip(options, assignment))
                for assignment in permutations(pool, len(options))):
            return {'error': '候选池中没有足够的不同局部译法；保留原始输入。', 'search': trace}
        return {'action': action, 'focus': focus, 'options': options, 'search': trace}
    return {'action': action, 'candidates': [winner, *(item for item in pool if item != winner)][:4], 'search': trace}
