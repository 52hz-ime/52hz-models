"""How much the model favours one spelling over another, read from the logits of a one-letter answer."""

import json
import math
from urllib.request import Request, urlopen


SYSTEM = "你在判断用户打拼音时更想表达哪个词。只回答 A 或 B，不要解释。"
MISSING = -30.0


def preference(endpoint, context, keys, first, second, frame="", timeout=60):
    """Support for the second spelling against the first, from 0 to 1.

    The question is asked about one spot: the keys typed there, the sentence with that spot left blank
    and what came before it, so that both the grammar and the context can be weighed.
    """
    lines = [f"上文：{context}"] if context else []
    lines += [f"句子：{frame}"] if frame else []
    payload = {'model': '52hz-qwen', 'temperature': 0, 'max_tokens': 1, 'logprobs': True, 'top_logprobs': 8,
               'messages': [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': "\n".join([*lines, f"用户在这一处打的拼音：{keys}",
                                                                  f"填进去更通顺、更符合上文的是 A「{first}」还是 B「{second}」？"])}]}
    request = Request(endpoint.rstrip('/') + '/v1/chat/completions', data=json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=timeout) as response:
        packet = json.load(response)
    weights = {}
    for item in packet['choices'][0]['logprobs']['content'][0].get('top_logprobs', []):
        letter = item['token'].strip().strip('"“”`\'').upper()[:1]
        if letter in 'AB':
            weights[letter] = max(weights.get(letter, MISSING), item['logprob'])
    first_weight, second_weight = math.exp(weights.get('A', MISSING)), math.exp(weights.get('B', MISSING))
    return second_weight / (first_weight + second_weight)


def favour(endpoint, context, keys, expected, other, frame="", timeout=60):
    """How strongly the model favours the expected spelling over the other one, from 0 to 1.

    The question is asked once with each spelling in front and the two readings are averaged, so that a
    spelling does not win merely because the model prefers to answer A (or B).
    """
    in_front = 1.0 - preference(endpoint, context, keys, expected, other, frame, timeout)
    behind = preference(endpoint, context, keys, other, expected, frame, timeout)
    return (in_front + behind) / 2
