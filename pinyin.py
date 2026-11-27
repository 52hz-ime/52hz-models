"""Turn candidate text back into the pinyin keys it would be typed with."""

import re
from functools import lru_cache

from pypinyin import Style, pinyin


CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
CJK_RUN = re.compile(CJK.pattern + "+")


@lru_cache(maxsize=65536)
def to_keys(text):
    """Chinese runs become toneless pinyin and every other character is kept, the same way benchmark keys are made."""
    keys, run = [], []

    def flush():
        if run:
            keys.extend(item[0] for item in pinyin("".join(run), style=Style.NORMAL, heteronym=False, errors="keep"))
            run.clear()

    for char in text:
        if CJK.match(char):
            run.append(char)
        else:
            flush()
            keys.append(char)
    flush()
    return "".join(keys)


@lru_cache(maxsize=65536)
def units_of(text):
    """Split text into one unit per character: the character, the ways it can be typed, and whether it is Chinese.

    The first way is the phrase-aware reading that to_keys uses; the others are the character's other
    readings, so that a validator does not mistake a less common reading for a typo.
    """
    units, index = [], 0
    while index < len(text):
        if CJK.match(text[index]):
            end = index + 1
            while end < len(text) and CJK.match(text[end]):
                end += 1
            syllables = pinyin(text[index:end], style=Style.NORMAL, heteronym=True, errors="keep")
            units.extend((char, tuple(dict.fromkeys(item)), True) for char, item in zip(text[index:end], syllables))
            index = end
        else:
            units.append((text[index], (text[index],), False))
            index += 1
    return tuple(units)
