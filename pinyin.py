"""Turn candidate text back into the pinyin keys it would be typed with."""

import re
from functools import lru_cache

from pypinyin import Style, pinyin


CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


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
