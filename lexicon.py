"""Character pairs counted in the corpus, to tell a real word from an invented spelling."""

import json
import os

from pinyin import CJK_RUN


# A pair has to have been seen at least this often to count as a word.
KNOWN = 10


class Pairs:
    def __init__(self, path=None):
        self.counts = {}
        path = path or os.environ.get('IME_LEXICON', '')
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if len(row["word"]) == 2:
                        self.counts[row["word"]] = row["count"]

    @property
    def loaded(self):
        return bool(self.counts)

    def novel(self, text, earlier):
        """Character pairs of a text that the corpus does not know and that the earlier text did not have."""
        pairs = lambda value: {run[index:index + 2] for run in CJK_RUN.findall(value) for index in range(len(run) - 1)}
        return sorted(pair for pair in pairs(text) - pairs(earlier) if self.counts.get(pair, 0) < KNOWN)


PAIRS = Pairs()
