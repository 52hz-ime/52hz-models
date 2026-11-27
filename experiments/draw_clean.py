"""Draw a set of clauses typed without any error, to measure how well the model turns pinyin into characters.

usage: python experiments/draw_clean.py CASES.jsonl OUT.jsonl [SIZE=1000] [SEED=3003]
The clean cases of the benchmark are drawn round-robin over source x length cells, so every source and every
length is represented; nothing is tuned on this set.
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BUCKETS = ((8, "4-8"), (14, "9-14"), (20, "15-20"), (10 ** 6, "21-30"))


def bucket(size):
    return next(name for limit, name in BUCKETS if size <= limit)


def draw(cases, size, rng):
    cells = {}
    for row in cases:
        cells.setdefault((row["source_id"], bucket(len(row["text"]))), []).append(row)
    queues = {cell: rng.sample(rows, len(rows)) for cell, rows in sorted(cells.items())}
    picked = []
    while len(picked) < size and any(queues.values()):
        for cell in sorted(queues):
            if queues[cell] and len(picked) < size:
                picked.append(queues[cell].pop())
    return picked


def main():
    cases_path, out = Path(sys.argv[1]), Path(sys.argv[2])
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 3003
    clean = [json.loads(line) for line in open(cases_path, encoding="utf-8") if not json.loads(line)["noise_ops"]]
    picked = draw(clean, size, random.Random(seed))
    with out.open("w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "cases": len(picked), "of": len(clean), "seed": seed,
                      "by_source": dict(sorted(Counter(row["source_id"] for row in picked).items())),
                      "by_length": dict(sorted(Counter(bucket(len(row["text"])) for row in picked).items()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
