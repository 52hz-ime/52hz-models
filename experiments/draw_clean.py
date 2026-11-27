"""Draw a set of clauses typed without any error, to measure how well the model turns pinyin into characters.

usage: python experiments/draw_clean.py CASES.jsonl OUT.jsonl [SIZE=1000] [SEED=3003] [--exclude USED.jsonl ...]
The clean cases of the benchmark are drawn round-robin over source x length cells, so every source and every
length is represented; nothing is tuned on this set. With --exclude, no case of the listed sets is drawn, so
a configuration chosen on one set can be confirmed on another.
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


def draw(cases, size, rng, exclude=()):
    left_out = set(exclude)
    cases = [row for row in cases if row["id"] not in left_out]
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
    rest = sys.argv[1:]
    used = []
    if "--exclude" in rest:
        at = rest.index("--exclude")
        used = rest[at + 1:]
        rest = rest[:at]
    cases_path, out = Path(rest[0]), Path(rest[1])
    size = int(rest[2]) if len(rest) > 2 else 1000
    seed = int(rest[3]) if len(rest) > 3 else 3003
    clean = [json.loads(line) for line in open(cases_path, encoding="utf-8") if not json.loads(line)["noise_ops"]]
    excluded = {json.loads(line)["id"] for path in used for line in open(path, encoding="utf-8")}
    picked = draw(clean, size, random.Random(seed), excluded)
    if len(picked) < size:
        sys.exit(f"excluding {len(excluded)} cases leaves too few to draw {size}: got {len(picked)}")
    with out.open("w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "cases": len(picked), "of": len(clean), "excluded": len(excluded), "seed": seed,
                      "by_source": dict(sorted(Counter(row["source_id"] for row in picked).items())),
                      "by_length": dict(sorted(Counter(bucket(len(row["text"])) for row in picked).items()))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
