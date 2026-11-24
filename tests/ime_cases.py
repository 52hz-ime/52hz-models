"""Export blind regression requests or score a provider's JSONL responses."""

import argparse
import json
import random
import sys
from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures" / "ime_cases.jsonl"
NEIGHBORS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def read_rows(path):
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                    raise ValueError("expected an object with a string id")
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            yield row


def noise_variants(syllables, seed):
    keys = "".join(syllables)
    rng = random.Random(seed)
    yield keys
    index = rng.randrange(len(keys))
    yield keys[:index] + keys[index + 1:]
    index = rng.randrange(len(keys))
    yield keys[:index] + keys[index] + keys[index:]
    index = rng.randrange(len(keys))
    row = next(row for row in NEIGHBORS if keys[index] in row)
    column = row.index(keys[index])
    neighbor = rng.choice([row[pos] for pos in (column - 1, column + 1) if 0 <= pos < len(row)])
    yield keys[:index] + neighbor + keys[index + 1:]
    index = rng.choice([pos for pos in range(len(keys) - 1) if keys[pos] != keys[pos + 1]])
    yield keys[:index] + keys[index:index + 2][::-1] + keys[index + 2:]
    index = rng.choice([pos for pos in range(len(syllables) - 1) if syllables[pos] != syllables[pos + 1]])
    yield "".join((*syllables[:index], syllables[index + 1], syllables[index], *syllables[index + 2:]))


def cases():
    yield from read_rows(FIXTURES)
    for index, keys in enumerate(noise_variants(("wan", "fan", "chi", "shen", "me"), seed=52)):
        yield {"id": f"1{index:02}", "request": {"keys": keys, "context": "刚下班，还没订餐。", "event": "space"},
               "expected": {"action": "commit", "text": "晚饭吃什么"}}


def matches(expected, reply):
    if reply.get("action") != expected["action"]:
        return False
    if expected["action"] == "ask":
        options = reply.get("options")
        return (reply.get("focus") == expected["focus"] and isinstance(options, list)
                and all(isinstance(option, str) for option in options)
                and len(options) == len(expected["options"]) and set(options) == set(expected["options"]))
    candidates = reply.get("candidates")
    return (isinstance(candidates, list) and all(isinstance(text, str) for text in candidates)
            and candidates[:1] == [expected["text"]])


def score(replies, failures=None):
    indexed = {case["id"]: case for case in cases()}
    seen = set()
    failed = []
    wrong_commits = []
    pool_cases = pool_hits = winner_hits = 0
    for reply in replies:
        case_id = reply["id"]
        if case_id not in indexed or case_id in seen:
            raise ValueError(f"unknown or duplicate response id: {case_id}")
        seen.add(case_id)
        case = indexed[case_id]
        trace = reply.get('search')
        if isinstance(trace, dict) and 'text' in case['expected']:
            gold = case['expected']['text']
            pool_cases += 1
            if gold in trace['pool']:
                pool_hits += 1
                winner_hits += trace.get('winner') == gold
        if not matches(case["expected"], reply):
            failed.append(case_id)
            if failures is not None:
                failures.append({**case, "actual": reply})
            if reply.get("action") == "commit":
                wrong_commits.append(case_id)
    if missing := indexed.keys() - seen:
        raise ValueError(f"missing response ids: {', '.join(sorted(missing))}")
    report = {"total": len(seen), "passed": len(seen) - len(failed), "failed_ids": sorted(failed),
              "wrong_commit_ids": sorted(wrong_commits)}
    if pool_cases:
        report['sentence_search'] = {'target_sentences': pool_cases, 'in_pool': pool_hits, 'winner_correct': winner_hits}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("requests")
    scoring = commands.add_parser("score")
    scoring.add_argument("responses", type=Path)
    scoring.add_argument("--failures", type=Path, help="write per-case expected and actual results as JSONL")
    scoring.add_argument("--report-only", action="store_true", help="report wrong answers without failing the command")
    args = parser.parse_args()
    if args.command == "requests":
        for case in cases():
            print(json.dumps({"id": case["id"], **case["request"]}, ensure_ascii=False))
        return 0
    try:
        failures = []
        if args.failures and args.failures.resolve() == args.responses.resolve():
            raise ValueError("failures and responses must use different paths")
        report = score(read_rows(args.responses), failures)
        if args.failures:
            with args.failures.open('w', encoding='utf-8') as target:
                for failure in failures:
                    print(json.dumps(failure, ensure_ascii=False), file=target)
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(report, ensure_ascii=False))
    return 1 if report["failed_ids"] and not args.report_only else 0


if __name__ == "__main__":
    sys.exit(main())
