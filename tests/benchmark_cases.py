"""Export and score generated 52Hz benchmark cases."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


LENGTH_BUCKETS = ((8, "<=8"), (14, "9-14"), (20, "15-20"), (30, "21-30"), (10 ** 6, ">30"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"{path}:{number}: expected an object with string id")
            yield row


def requests(cases):
    for row in read_rows(cases):
        yield {"id": row["id"], "keys": row["keys"], "context": row.get("context", ""),
               "memory": [], "selected": {}, "event": "space"}


def label(row):
    operations = row.get("noise_ops") or []
    return operations[0].get("type", "clean") if operations else "clean"


def length_bucket(size):
    return next(name for limit, name in LENGTH_BUCKETS if size <= limit)


def outcome(case, reply):
    candidates = reply.get("candidates")
    if reply.get("action") == "commit" and isinstance(candidates, list) and candidates[:1] == [case["text"]]:
        return "passed"
    if reply.get("action") == "commit":
        return "wrong_commit"
    if reply.get("action") == "ask":
        return "ask"
    return "error"


def character_errors(gold, actual):
    """Levenshtein distance: insertions, deletions and substitutions each count once."""
    previous = list(range(len(actual) + 1))
    for gold_index, gold_char in enumerate(gold, 1):
        current = [gold_index]
        for actual_index, actual_char in enumerate(actual, 1):
            current.append(min(previous[actual_index] + 1, current[-1] + 1,
                               previous[actual_index - 1] + (gold_char != actual_char)))
        previous = current
    return previous[-1]


class Tally:
    def __init__(self):
        self.counts = Counter()

    def add(self, outcome_name):
        self.counts[outcome_name] += 1

    def report(self):
        total = sum(self.counts.values())
        passed = self.counts["passed"]
        return {"total": total, "passed": passed, "accuracy": passed / total if total else 0.0,
                "wrong_commit": self.counts["wrong_commit"], "ask": self.counts["ask"], "error": self.counts["error"]}


def score(cases, responses, failures=None):
    expected = {row["id"]: row for row in read_rows(cases)}
    seen = set()
    failed = []
    overall = Tally()
    by_operation = defaultdict(Tally)
    by_source = defaultdict(Tally)
    by_length = defaultdict(Tally)
    pool = Counter()
    characters = Counter()
    for reply in read_rows(responses):
        case_id = reply["id"]
        if case_id not in expected or case_id in seen:
            raise ValueError(f"unknown or duplicate response id: {case_id}")
        seen.add(case_id)
        case = expected[case_id]
        result = outcome(case, reply)
        overall.add(result)
        by_operation[label(case)].add(result)
        by_source[case.get("source_id", "unknown")].add(result)
        by_length[length_bucket(len(case["text"]))].add(result)
        trace = reply.get("search")
        if isinstance(trace, dict) and isinstance(trace.get("pool"), list):
            pool["total"] += 1
            pool["passed"] += case["text"] in trace["pool"]
        if result in ("passed", "wrong_commit"):
            characters["total"] += len(case["text"])
            characters["errors"] += character_errors(case["text"], reply["candidates"][0])
        if result != "passed":
            failed.append(case_id)
            if failures is not None:
                failures.append({"expected": case, "actual": reply})
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"missing response ids: {len(missing)}")
    report = {"total": len(seen), "passed": len(seen) - len(failed), "failed": len(failed), "failed_ids": failed,
              "wrong_commits": overall.counts["wrong_commit"], "asks": overall.counts["ask"], "errors": overall.counts["error"],
              "in_pool": {"total": pool["total"], "passed": pool["passed"], "accuracy": pool["passed"] / pool["total"] if pool["total"] else None},
              "char_error_rate": characters["errors"] / characters["total"] if characters["total"] else None,
              "by_operation": {key: value.report() for key, value in sorted(by_operation.items())},
              "by_source": {key: value.report() for key, value in sorted(by_source.items())},
              "by_length": {name: by_length[name].report() for name in (bucket for _, bucket in LENGTH_BUCKETS) if name in by_length}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("requests")
    export.add_argument("cases", type=Path)
    scoring = commands.add_parser("score")
    scoring.add_argument("cases", type=Path)
    scoring.add_argument("responses", type=Path)
    scoring.add_argument("--failures", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "requests":
            for row in requests(args.cases):
                print(json.dumps(row, ensure_ascii=False))
            return 0
        failures = []
        report = score(args.cases, args.responses, failures)
        if args.failures:
            args.failures.parent.mkdir(parents=True, exist_ok=True)
            with args.failures.open("w", encoding="utf-8") as handle:
                for row in failures:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
