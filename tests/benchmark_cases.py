"""Export and score generated 52Hz benchmark cases."""

import argparse
import json
from collections import Counter
from pathlib import Path


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


def score(cases, responses, failures=None):
    expected = {row["id"]: row for row in read_rows(cases)}
    seen = set()
    failed = []
    by_operation = Counter()
    passed_by_operation = Counter()
    by_source = Counter()
    passed_by_source = Counter()
    for reply in read_rows(responses):
        case_id = reply["id"]
        if case_id not in expected or case_id in seen:
            raise ValueError(f"unknown or duplicate response id: {case_id}")
        seen.add(case_id)
        case = expected[case_id]
        operation = label(case)
        source = case.get("source_id", "unknown")
        by_operation[operation] += 1
        by_source[source] += 1
        candidates = reply.get("candidates")
        passed = reply.get("action") == "commit" and isinstance(candidates, list) and candidates[:1] == [case["text"]]
        if passed:
            passed_by_operation[operation] += 1
            passed_by_source[source] += 1
        else:
            failed.append(case_id)
            if failures is not None:
                failures.append({"expected": case, "actual": reply})
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"missing response ids: {len(missing)}")
    report = {"total": len(seen), "passed": len(seen) - len(failed), "failed": len(failed),
              "failed_ids": failed, "by_operation": {}, "by_source": {}}
    for key, total in sorted(by_operation.items()):
        report["by_operation"][key] = {"total": total, "passed": passed_by_operation[key],
                                       "accuracy": passed_by_operation[key] / total}
    for key, total in sorted(by_source.items()):
        report["by_source"][key] = {"total": total, "passed": passed_by_source[key],
                                    "accuracy": passed_by_source[key] / total}
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
