"""Export and score generated 52Hz benchmark cases."""

import argparse
import difflib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lexicon import Pairs  # noqa: E402
from pinyin import to_keys  # noqa: E402
from preference import favour  # noqa: E402


LENGTH_BUCKETS = ((8, "<=8"), (14, "9-14"), (20, "15-20"), (30, "21-30"), (10 ** 6, ">30"))
# When the model favours the expected spelling at least this strongly, the context did decide after all.
DECIDED = 0.85


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
    """The one noise operation of a case, x2/x3 for several errors at once, or clean."""
    operations = row.get("noise_ops") or []
    if not operations:
        return "clean"
    return operations[0].get("type", "clean") if len(operations) == 1 else f"x{len(operations)}"


def errors_in(row):
    return len(row.get("noise_ops") or [])


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


def differing(gold, text):
    """Where two sentences differ: the expected piece, the other piece, and the sentence with that spot left blank."""
    return [(gold[first:last], text[start:stop], gold[:first] + "＿" + gold[last:]) for tag, first, last, start, stop in
            difflib.SequenceMatcher(None, gold, text, autojunk=False).get_opcodes() if tag == "replace" and text[start:stop]]


def homophone_verdict(case, reply, pairs, judge):
    """For a wrong commit that reads exactly like the expected text: is it real words, and can the context decide?

    Returns "tolerated", "homophone" (real words but the context could have decided, or no judge was given to
    tell), or None when the reply is not a homophone of the expected text or is not made of words.
    """
    gold, text = case["text"], reply["candidates"][0]
    if to_keys(text) != to_keys(gold):
        return None
    spans = differing(gold, text)
    if any(pairs.novel(other, "") for _, other, _ in spans):
        return None
    # A high value means the model, asked about that spot, clearly prefers the expected spelling: the context did
    # decide, and the decoder should have got it right.
    if judge and any(favour(judge, case.get("context", ""), to_keys(own), own, other, frame) >= DECIDED
                     for own, other, frame in spans):
        return "homophone"
    return "tolerated" if judge else "homophone"


class Tally:
    def __init__(self):
        self.counts = Counter()

    def add(self, name):
        self.counts[name] += 1

    def report(self, judged):
        # A tolerated reply is counted once, as the wrong commit it is; the verdicts are extra information.
        total = sum(self.counts[name] for name in ("passed", "wrong_commit", "ask", "error"))
        passed = self.counts["passed"]
        homophone = self.counts["homophone"] + self.counts["tolerated"]
        return {"total": total, "passed": passed, "accuracy": passed / total if total else 0.0,
                "wrong_commit": self.counts["wrong_commit"], "ask": self.counts["ask"], "error": self.counts["error"],
                "homophone_passing": homophone, "tolerated": self.counts["tolerated"] if judged else None}


def score(cases, responses, failures=None, pairs=None, judge=None):
    """Score replies against cases.

    `passed` is the strict one-shot success. A wrong commit that reads exactly like the expected text in real
    words is counted in `homophone_passing`; of those, the ones the model cannot decide from the context when
    asked (needs `judge`, the model endpoint) are counted in `tolerated`. Without `pairs`, the corpus word list,
    neither is computed.
    """
    expected = {row["id"]: row for row in read_rows(cases)}
    seen = set()
    failed = []
    overall = Tally()
    by_operation, by_source, by_length, by_errors = defaultdict(Tally), defaultdict(Tally), defaultdict(Tally), defaultdict(Tally)
    pool = Counter()
    characters = Counter()
    for reply in read_rows(responses):
        case_id = reply["id"]
        if case_id not in expected or case_id in seen:
            raise ValueError(f"unknown or duplicate response id: {case_id}")
        seen.add(case_id)
        case = expected[case_id]
        result = outcome(case, reply)
        verdict = None
        if result == "wrong_commit" and pairs is not None:
            verdict = homophone_verdict(case, reply, pairs, judge)
        tallies = [overall, by_operation[label(case)], by_source[case.get("source_id", "unknown")],
                   by_length[length_bucket(len(case["text"]))], by_errors[errors_in(case)]]
        for tally in tallies:
            tally.add(result)
            if verdict:
                tally.add(verdict)
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
                failures.append({"expected": case, "actual": reply, "verdict": verdict})
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"missing response ids: {len(missing)}")
    groups = lambda grouped: {str(key): value.report(judge) for key, value in sorted(grouped.items())}
    report = {"total": len(seen), "passed": len(seen) - len(failed), "failed": len(failed), "failed_ids": failed,
              "wrong_commits": overall.counts["wrong_commit"], "asks": overall.counts["ask"], "errors": overall.counts["error"],
              "homophone_passing": overall.counts["homophone"] + overall.counts["tolerated"] if pairs is not None else None,
              "tolerated": overall.counts["tolerated"] if judge else None,
              "in_pool": {"total": pool["total"], "passed": pool["passed"], "accuracy": pool["passed"] / pool["total"] if pool["total"] else None},
              "char_error_rate": characters["errors"] / characters["total"] if characters["total"] else None,
              "by_operation": groups(by_operation), "by_source": groups(by_source), "by_errors": groups(by_errors),
              "by_length": {name: by_length[name].report(judge) for name in (bucket for _, bucket in LENGTH_BUCKETS) if name in by_length}}
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
    scoring.add_argument("--lexicon", type=Path, help="52hz-eval 的 make lexicon 生成的词表，用来判断输出是否成词")
    scoring.add_argument("--judge", help="模型服务地址；给出时才判断上下文能否分出同音字")
    args = parser.parse_args()
    try:
        if args.command == "requests":
            for row in requests(args.cases):
                print(json.dumps(row, ensure_ascii=False))
            return 0
        failures = []
        report = score(args.cases, args.responses, failures, Pairs(args.lexicon) if args.lexicon else None, args.judge)
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
