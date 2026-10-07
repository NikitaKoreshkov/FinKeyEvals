# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 FinKey
"""
CLI: ``python -m finkey_evals run cases.json [options]``

Exit code 0 when the suite passes — CI-ready without plugins.
"""
from __future__ import annotations

import argparse
import json
import sys

from .harness import load_cases_json, run_harness
from .evaluator import ResponseEvaluator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="finkey_evals")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="score a JSON case file")
    run_p.add_argument("cases", help="path to cases.json")
    run_p.add_argument("--min-pass-rate", type=float, default=1.0,
                       help="soft gate: fail only if pass-rate below this (default 1.0)")
    run_p.add_argument("--out", default="", help="write HarnessReport JSON here")
    run_p.add_argument("--quiet", action="store_true", help="print summary line only")

    smoke_p = sub.add_parser("smoke", help="run the built-in smoke suite")
    smoke_p.add_argument("--out", default="")
    smoke_p.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "smoke":
        from .harness import default_smoke_suite
        cases = default_smoke_suite()
    else:
        cases = load_cases_json(args.cases)

    report = run_harness(cases, min_pass_rate=getattr(args, "min_pass_rate", 1.0))
    payload = report.to_dict()

    if getattr(args, "out", ""):
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    if args.quiet:
        print(f"{'PASS' if report.ok else 'FAIL'} {report.passed}/{report.n} "
              f"failed={report.failed} {report.elapsed_ms:.1f}ms")
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        print()
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
