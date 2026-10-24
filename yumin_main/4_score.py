"""Score solvers from their summary CSVs.

    Total Score = sum(CT) + sum(RT) * alpha + sum(core) * beta   (alpha=1, beta=10)

Lower is better. Values come from summary_<solver>.csv, written by 2_run.py.

Usage:
    python 4_score.py yumin_solver          # scores summary_yumin_solver.csv
    python 4_score.py yumin greedy ky       # compare several solvers
    python 4_score.py --all                 # every summary_*.csv in the folder
    python 4_score.py summary_greedy.csv    # explicit path also works
"""

import argparse
import csv
from decimal import Decimal, InvalidOperation
from pathlib import Path

import run_utils

ALPHA = Decimal(1)
BETA = Decimal(10)


def calculate_score(path):
    """Sum ct / solve_s / core over a summary CSV, validating every row."""
    totals = {name: Decimal(0) for name in ("ct", "solve_s", "core")}
    count = 0
    flags = {"penalty": 0}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = set(totals) - fields
        if missing:
            raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            for name in totals:
                try:
                    value = Decimal(row[name])
                except (InvalidOperation, TypeError):
                    raise ValueError(
                        f"Line {reader.line_num}: invalid {name}: {row[name]!r}"
                    ) from None
                if not value.is_finite() or value < 0:
                    raise ValueError(f"Line {reader.line_num}: {name} must be finite and nonnegative")
                if name == "core" and value != value.to_integral_value():
                    raise ValueError(f"Line {reader.line_num}: core must be an integer")
                totals[name] += value
            # cover/order are written by 2_run.py after penalty_solver ran,
            # so a False here means components stayed uncovered.
            if {"cover", "order"} <= fields:
                if row["cover"].strip().lower() != "true" or row["order"].strip().lower() != "true":
                    flags["penalty"] += 1
            count += 1
    if not count:
        raise ValueError("CSV contains no data rows")
    return count, totals, flags


def resolve(target):
    """Map a solver name (or a path) to its summary CSV."""
    p = Path(target)
    if p.suffix == ".csv" or p.is_file():
        return p
    candidates = [f"summary_{target}.csv",
                  f"summary_{run_utils.resolve_solver_dir(target)}.csv",
                  f"summary_{target.removesuffix('_solver')}.csv"]
    for name in candidates:
        c = Path(name)
        if c.is_file():
            return c
    raise ValueError(f"No summary CSV for '{target}' (tried: {', '.join(dict.fromkeys(candidates))})")


def score_one(path, alpha, beta):
    count, totals, flags = calculate_score(path)
    total = totals["ct"] + totals["solve_s"] * alpha + totals["core"] * beta
    return {"path": path, "n": count, "ct": totals["ct"], "rt": totals["solve_s"],
            "core": totals["core"], "total": total, "penalty": flags["penalty"]}


def report(r, alpha, beta):
    def line(label, value):
        print(f"{label:<14}: {value}")

    line("File", r["path"])
    line("Datasets", r["n"])
    line("CT sum", f"{r['ct']:,.2f}")
    line(f"RT sum x{alpha:g}", f"{r['rt'] * alpha:,.2f}")
    line("Core sum", f"{r['core']:,.0f}")
    line(f"Core cost x{beta:g}", f"{r['core'] * beta:,.2f}")
    line("TOTAL SCORE", f"{r['total']:,.2f}")
    if r["penalty"]:
        line("WARNING", f"{r['penalty']} dataset(s) still violate cover/order after penalty_solver")
    if r["rt"] == 0:
        line("NOTE", "RT sum is 0 - solve_s is rounded to 2 decimals by 2_run.py,"
                     " and --eval runs record 0.")


def main():
    parser = argparse.ArgumentParser(
        description="Total Score = sum(CT) + sum(RT)*alpha + sum(core)*beta. Lower is better.")
    parser.add_argument("targets", nargs="*", metavar="SOLVER",
                        help="solver name (e.g. yumin_solver, greedy) or a summary CSV path")
    parser.add_argument("--all", action="store_true", help="score every summary_*.csv here")
    parser.add_argument("--alpha", type=Decimal, default=ALPHA, help="RT weight (default: 1)")
    parser.add_argument("--beta", type=Decimal, default=BETA, help="core weight (default: 10)")
    args = parser.parse_args()

    targets = list(args.targets)
    if args.all:
        targets += [str(p) for p in sorted(Path(".").glob("summary_*.csv"))]
    if not targets:
        parser.error("give at least one solver name, or --all")

    results = []
    for target in dict.fromkeys(targets):
        try:
            results.append(score_one(resolve(target), args.alpha, args.beta))
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            parser.exit(1, f"Error: {exc}\n")

    for i, r in enumerate(results):
        if i:
            print()
        report(r, args.alpha, args.beta)

    if len(results) > 1:
        best = min(r["total"] for r in results)
        print(f"\n{'solver':<24} {'CT':>12} {'RT':>8} {'core':>7} {'TOTAL':>12} {'vs best':>10}")
        print("-" * 77)
        for r in sorted(results, key=lambda r: r["total"]):
            name = r["path"].stem.removeprefix("summary_")
            print(f"{name:<24} {r['ct']:>12,.2f} {r['rt'] * args.alpha:>8,.2f} "
                  f"{r['core'] * args.beta:>7,.0f} {r['total']:>12,.2f} {r['total'] - best:>+10,.2f}")

    print("\nLower is better. Based on the values stored in the CSV.")


if __name__ == "__main__":
    main()
