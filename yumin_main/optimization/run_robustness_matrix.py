"""Fill in missing same-dataset greedy/solver evaluations and refresh changed solvers.

Run from the repository root: PYTHONDONTWRITEBYTECODE=1 python
optimization/run_robustness_matrix.py.  Saved plans and summaries remain in
their normal dataset and result directories, never in solver source folders.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from importlib import import_module
from pathlib import Path

from optimization.compare_scores import SOLVERS, summaries

score_module = import_module("4_score")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=SOLVERS)
    parser.add_argument("--refresh-all", action="store_true",
                        help="rerun the selected solvers even with a valid summary")
    args = parser.parse_args()
    roots = list(summaries())
    for dataset, result_dir in roots:
        if not Path(dataset).is_dir():
            continue
        result_dir.mkdir(parents=True, exist_ok=True)
        for solver in args.solvers:
            summary = result_dir / f"summary_{solver}.csv"
            # These three changed since the earlier practice/additional runs.
            refresh = args.refresh_all or (
                dataset in ("simulation_data", "additional_simulation_data")
                and solver in ("yumin_solver", "yumin_solver_2", "yumin_solver_3"))
            if summary.is_file() and not refresh:
                result = score_module.score_one(summary, 1, 10)
                if result["n"] == 96 and result["penalty"] == 0:
                    print(f"SKIP {dataset:<44} {solver:<17} {result['total']:>10.2f}",
                          flush=True)
                    continue
            proc = subprocess.run(
                [sys.executable, "2_run.py", "--data", dataset, "--solver", solver,
                 "--summary-dir", str(result_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                check=False,
            )
            if proc.returncode:
                print(proc.stdout[-5000:], file=sys.stderr)
                raise RuntimeError(f"{dataset}: {solver} failed with code {proc.returncode}")
            result = score_module.score_one(summary, 1, 10)
            if result["n"] != 96 or result["penalty"]:
                raise RuntimeError(f"{dataset}: {solver} has an invalid board or count")
            print(f"DONE {dataset:<44} {solver:<17} {result['total']:>10.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
