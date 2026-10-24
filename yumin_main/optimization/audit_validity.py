"""Re-check saved FOV plans *before* the runner's optional penalty repair.

2_run.py records cover/order after penalty_solver, so a summary alone cannot
prove that the submitted plan was originally valid.  This audits saved plans
with the contest Checker and leaves the input and score summaries unchanged.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import run_utils
from optimization.compare_scores import SOLVERS, summaries
from utils.checker import Checker

logging.disable(logging.CRITICAL)


def main():
    report = []
    for dataset_root, result_dir in summaries():
        root = Path(dataset_root)
        if not root.is_dir():
            continue
        boards = sorted(path for path in root.iterdir() if path.is_dir())
        for solver in SOLVERS:
            if not (result_dir / f"summary_{solver}.csv").is_file():
                continue
            failures = []
            for board in boards:
                if (board / "output_fov.csv").exists():
                    raise RuntimeError(f"refusing to stage over {board / 'output_fov.csv'}")
                with run_utils.stage(str(board), run_utils.resolve_solver_dir(solver)):
                    checker = Checker(str(board))
                    try:
                        cover = checker.cover_all_components()
                        order = checker.fov_inspection_order()
                    finally:
                        run_utils.release_logger(checker)
                if not (cover and order):
                    failures.append(board.name)
            row = {"dataset_group": dataset_root, "solver": solver,
                   "boards": len(boards), "valid": len(boards) - len(failures),
                   "invalid": len(failures), "invalid_boards": ";".join(failures)}
            report.append(row)
            print(f"{dataset_root:<45} {solver:<17} "
                  f"{row['valid']}/{row['boards']} valid", flush=True)
    output = Path("robustness_results/validity_audit.csv")
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
