"""Score solver 1/2 parameter variants on isolated copies of input CSVs."""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from optimization.benchmark_candidates import choose_boards
from utils.checker import Checker
from yumin_solver import solver as v1
from yumin_solver_2 import solver as v2

logging.disable(logging.CRITICAL)


def run_one(dataset: Path, solver, settings):
    with tempfile.TemporaryDirectory(prefix="aoi_bench_") as tmp:
        folder = Path(tmp)
        for name in ("input_component.csv", "input_parameter.csv",
                     "input_size.csv", "input_step_region.csv"):
            shutil.copyfile(dataset / name, folder / name)
        for key, value in settings.items():
            setattr(solver, key, value)
        start = time.perf_counter()
        solver.solve(str(folder))
        runtime = time.perf_counter() - start
        checker = Checker(str(folder))
        cover = checker.cover_all_components()
        order = checker.fov_inspection_order()
        if not cover or not order:
            checker.penalty_solver()
        ct = checker.compute_ct()
        core = int(checker._optimal_core)
        for handler in list(checker.logger.handlers):
            handler.close()
            checker.logger.removeHandler(handler)
        return ct + 10 * core + runtime, ct, core, runtime, cover and order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="additional_simulation_data")
    parser.add_argument("--per-scenario", type=int, default=2)
    parser.add_argument("--output", default="optimization/solver12_candidates.csv")
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Run only these named variants (default: all variants)")
    args = parser.parse_args()
    configs = {
        "v1_original": (v1, {"SHARE_SIDES": False, "SHARE_BARCODE": False, "LPT": False,
                             "GRID_PHASES": 1}),
        "v1_side": (v1, {"SHARE_SIDES": True, "SHARE_BARCODE": False, "LPT": False,
                         "GRID_PHASES": 1}),
        "v1_side_barcode": (v1, {"SHARE_SIDES": True, "SHARE_BARCODE": True, "LPT": False,
                                 "GRID_PHASES": 1}),
        "v1_side_barcode_lpt": (v1, {"SHARE_SIDES": True, "SHARE_BARCODE": True,
                                     "LPT": True, "GRID_PHASES": 1}),
        "v1_grid2": (v1, {"SHARE_SIDES": True, "SHARE_BARCODE": True,
                          "LPT": True, "GRID_PHASES": 2}),
        "v1_grid4": (v1, {"SHARE_SIDES": True, "SHARE_BARCODE": True,
                          "LPT": True, "GRID_PHASES": 4}),
        "v1_grid8": (v1, {"SHARE_SIDES": True, "SHARE_BARCODE": True,
                          "LPT": True, "GRID_PHASES": 8}),
        "v2_original": (v2, {"SPLIT_SIDE": False, "LPT": False}),
        "v2_lpt": (v2, {"SPLIT_SIDE": False, "LPT": True}),
    }
    if args.candidates:
        unknown = set(args.candidates) - set(configs)
        if unknown:
            parser.error(f"unknown candidates: {', '.join(sorted(unknown))}")
        configs = {key: configs[key] for key in args.candidates}
    boards = choose_boards(Path(args.data), args.per_scenario)
    rows = []
    for name, (solver, settings) in configs.items():
        total = 0.0
        invalid = 0
        for board in boards:
            score, ct, core, runtime, valid = run_one(board, solver, settings)
            total += score
            invalid += not valid
            rows.append({"candidate": name, "dataset": board.name, "score": score,
                         "ct": ct, "core": core, "runtime": runtime, "valid": valid})
        print(f"{name}: {total:.2f} invalid={invalid}/{len(boards)}", flush=True)
    with Path(args.output).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
