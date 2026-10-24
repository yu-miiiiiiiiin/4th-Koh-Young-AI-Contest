"""Score POMO solver-3 search budgets and policy starts on isolated inputs."""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from optimization.benchmark_candidates import choose_boards
from utils.checker import Checker
from yumin_solver_3 import pomo, solver
from yumin_solver_4 import route

logging.disable(logging.CRITICAL)


def run_one(dataset, config):
    with tempfile.TemporaryDirectory(prefix="aoi_pomo_") as tmp:
        folder = Path(tmp)
        for name in ("input_component.csv", "input_parameter.csv",
                     "input_size.csv", "input_step_region.csv"):
            shutil.copyfile(dataset / name, folder / name)
        old = (solver.SEC_PER_FOV, solver.MAX_BUDGET, solver.MIN_BUDGET,
               pomo.W0.copy(), pomo.N_START)
        solver.SEC_PER_FOV, solver.MAX_BUDGET, solver.MIN_BUDGET = config[:3]
        pomo.W0 = config[3].copy()
        pomo.N_START = config[4]
        try:
            start = time.perf_counter()
            solver.solve(str(folder))
            runtime = time.perf_counter() - start
        finally:
            solver.SEC_PER_FOV, solver.MAX_BUDGET, solver.MIN_BUDGET = old[:3]
            pomo.W0, pomo.N_START = old[3:]
        checker = Checker(str(folder))
        valid = checker.cover_all_components() and checker.fov_inspection_order()
        if not valid:
            checker.penalty_solver()
        ct = checker.compute_ct()
        core = checker._optimal_core
        for handler in list(checker.logger.handlers):
            handler.close()
            checker.logger.removeHandler(handler)
        return ct + 10 * core + runtime, ct, core, runtime, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="additional_simulation_data")
    parser.add_argument("--per-scenario", type=int, default=2)
    parser.add_argument("--output", default="optimization/solver3_candidates.csv")
    args = parser.parse_args()
    old = pomo.W0.copy()
    configs = {
        "v3_original": (0.010, 1.60, 0.05, old, 8),
        "v3_half": (0.005, 0.80, 0.025, old, 8),
        "v3_quarter": (0.0025, 0.40, 0.015, old, 8),
        "v3_v4_policy": (0.005, 0.80, 0.025, route.W0.copy(), 8),
        "v3_start4": (0.005, 0.80, 0.025, old, 4),
    }
    boards = choose_boards(Path(args.data), args.per_scenario)
    rows = []
    for name, config in configs.items():
        total = 0.0
        invalid = 0
        for board in boards:
            score, ct, core, runtime, valid = run_one(board, config)
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
