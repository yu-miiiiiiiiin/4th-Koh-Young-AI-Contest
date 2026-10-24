"""Break saved solver results into score floor, excess, and runtime by scenario."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yumin_solver_4 import refine
from yumin_solver_4 import sim
from yumin_solver_4.solver import Board


def load_plan(dataset: Path, solver: str):
    board = Board(str(dataset))
    out = pd.read_csv(dataset / solver / "output_fov.csv")
    members = [list(map(int, json.loads(value))) for value in out["comp_idx"]]
    steps = [max((int(board.step[i]) for i in group), default=0) for group in members]
    blocks = [max((int(board.type[i]) for i in group), default=0) for group in members]
    plan = refine.Plan(
        board.inst,
        out["x"].to_numpy(float),
        out["y"].to_numpy(float),
        steps,
        blocks,
        members,
        board.time,
        board.side,
        board.capture,
        board.side_capture,
    )
    return board, plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="additional_simulation_data")
    parser.add_argument("--solver", default="yumin_solver_4")
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    data = Path(args.data)
    summary = Path(args.summary or f"additional_results/summary_{args.solver}.csv")
    runtime = {}
    with summary.open(newline="") as stream:
        for row in csv.DictReader(stream):
            runtime[row["dataset"]] = float(row["solve_s"])

    groups = {}
    all_rows = []
    for dataset in sorted(path for path in data.iterdir() if path.is_dir()):
        _, plan = load_plan(dataset, args.solver)
        score, core = plan.score()
        floor = refine.profile_floor(plan)
        work = float(sum(plan.c_time[c] for group in plan.members for c in group))
        head = float(plan.t_img()[0]) + plan.inst.t_recon
        work_floor = min(head + work / (k - 3) + sim.BETA * k
                         for k in range(4, plan.inst.max_core + 1))
        capture_chain = float(plan.t_img().sum())
        capture_chain += sim.Z_DELAY * sum(
            plan.step[i] != plan.step[i - 1] for i in range(1, len(plan.step)))
        no_move_floor = min(max(capture_chain, head + work / (k - 3)) + sim.BETA * k
                            for k in range(4, plan.inst.max_core + 1))
        scenario = "_".join(dataset.name.split("_")[:-1])
        row = {
            "dataset": dataset.name,
            "scenario": scenario,
            "fov": len(plan.members),
            "score": score,
            "floor": floor,
            "work_floor": work_floor,
            "no_move_floor": no_move_floor,
            "excess": score - floor,
            "runtime": runtime.get(dataset.name, 0.0),
            "core": core,
        }
        all_rows.append(row)
        groups.setdefault(scenario, []).append(row)

    def report(name, rows):
        score = sum(r["score"] for r in rows)
        floor = sum(r["floor"] for r in rows)
        work_floor = sum(r["work_floor"] for r in rows)
        no_move_floor = sum(r["no_move_floor"] for r in rows)
        runtime_sum = sum(r["runtime"] for r in rows)
        print(f"{name:<22} n={len(rows):>2} J={score:>9.2f} floor={floor:>9.2f} "
              f"workLB={work_floor:>9.2f} noMoveLB={no_move_floor:>9.2f} "
              f"excess={score-floor:>7.2f} RT={runtime_sum:>6.2f} total={score+runtime_sum:>9.2f}")

    for name, rows in groups.items():
        report(name, rows)
    print("-" * 90)
    report("ALL", all_rows)
    print("\nLargest profile excess:")
    for row in sorted(all_rows, key=lambda item: item["excess"], reverse=True)[:15]:
        print(f"{row['dataset']:<25} excess={row['excess']:>7.2f} "
              f"J={row['score']:>7.2f} floor={row['floor']:>7.2f} fov={row['fov']:>3}")


if __name__ == "__main__":
    main()
