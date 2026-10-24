"""Benchmark solver-4/5 parameter candidates without writing solver outputs."""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from optimization.analyze_outputs import load_plan
from yumin_solver_4 import route
from yumin_solver_4 import solver as v4
from yumin_solver_4 import cover as cover4
from yumin_solver_5 import solver as v5

ROBUST_W = [6.196975010532142, 2.8408622639037344, 2.578401766121734,
            -0.2861068019226913, 0.14497556151237934,
            0.6157730953084132, -0.6453031857083551]


def with_mixes(path, mixes, **kwargs):
    old = v4.MIXES
    v4.MIXES = mixes
    try:
        return v4.plan(path, **kwargs)
    finally:
        v4.MIXES = old


def with_cover_tie(path, tie, **kwargs):
    old = cover4.TIE_BREAK
    cover4.TIE_BREAK = tie
    try:
        return v4.plan(path, **kwargs)
    finally:
        cover4.TIE_BREAK = old


def with_policy(path, weights=ROBUST_W, tie=None, **kwargs):
    old_w = route.W0
    old_tie = cover4.TIE_BREAK
    route.W0 = np.asarray(weights, dtype=float)
    if tie is not None:
        cover4.TIE_BREAK = tie
    try:
        return v4.plan(path, **kwargs)
    finally:
        route.W0 = old_w
        cover4.TIE_BREAK = old_tie


def with_relax_passes(path, passes, **kwargs):
    old = route.RELAX_PASSES
    route.RELAX_PASSES = passes
    try:
        return v4.plan(path, **kwargs)
    finally:
        route.RELAX_PASSES = old


def candidates():
    return {
        "v4_b050": lambda p: v4.plan(p, bscale=0.50),
        "v4_b080": lambda p: v4.plan(p, bscale=0.80),
        "v4_b100": lambda p: v4.plan(p, bscale=1.00),
        "v4_b125": lambda p: v4.plan(p, bscale=1.25),
        "v4_stall2": lambda p: v4.plan(p, stall=2),
        "v4_stall6": lambda p: v4.plan(p, stall=6),
        "v4_start4": lambda p: v4.plan(p, n_start=4),
        "v4_start12": lambda p: v4.plan(p, n_start=12),
        "v4_frame4": lambda p: v4.plan(p, n_frame=4),
        "v4_no_shape": lambda p: v4.plan(p, do_shape=False),
        "v4_no_flex": lambda p: v4.plan(p, do_flexible=False),
        "v4_round0": lambda p: v4.plan(p, rounds=0),
        "v4_round2": lambda p: v4.plan(p, rounds=2),
        "v4_hier": lambda p: v4.plan(p, strategy="hier"),
        "v4_pomo": lambda p: v4.plan(p, strategy="pomo"),
        "v4_refine": lambda p: v4.plan(p, strategy="refine"),
        "v4_ils": lambda p: v4.plan(p, do_ils=True),
        "v4_shrink": lambda p: v4.plan(p, shrink=True),
        "v4_mix025": lambda p: with_mixes(p, ((1.0, 0.0), (1.0, 0.25))),
        "v4_mixwide": lambda p: with_mixes(
            p, ((1.0, 0.0), (1.0, 0.2), (1.0, 0.5), (1.0, 1.0))),
        "v4_relax_assign": lambda p: v4.plan(p, relax_assign=True),
        "v4_relax_both": lambda p: v4.plan(
            p, relax_assign=True, relax_dissolve=True),
        "v4_flex_assign": lambda p: v4.plan(p, do_flexible_assign=True),
        "v4_all_assign": lambda p: v4.plan(
            p, do_flexible_assign=True, relax_assign=True, relax_dissolve=True),
        "cover_slack": lambda p: with_cover_tie(p, "slack"),
        "cover_area": lambda p: with_cover_tie(p, "area"),
        "cover_balance": lambda p: with_cover_tie(p, "balance"),
        "cover_packed": lambda p: with_cover_tie(p, "packed"),
        "cover_compact": lambda p: with_cover_tie(p, "compact"),
        "cover_load_balance": lambda p: with_cover_tie(p, "load_balance"),
        "cover_load_packed": lambda p: with_cover_tie(p, "load_packed"),
        "cover_packed_relax": lambda p: with_cover_tie(
            p, "packed", relax_assign=True),
        "policy_robust": lambda p: with_policy(p),
        "policy_packed": lambda p: with_policy(p, tie="packed"),
        "policy_packed_relax": lambda p: with_policy(
            p, tie="packed", relax_assign=True),
        "route_relax1": lambda p: with_relax_passes(p, 1),
        "route_relax4": lambda p: with_relax_passes(p, 4),
        "route_relax8": lambda p: with_relax_passes(p, 8),
        "flex_cap010": lambda p: v4.plan(p, flex_pass_cap=0.10),
        "flex_cap020": lambda p: v4.plan(p, flex_pass_cap=0.20),
        "flex_gate0": lambda p: v4.plan(
            p, flex_pass_cap=0.10, flex_dissolve_gate=0.0, flex_tail_gate=0.0),
        "exact_0025": lambda p: v5.plan(p, objective="score", cover_seconds=0.025),
        "exact_0050": lambda p: v5.plan(p, objective="score", cover_seconds=0.050),
        "exact_0100": lambda p: v5.plan(p, objective="score", cover_seconds=0.100),
        "exact_0500": lambda p: v5.plan(p, objective="score", cover_seconds=0.500),
        "exact_1000": lambda p: v5.plan(p, objective="score", cover_seconds=1.000),
        "exact_minfov": lambda p: v5.plan(p, objective="min_fov"),
    }


def choose_boards(root: Path, per_scenario: int):
    grouped = defaultdict(list)
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        grouped["_".join(path.name.split("_")[:-1])].append(path)
    return [path for group in grouped.values() for path in group[:per_scenario]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="additional_simulation_data")
    parser.add_argument("--baseline-solver", default="yumin_solver_4")
    parser.add_argument("--per-scenario", type=int, default=2)
    parser.add_argument("--candidates", nargs="*", default=[])
    parser.add_argument("--output", default="optimization/candidate_results.csv")
    args = parser.parse_args()

    all_candidates = candidates()
    names = args.candidates or list(all_candidates)
    unknown = sorted(set(names) - set(all_candidates))
    if unknown:
        parser.error(f"unknown candidates: {', '.join(unknown)}")

    boards = choose_boards(Path(args.data), args.per_scenario)
    rows = []
    baseline = {}
    for board_path in boards:
        _, plan = load_plan(board_path, args.baseline_solver)
        baseline[board_path.name] = plan.score()[0]

    for name in names:
        total_j = 0.0
        total_runtime = 0.0
        total_base = 0.0
        wins = 0
        print(f"[{name}]", flush=True)
        for board_path in boards:
            start = time.perf_counter()
            _, score = all_candidates[name](str(board_path))
            elapsed = time.perf_counter() - start
            base = baseline[board_path.name]
            total_j += score
            total_runtime += elapsed
            total_base += base
            wins += score < base - 1e-9
            rows.append({"candidate": name, "dataset": board_path.name,
                         "j": score, "runtime": elapsed, "total": score + elapsed,
                         "baseline_j": base, "delta_j": score - base})
        print(f"  J={total_j:.2f} RT={total_runtime:.2f} total={total_j+total_runtime:.2f} "
              f"deltaJ={total_j-total_base:+.2f} wins={wins}/{len(boards)}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
