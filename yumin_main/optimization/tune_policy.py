"""Offline robust evolution strategy for the POMO rollout's initial weights."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from yumin_solver_4 import cover, route
from yumin_solver_4.solver import Board, build_cover


def choose(root: Path, per_scenario: int):
    grouped = defaultdict(list)
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        grouped["_".join(path.name.split("_")[:-1])].append(path)
    return [path for paths in grouped.values() for path in paths[:per_scenario]]


def make_router(path: Path):
    board = Board(str(path))
    prefix, fovs = build_cover(board, 0.0, lambda: 0.0,
                               np.random.default_rng(20260920))
    return route.Router(board.inst, fovs, prefix)


def greedy_score(router, weights, n_start=8, n_frame=8):
    if router.n == 0:
        return 0.0
    starts = router._starts(n_start)
    best = min(-router.evaluate(seq)[0] for seq in route.serpentine(router))
    for frame in range(min(n_frame, len(router.frames))):
        lock = bool(frame & 1)
        for start in starts:
            seq, _ = router.rollout(weights, frame, start, lock=lock)
            best = min(best, -router.evaluate(seq)[0])
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--per-scenario", type=int, default=1)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--output", default="optimization/policy_fit.json")
    args = parser.parse_args()

    old_tie = cover.TIE_BREAK
    cover.TIE_BREAK = "packed"
    items = []
    try:
        for root_id, root in enumerate(args.roots):
            for path in choose(Path(root), args.per_scenario):
                items.append((root_id, path.name, make_router(path)))
    finally:
        cover.TIE_BREAK = old_tie
    print(f"Loaded {len(items)} balanced boards", flush=True)

    cache = {}

    def raw(weights):
        key = tuple(np.round(weights, 8))
        if key not in cache:
            cache[key] = np.array([greedy_score(router, weights)
                                   for _, _, router in items])
        return cache[key]

    initial = route.W0.copy()
    base = raw(initial)
    roots = np.array([root_id for root_id, _, _ in items])

    def objective(weights):
        scores = raw(weights)
        delta = scores - base
        root_delta = np.array([delta[roots == r].sum() for r in np.unique(roots)])
        # Mean improvement is primary; regressions on any one seed family are
        # penalised so a lucky average cannot become the deployed policy.
        risk = 0.75 * max(0.0, float(root_delta.max())) + 0.15 * float(root_delta.std())
        return float(delta.sum() + risk), scores, root_delta

    rng = np.random.default_rng(args.seed)
    best = initial.copy()
    best_obj, best_scores, best_roots = objective(best)
    sigma = np.array([1.8, 1.0, 0.6, 0.5, 0.8, 0.4, 0.4])
    history = []
    for generation in range(args.generations):
        population = [best]
        for _ in range(args.population):
            trial = best + rng.normal(size=len(best)) * sigma
            trial[0] = np.clip(trial[0], 0.0, 10.0)
            trial[1:] = np.clip(trial[1:], -3.0, 8.0)
            population.append(trial)
        evaluated = [(objective(weights)[0], weights) for weights in population]
        obj, winner = min(evaluated, key=lambda pair: pair[0])
        improved = obj < best_obj - 1e-9
        if improved:
            best = winner.copy()
            best_obj, best_scores, best_roots = objective(best)
            sigma *= 0.96
        else:
            sigma *= 0.82
        entry = {"generation": generation, "objective": best_obj,
                 "sum_delta": float((best_scores - base).sum()),
                 "root_delta": best_roots.tolist(), "weights": best.tolist()}
        history.append(entry)
        print(json.dumps(entry), flush=True)

    result = {"initial": initial.tolist(), "weights": best.tolist(),
              "objective": best_obj, "sum_delta": float((best_scores - base).sum()),
              "root_delta": best_roots.tolist(), "boards": len(items),
              "roots": args.roots, "history": history}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
