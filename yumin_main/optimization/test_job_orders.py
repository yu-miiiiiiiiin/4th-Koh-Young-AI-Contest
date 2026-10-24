"""Measure alternative within-FOV component orders on saved plans."""
from __future__ import annotations

import argparse
from pathlib import Path

from optimization.analyze_outputs import load_plan


def reorder(values, mode):
    values = sorted(values, reverse=True)
    if mode == "lpt":
        return values
    if mode == "spt":
        return values[::-1]
    if mode == "zigzag":
        out = []
        lo, hi = 0, len(values) - 1
        while lo <= hi:
            out.append(values[lo])
            lo += 1
            if lo <= hi:
                out.append(values[hi])
                hi -= 1
        return out
    if mode == "pair":
        out = []
        for i in range(0, len(values), 2):
            out.extend(reversed(values[i:i + 2]))
        return out
    raise ValueError(mode)


def score_mode(board, plan, mode):
    jobs = [reorder([board.time[i] for i in group], mode)
            for group in plan.members]
    return board.inst.evaluate(plan.x, plan.y, plan.t_img(), jobs, plan.step)[0]


def exact_order_score(board, plan):
    """Unlike Plan.score(), preserve the candidate's within-shot job order."""
    jobs = [[board.time[i] for i in group] for group in plan.members]
    return board.inst.evaluate(plan.x, plan.y, plan.t_img(), jobs, plan.step)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="additional_simulation_data")
    parser.add_argument("--solver", default="yumin_solver_4")
    args = parser.parse_args()
    totals = {mode: 0.0 for mode in ("lpt", "spt", "zigzag", "pair")}
    wins = {mode: 0 for mode in totals}
    adaptive = 0.0
    adaptive_calls = 0
    adaptive_wins = 0
    for dataset in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        board, plan = load_plan(dataset, args.solver)
        results = {}
        for mode in totals:
            results[mode] = score_mode(board, plan, mode)
            totals[mode] += results[mode]
        # Keep LPT except for the closing window; score only a few deterministic
        # alternatives, accepting a reorder only if the exact J improves.
        current = plan.copy()
        current_j = results["lpt"]
        start = max(0, len(plan.members) - min(20, max(8, len(plan.members) // 4)))
        for f in range(len(plan.members) - 1, start - 1, -1):
            if len(current.members[f]) < 3:
                continue
            old = list(current.members[f])
            for mode in ("zigzag", "spt", "pair"):
                # Reorder component IDs with the same time-key pattern.
                pairs = sorted(old, key=lambda i: -board.time[i])
                if mode == "spt":
                    cand = pairs[::-1]
                elif mode == "pair":
                    cand = [v for i in range(0, len(pairs), 2)
                            for v in reversed(pairs[i:i + 2])]
                else:
                    cand = []
                    lo, hi = 0, len(pairs) - 1
                    while lo <= hi:
                        cand.append(pairs[lo]); lo += 1
                        if lo <= hi:
                            cand.append(pairs[hi]); hi -= 1
                current.members[f] = cand
                score = exact_order_score(board, current)
                adaptive_calls += 1
                if score < current_j - 1e-9:
                    current_j = score
                    old = cand
                    adaptive_wins += 1
            current.members[f] = old
        adaptive += current_j
        best = min(results.values())
        for mode, score in results.items():
            wins[mode] += score <= best + 1e-9
    for mode in totals:
        print(f"{mode:<8} J={totals[mode]:.6f} wins={wins[mode]}")
    print(f"adaptive J={adaptive:.6f} gain={totals['lpt']-adaptive:.6f} "
          f"accepted={adaptive_wins}/{adaptive_calls}")


if __name__ == "__main__":
    main()
