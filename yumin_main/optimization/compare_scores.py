"""Compare saved 96-board solver summaries against the same-set greedy baseline.

The score is exactly 4_score.py's CT + solve_s + 10 * core.  A positive
improvement percentage means the candidate has a lower (better) score.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
from pathlib import Path

from importlib import import_module

score_module = import_module("4_score")

SOLVERS = ("greedy", "yumin_solver", "yumin_solver_2", "yumin_solver_3",
           "yumin_solver_4", "yumin_solver_5")


def summaries():
    yield "simulation_data", Path(".")
    yield "additional_simulation_data", Path("additional_results")
    for path in sorted(Path("robustness_results").glob("*_seed_*")):
        if path.is_dir():
            yield f"robustness_data/{path.name}", path


def board_scores(file: Path):
    with file.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["dataset"]: (Decimal(row["ct"]) + Decimal(row["solve_s"])
                              + 10 * Decimal(row["core"])) for row in rows}


def rows_for_group(name: str, folder: Path):
    files = {solver: folder / f"summary_{solver}.csv" for solver in SOLVERS}
    if not files["greedy"].is_file():
        return [], []
    greedy = score_module.score_one(files["greedy"], Decimal(1), Decimal(10))
    base = greedy["total"]
    greedy_boards = board_scores(files["greedy"])
    rows = []
    board_rows = []
    for solver, file in files.items():
        if not file.is_file():
            continue
        result = score_module.score_one(file, Decimal(1), Decimal(10))
        if result["n"] != greedy["n"]:
            raise ValueError(f"{name}: {solver} has {result['n']} boards; greedy has {greedy['n']}")
        this_boards = board_scores(file)
        if this_boards.keys() != greedy_boards.keys():
            raise ValueError(f"{name}: {solver} has a different board list")
        board_pct = []
        for board, greedy_score in greedy_boards.items():
            score = this_boards[board]
            pct = (greedy_score - score) / greedy_score * 100
            board_pct.append(pct)
            board_rows.append({"dataset_group": name, "dataset": board,
                               "solver": solver, "score": score,
                               "greedy_score": greedy_score,
                               "improvement_pct": pct})
        rows.append({
            "dataset_group": name, "solver": solver, "boards": result["n"],
            "ct": result["ct"], "runtime": result["rt"],
            "cores": result["core"], "score": result["total"],
            "greedy_score": base,
            "improvement_pct": (base - result["total"]) / base * 100,
            "boards_better": sum(pct > 0 for pct in board_pct),
            "worst_board_improvement_pct": min(board_pct),
            "invalid": result["penalty"],
        })
    return rows, board_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="robustness_results/score_comparison.csv")
    parser.add_argument("--board-output", default="robustness_results/board_comparison.csv")
    args = parser.parse_args()
    rows, board_rows = [], []
    for name, folder in summaries():
        group_rows, group_boards = rows_for_group(name, folder)
        rows.extend(group_rows)
        board_rows.extend(group_boards)
    if not rows:
        parser.error("no matching greedy and solver summaries found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    board_output = Path(args.board_output)
    board_output.parent.mkdir(parents=True, exist_ok=True)
    with board_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(board_rows[0]))
        writer.writeheader()
        writer.writerows(board_rows)
    for row in rows:
        print(f"{row['dataset_group']:<45} {row['solver']:<17} "
              f"{row['score']:>10.2f}  {row['improvement_pct']:>+8.2f}%  "
              f"wins={row['boards_better']:>2}/{row['boards']}  "
              f"worst={row['worst_board_improvement_pct']:>+7.2f}%  "
              f"invalid={row['invalid']}")
    print(f"Saved {output} and {board_output}")


if __name__ == "__main__":
    main()
