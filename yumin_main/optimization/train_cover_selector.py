"""Fit and cross-validate a robust selector for the load-aware cover tie-break."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yumin_solver_5.budget_model import board_features


SOURCES = {
    "practice": ("simulation_data", "optimization/practice_load_cover.csv"),
    "additional": ("additional_simulation_data", "optimization/load_cover_candidates.csv"),
    "train": ("robustness_data/train_seed_31001", "optimization/train_load_cover.csv"),
    "validation": ("robustness_data/validation_seed_41001", "optimization/validation_load_cover.csv"),
}


def load_rows():
    rows = []
    for root_name, (data_root, result_path) in SOURCES.items():
        frame = pd.read_csv(result_path)
        frame = frame[frame["candidate"] == "cover_load_packed"]
        for row in frame.itertuples():
            path = Path(data_root) / row.dataset
            rows.append((root_name, row.dataset, board_features(str(path)), float(row.delta_j)))
    return rows


def fit(x, y, ridge):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    design = np.column_stack((np.ones(len(z)), z))
    reg = np.eye(design.shape[1]) * ridge
    reg[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return mean, std, coef


def predict(x, model):
    mean, std, coef = model
    return np.column_stack((np.ones(len(x)), (x - mean) / std)) @ coef


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="optimization/cover_selector.json")
    args = parser.parse_args()
    rows = load_rows()
    roots = np.array([r[0] for r in rows])
    x = np.array([r[2] for r in rows])
    y = np.array([r[3] for r in rows])

    best = None
    for ridge in (0.1, 1.0, 3.0, 10.0, 30.0, 100.0):
        for threshold in (0.0, 0.05, 0.10, 0.20, 0.40):
            root_savings = []
            selected = 0
            for holdout in SOURCES:
                train = roots != holdout
                test = ~train
                pred = predict(x[test], fit(x[train], y[train], ridge))
                use = pred < -threshold
                selected += int(use.sum())
                root_savings.append(float(y[test][use].sum()))
            # delta<0 is a saving. Penalise the worst held-out seed regression.
            objective = sum(root_savings) + 2.0 * max(0.0, max(root_savings))
            item = (objective, sum(root_savings), max(root_savings), ridge,
                    threshold, selected, root_savings)
            if best is None or item < best:
                best = item

    objective, saving, worst, ridge, threshold, selected, root_savings = best
    model = fit(x, y, ridge)
    result = {"ridge": ridge, "threshold": threshold, "selected": selected,
              "cv_delta": saving, "cv_worst_root_delta": worst,
              "root_deltas": dict(zip(SOURCES, root_savings)),
              "mean": model[0].tolist(), "std": model[1].tolist(),
              "coef": model[2].tolist(), "rows": len(rows)}
    print(json.dumps(result, indent=2))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
