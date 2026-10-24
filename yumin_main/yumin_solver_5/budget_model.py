"""Small offline-trained model for score-aware route-budget selection.

The deployed model is deliberately conservative.  A four-arm ridge regressor
looked strong in-sample but its sub-point labels were dominated by wall-clock
noise.  The current model clusters boards by stable geometry/workload features
and changes the validated 0.8 budget only for a cluster whose aggregate gain
also reproduced on fresh generator seeds.
"""
from __future__ import annotations

import os

import numpy as np

from yumin_solver_4.solver import Board


MODEL_PATH = os.path.join(os.path.dirname(__file__), 'budget_model.npz')
SCALES = np.array([0.55, 0.70, 0.80, 0.95], dtype=float)
_CACHE = None


def board_features(data_folder):
    # Accepting a Board avoids reading the same three CSV files again when the
    # score wrapper passes the parsed instance on to solver_4.
    b = data_folder if isinstance(data_folder, Board) else Board(data_folder)
    width = max(float(b.br_x.max() - b.tl_x.min()), 1e-9)
    height = max(float(b.br_y.max() - b.tl_y.min()), 1e-9)
    area = (b.br_x - b.tl_x) * (b.br_y - b.tl_y)
    board_cells = width * height / max(b.fov_w * b.fov_h, 1e-9)
    return np.array([
        math_log1p(b.n),
        math_log1p(board_cells),
        float(np.mean(b.side)),
        float(np.mean(b.big)),
        float(np.mean(b.step != 0)),
        float(np.mean(area)),
        float(np.std(area)),
        float(np.mean(b.time)),
        float(np.std(b.time)),
        float(b.inst.t_recon),
        float(b.inst.max_core),
        float(b.fov_w / max(b.fov_h, 1e-9)),
    ], dtype=float)


def math_log1p(value):
    return float(np.log1p(float(value)))


def _load():
    global _CACHE
    if _CACHE is None and os.path.isfile(MODEL_PATH):
        with np.load(MODEL_PATH) as data:
            _CACHE = {k: data[k] for k in data.files}
    return _CACHE


def select_bscale(data_folder, fallback=0.8):
    model = _load()
    if model is None:
        return float(fallback)
    x = board_features(data_folder)
    kind = str(model.get('kind', np.array(['ridge']))[0])
    z = (x - model['mean']) / model['std']
    if kind == 'cluster':
        feature_idx = model['feature_idx'].astype(int)
        point = z[feature_idx]
        cluster = int(np.argmin(np.sum(
            (model['centers'] - point[None, :]) ** 2, axis=1)))
        return float(model['cluster_scales'][cluster])

    # Compatibility with the first experimental ridge model.
    z = np.concatenate(([1.0], z, z * z))
    predicted = z @ model['coef']
    base = int(np.argmin(np.abs(model['scales'] - fallback)))
    winner = int(np.argmin(predicted))
    if predicted[base] - predicted[winner] < float(model['margin'][0]):
        winner = base
    return float(model['scales'][winner])


def reset_cache():
    global _CACHE
    _CACHE = None
