"""yumin_solver_5: exact/ML cover followed by solver_4 routing and refinement."""
from __future__ import annotations

import copy
import os
import time

import numpy as np
import pandas as pd

from yumin_solver_4 import solver as v4
from .budget_model import select_bscale
from .exact_cover import improve_cover


_BASE_BUILD_COVER = v4.build_cover
LAST_STATS = {}


def _cover_budget(n_fov, objective='min_fov'):
    # Runtime is charged by the contest.  This is deliberately small; exact
    # small-blob proofs usually finish in milliseconds and LNS takes the rest.
    if objective == 'score':
        return 0.025
    if objective != 'min_fov':
        raise ValueError("objective must be 'min_fov' or 'score'")
    return min(0.22, max(0.045, 0.0018 * int(n_fov)))


def _build_improved_cover(data_folder, objective='min_fov', cover_seconds=None):
    board = v4.Board(data_folder)
    rng = np.random.default_rng(20260921)
    t0 = time.perf_counter()
    prefix, fovs = _BASE_BUILD_COVER(board, 0.0, lambda: time.perf_counter() - t0, rng)
    seconds = _cover_budget(len(fovs), objective) if cover_seconds is None else cover_seconds
    stats = improve_cover(fovs, board, seconds)
    stats['objective'] = objective
    stats['initial_fov'] = len(prefix) + len(fovs) + stats['removed']
    stats['final_cover_fov'] = len(prefix) + len(fovs)
    return prefix, fovs, stats


def plan(data_folder, objective='score', cover_seconds=None, **kwargs):
    """Return solver_4-compatible ``(rows, J)`` with the new cover stage."""
    if objective not in ('score', 'min_fov'):
        raise ValueError("objective must be 'score' or 'min_fov'")
    if objective == 'score' and cover_seconds is None:
        # A/B runs showed that pre-route cover search does not repay its runtime
        # reliably.  Keep solver_4's cover and use the conservative learned
        # route-budget policy; explicit cover_seconds opts back into the
        # experimental cover race.  An explicit bscale always takes priority.
        if 'bscale' not in kwargs:
            board = v4.Board(data_folder)
            kwargs['bscale'] = select_bscale(board, fallback=0.8)
            kwargs['_board'] = board
        rows, score = v4.plan(data_folder, shrink=False, **kwargs)
        LAST_STATS.clear()
        LAST_STATS.update({'objective': 'score', 'score_gate': False,
                           'removed': 0, 'output_fov': len(rows),
                           'score': float(score)})
        return rows, score
    prefix, fovs, stats = _build_improved_cover(data_folder, objective, cover_seconds)
    stats['score_gate'] = True

    # solver_4's downstream route/refine code is reused verbatim.  Supplying a
    # cached cover keeps its own route clock independent from exact-cover time.
    def cached_build(_board, _budget, _clock, _rng):
        return copy.deepcopy(prefix), copy.deepcopy(fovs)

    old = v4.build_cover
    v4.build_cover = cached_build
    try:
        rows, score = v4.plan(data_folder, shrink=False, **kwargs)
    finally:
        v4.build_cover = old

    LAST_STATS.clear()
    LAST_STATS.update(stats)
    LAST_STATS['output_fov'] = len(rows)
    LAST_STATS['score'] = float(score)
    return rows, score


def solve(data_folder: str) -> int:
    # Official submission path: runtime is part of the score, so use the
    # shorter score-aware exact-cover budget by default.
    rows, _ = plan(data_folder, objective='score')
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(data_folder, 'output_fov.csv'), index=True)
    return len(out)
