"""Paper-inspired solver_6: feasible seed -> rolling schedule -> Laguerre QP.

Public solve(folder) follows the competition interface. Existing solver_4
provides cover construction and the seed plan; no globals are monkey-patched.
"""
import json
import os
import time
import numpy as np
import pandas as pd
from yumin_solver_4 import solver as base, refine
from .planner import qp_positions, rolling_order, schedule_bound, membership_bound

LAST_STATS = {}


def improve_rows(board, rows, *, seconds=0.03, do_route=True, do_qp=True,
                 margin=1e-7, poles=(0.0, 0.5, 0.85), basis_mode='laguerre'):
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative')
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('margin must be finite and nonnegative')
    if basis_mode not in ('laguerre', 'full'):
        raise ValueError('basis_mode must be laguerre or full')
    if not poles or any(not np.isfinite(p) or not 0 <= p < 1 for p in poles):
        raise ValueError('poles must be a nonempty sequence in [0, 1)')
    started = time.perf_counter()
    members = [json.loads(r['comp_idx']) for r in rows]
    p = refine.Plan(board.inst, [r['x'] for r in rows], [r['y'] for r in rows],
                    [int(board.step[ms[0]]) for ms in members],
                    [max(int(board.type[c]) for c in ms) for ms in members],
                    members, board.time, board.side, board.capture, board.side_capture)
    initial_j = p.score()[0]
    stats = dict(initial_j=initial_j, route_candidates=0, bound_pruned=0,
                 route_accepted=0, qp_solved=0, qp_accepted=0, qp_max_gap=0.0,
                 qp_max_violation=0.0, accepted_poles=[])
    prefix = sum(block == 2 for block in p.block)
    room = max(0.0, initial_j - membership_bound(p))
    stats['membership_room'] = room
    stats['budget_skipped'] = room <= 0.02
    seconds = min(seconds, 0.4 * room)
    if seconds > 0 and rows and not stats['budget_skipped']:
        if do_route:
            allowance = seconds * (0.6 if do_qp else 1.0)
            p = rolling_order(p, prefix, started + allowance, stats)
        if do_qp:
            p = qp_positions(p, board, started + seconds, stats, margin=margin,
                             poles=poles, basis_mode=basis_mode)
    final_j = p.score()[0]
    stats.update(final_j=final_j, gain=initial_j-final_j,
                 fixed_plan_lower_bound=schedule_bound(p),
                 refinement_s=time.perf_counter()-started)
    result = [{'x': float(p.x[i]), 'y': float(p.y[i]),
               'comp_idx': json.dumps(sorted((int(c) for c in ms), key=lambda c: -board.time[c]))}
              for i, ms in enumerate(p.members)]
    return result, final_j, stats


def plan(data_folder, *, seconds=0.03, do_route=True, do_qp=True,
         margin=1e-7, poles=(0.0, 0.5, 0.85), basis_mode='laguerre',
         bscale=0.8):
    board = base.Board(data_folder)
    if board.n == 0:
        LAST_STATS.clear()
        return [], 0.0
    rows, _ = base.plan(data_folder, _board=board, bscale=bscale, shrink=False)
    rows, j, stats = improve_rows(board, rows, seconds=seconds, do_route=do_route,
                                 do_qp=do_qp, margin=margin, poles=poles,
                                 basis_mode=basis_mode)
    LAST_STATS.clear()
    LAST_STATS.update(stats)
    return rows, j


def solve(data_folder: str) -> int:
    rows, _ = plan(data_folder)
    pd.DataFrame(rows, columns=['x', 'y', 'comp_idx']).to_csv(
        os.path.join(data_folder, 'output_fov.csv'), index=True)
    return len(rows)
