"""Model (4): scenario-aware dual-lattice cover (Hochbaum-Maass shifting, done right).

A single regular lattice loses to the greedy because side and top components
have feasible-centre rectangles that differ by a factor of two in each axis --
one pitch cannot fit both.  This model therefore uses *two* lattices:

    fine lattice   pitch <= min width of the side components' R  (they are the
                   binding constraint) -- these FOVs also pick up every top
                   component that happens to fall in their full field, for free
    coarse lattice pitch <= min width of the remaining top components' R

Each lattice pitch guarantees that any rectangle at least that wide contains a
lattice point, so coverage is structurally guaranteed.  The origin of both
lattices is swept (the shifting step) and the best offset kept -- which is the
Hochbaum-Maass (1+1/l)^2 construction specialised to this two-class geometry.

Boards whose components sit on a regular grid (scenarios S04/S05) are exactly
where this beats a myopic greedy; elsewhere the greedy plan is kept.
"""
import os
import time

import numpy as np

from . import solvercore as sc
from . import core


def _env(n, d, c=float):
    v = os.environ.get('LT_' + n)
    return c(v) if v is not None else d


BUDGET = _env('BUDGET', 1.10)
SHIFTS = _env('SHIFTS', 4, int)
PCT = _env('PCT', 3.0)


def _lattice_cover(P, must, bonus, ofx, ofy, pct):
    """Pierce every R_i of `must` with a regular lattice; `bonus` rides along."""
    if len(must) == 0:
        return [], np.zeros(0, dtype=np.int64)
    idx = np.concatenate([must, bonus]) if len(bonus) else must
    nm = len(must)
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    px = max(float(np.percentile(x2[:nm] - x1[:nm], pct)), 1e-3)
    py = max(float(np.percentile(y2[:nm] - y1[:nm], pct)), 1e-3)
    gx = np.arange(float(x1.min()) + ofx * px, float(x2.max()) + px, px)
    gy = np.arange(float(y1.min()) + ofy * py, float(y2.max()) + py, py)
    if len(gx) * len(gy) > 30000 or len(gx) == 0 or len(gy) == 0:
        return [], np.zeros(0, dtype=np.int64)

    okx = (gx[:, None] >= x1[None, :] - sc.EPS) & (gx[:, None] <= x2[None, :] + sc.EPS)
    oky = (gy[:, None] >= y1[None, :] - sc.EPS) & (gy[:, None] <= y2[None, :] + sc.EPS)
    cov = (okx[:, None, :] & oky[None, :, :]).reshape(len(gx) * len(gy), len(idx))
    pts = np.stack(np.meshgrid(gx, gy, indexing='ij'), -1).reshape(-1, 2)
    live = cov.any(1)
    cov, pts = cov[live], pts[live]
    if len(pts) == 0:
        return [], np.zeros(0, dtype=np.int64)

    wgt = np.empty(len(idx))
    wgt[:nm] = 1.0
    wgt[nm:] = 0.05
    rem = np.ones(len(idx), dtype=bool)
    fovs, covered = [], []
    gain = cov @ wgt
    while rem[:nm].any():
        b = int(np.argmax(gain))
        take = np.where(cov[b] & rem)[0]
        if len(take) == 0 or not rem[:nm][cov[b][:nm]].any():
            break
        fovs.append(sc.make_fov(P, idx, take, pts[b][0], pts[b][1]))
        covered.extend(int(v) for v in idx[take])
        gain -= cov[:, take] @ wgt[take]
        rem[take] = False
    return fovs, np.array(covered, dtype=np.int64)


def _build_lattice(P, ofx, ofy):
    fovs, done = sc.base_plan(P)
    for st in (0, 1):
        base = np.where(~P.big & (P.typ != 2) & (P.step == st) & ~done)[0]
        if len(base) == 0:
            continue
        side = base[P.side[base] == 1]
        top = base[P.side[base] == 0]
        if len(side):                                  # fine lattice
            fs, cov = _lattice_cover(P, side, top, ofx, ofy, PCT)
            fovs.extend(fs)
            done[cov] = True
            top = top[~done[top]]
            side = side[~done[side]]
            if len(side):                              # anything the lattice missed
                fs, cov = core._cover(P, side, top, core.P_BONUS, time.perf_counter() + 5)
                fovs.extend(fs)
                done[cov] = True
                top = top[~done[top]]
        if len(top):                                   # coarse lattice
            fs, cov = _lattice_cover(P, top, np.zeros(0, dtype=np.int64), ofx, ofy, PCT)
            fovs.extend(fs)
            done[cov] = True
            top = top[~done[top]]
            if len(top):
                fs, cov = core._cover(P, top, np.zeros(0, dtype=np.int64), 0.0,
                                      time.perf_counter() + 5)
                fovs.extend(fs)
                done[cov] = True
    return fovs


def solve(data_folder: str) -> int:
    t0 = time.perf_counter()
    P = sc.Problem(data_folder)
    deadline = t0 + BUDGET
    rng = np.random.default_rng(17)

    plans = [core._finalise(P, core._build(P, core._groups_global, deadline))]
    offs = [(i / SHIFTS, j / SHIFTS) for i in range(SHIFTS) for j in range(SHIFTS)]
    for ofx, ofy in offs[:SHIFTS]:
        if time.perf_counter() > t0 + 0.55 * BUDGET:
            break
        plans.append(core._finalise(P, _build_lattice(P, ofx, ofy)))
    plan = min((p for p in plans if p), key=lambda p: core._cap_of(P, p))
    return sc.finish(P, plan, deadline, rng, data_folder)
