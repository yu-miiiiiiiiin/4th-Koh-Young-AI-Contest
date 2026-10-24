def _seq_bound(P, ev, ids, k):
    """Order-independent lower bound on CT for this FOV set.

    Core 0 has to capture every FOV and travel a Hamiltonian path between them;
    a path visits every FOV once, so its length is at least the sum of each
    FOV's cheapest outgoing hop less the single largest of those (the endpoint
    has none).  Every step level present in a type block also forces at least
    one 5 s transition.  Independently, the inspection cores cannot finish
    before the first capture, its reconstruction and total work over k-3.  No
    re-ordering beats the larger of the two, so once CT reaches it the search on
    this dataset is finished and its remaining budget is pure runtime cost."""
    a = np.asarray(ids, dtype=np.int64)
    n = len(a)
    cap = float(ev.timgf[a].sum())
    hop = 0.0
    if n > 1:
        M = ev.mvf[np.ix_(a, a)].copy()
        np.fill_diagonal(M, np.inf)
        mins = M.min(1)
        hop = float(mins.sum() - mins.max())
    zt = 0.0
    ty = ev._ftyp if hasattr(ev, '_ftyp') else None
    if ty is not None:
        for t in np.unique(ty):
            ns = len(np.unique(ev.stpi[a[ty[a] == t]]))
            zt += Z_PENALTY * max(0, ns - 1)
    work = float(ev.load.sum())
    tail = P.recon
    first = float(ev.timgf[a[0]]) + P.recon if n else 0.0
    return max(cap + hop + zt + tail, work / max(k - 3, 1) + first)
"""
ICCAS 2026 Koh Young AI Contest — FOV planning solver.

Pipeline
    0. preprocess          : feasible-centre rectangles R_i, big-component split,
                             (step x side) decomposition
    1. construction        : weighted geometric set cover over a discretised
                             candidate grid (integral-image max-coverage),
                             run inside Hochbaum-Maass shifted blocks and
                             globally -> a portfolio of plans
    2. local search        : box-intersection merge + redundant-FOV drop
    3. sequencing          : type/step block order (minimises 5 s Z-transitions)
                             + serpentine / NN+2-opt inside each block
    4. pipeline post-pass  : LPT ordering of comp_idx
    5. evaluation          : exact re-implementation of Checker.get_real_cost,
                             scored as min_k (CT + 10k) -- identical to grading
"""
import os
import csv
import functools
import json
import math
import time

import numpy as np

try:
    from . import fastops
    _FAST = fastops.HAVE_NUMBA
except Exception:            # pragma: no cover
    fastops = None
    _FAST = False

# checker allows 1e-5 slack; stay well inside it
EPS = 1e-9
Z_PENALTY = 5.0
BETA_CORE = 10.0

# ------------------------------------------------------------------ parameters
def _env(name, default, cast=float):
    v = os.environ.get('HM_' + name)
    return cast(v) if v is not None else default


P_GRID_CELLS = _env('P_GRID_CELLS', 300, int)      # max candidate-grid resolution per axis
P_GRID_MIN = _env('P_GRID_MIN', 0.30, float)       # mm, finest candidate spacing
P_BONUS = _env('P_BONUS', 0.40, float)          # weight of a "free ride" component during the side pass
P_HM_L = _env('P_HM_L', 2, int)              # Hochbaum-Maass block size multiplier (block = L*W x L*H)
P_GROW_PASSES = _env('P_GROW_PASSES', 2, int)       # box-growing rounds after each max-coverage pick
P_MERGE_SWEEPS = _env('P_MERGE_SWEEPS', 24, int)
P_TIME_BUDGET = _env('P_TIME_BUDGET', 0.90, float)    # s, soft wall-clock cap per dataset
P_PORTFOLIO = _env('P_PORTFOLIO', 0, int)   # 1 = also try HM block / lattice constructions
P_CONSTRUCT_FRAC = _env('P_CONSTRUCT_FRAC', 0.35, float) # fraction of the budget spent on constructions
P_LNS_END = _env('P_LNS_END', 0.45, float)        # LNS runs until this fraction of the budget

# ---- improvement switches (evaluated one at a time, see exp/results.csv) ----
IMP_A = _env('IMP_A', 0, int)   # adaptive budget: give LNS time only when core0-bound
IMP_B = _env('IMP_B', 0, int)   # component re-assignment between FOVs (tail shortening)
IMP_C = _env('IMP_C', 1, int)   # extreme-point candidate cover (exact, incremental gains)
P_FAST_N = _env('P_FAST_N', 900, int)
P_LNS_DEV = _env('P_LNS_DEV', 0.55, float) # record-to-record acceptance deviation
P_LNS_RESET = _env('P_LNS_RESET', 40, int)
P_LNS_STOP = _env('P_LNS_STOP', 150, int)
P_KREFRESH = _env('P_KREFRESH', 1, int)  # re-derive k* every N accepted polish moves  # give up after this many non-improving iterations  # non-improving iterations before restart
# Measured on all 96: five modes beat thirteen, because mode evaluation eats
# into the polish budget and polish is worth more per second at the margin.
P_SO_MODE = os.environ.get('HM_P_SO_MODE', 'global')
P_MODES = os.environ.get('HM_P_MODES',
                         'serp_h,serp_v,nn,band2,band3,band4,bandnn2,'
                         'serp_h2').split(',')  # below this many rectangles use the exact-candidate cover
IMP_D = _env('IMP_D', 0, int)   # joint cost-aware single-pass cover variant
IMP_E = _env('IMP_E', 1, int)   # stronger LNS (randomised repair, acceptance, operators)
IMP_F = _env('IMP_F', 0, int)   # k-exchange local search (2->1 FOV)
IMP_G = _env('IMP_G', 1, int)   # big-component tile sliding
IMP_H = _env('IMP_H', 0, int)   # speed-ups (incremental integral image, faster merge)
IMP_I = _env('IMP_I', 0, int)   # iterated local search on the visit order
IMP_K = _env('IMP_K', 0, int)   # polish the best two candidate sequences, not just one
IMP_O = _env('IMP_O', 1, int)   # Or-opt segment moves during the order polish
IMP_N = _env('IMP_N', 0, int)   # local-maxima candidate pool + exact selection
P_POOL_CAP = _env('P_POOL_CAP', 4000, int)
P_POOL_GRID = _env('P_POOL_GRID', 420, int)
IMP_M = _env('IMP_M', 0, int)   # cross-pair extreme-point cover (complete candidate pool)
P_CROSS_NB = _env('P_CROSS_NB', 14, int)
P_CROSS_CAP = _env('P_CROSS_CAP', 9000, int)
IMP_J = _env('IMP_J', 0, int)   # multi-restart cleanup (merge/redistribute are order dependent)
P_J_TRIES = _env('P_J_TRIES', 6, int)


# ================================================================== data loader
def _read(path):
    with open(path, newline='') as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


class Problem:
    def __init__(self, folder):
        _, rows = _read(os.path.join(folder, 'input_component.csv'))
        a = np.array([[float(v) for v in r[1:5]] for r in rows], dtype=np.float64)
        self.tlx, self.tly, self.brx, self.bry = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        self.typ = np.array([int(r[5]) for r in rows], dtype=np.int64)
        self.side = np.array([int(r[6]) for r in rows], dtype=np.int64)
        self.step = np.array([int(r[7]) for r in rows], dtype=np.int64)
        self.time = np.array([float(r[8]) for r in rows], dtype=np.float64)
        self.n = len(rows)

        _, srows = _read(os.path.join(folder, 'input_size.csv'))
        self.pcb_w, self.fov_w = float(srows[0][1]), float(srows[0][2])
        self.pcb_h, self.fov_h = float(srows[1][1]), float(srows[1][2])

        _, prows = _read(os.path.join(folder, 'input_parameter.csv'))
        (self.capture, self.recon, self.side_capture, mc,
         self.vx, self.vy, self.ax, self.ay) = [float(v) for v in prows[0]]
        self.max_core = max(int(mc), 4)

        # half-extent of the FOV region a component must sit inside
        self.hw = np.where(self.side == 1, self.fov_w / 4.0, self.fov_w / 2.0)
        self.hh = np.where(self.side == 1, self.fov_h / 4.0, self.fov_h / 2.0)
        self.w = self.brx - self.tlx
        self.h = self.bry - self.tly
        self.big = (self.w > 2 * self.hw) | (self.h > 2 * self.hh)

        # feasible-centre rectangle R_i (empty for big components)
        self.rx1 = self.brx - self.hw
        self.rx2 = self.tlx + self.hw
        self.ry1 = self.bry - self.hh
        self.ry2 = self.tly + self.hh


# ==================================================================== FOV model
class Fov:
    __slots__ = ('comps', 'x1', 'x2', 'y1', 'y2', 'step', 'fixed', 'x', 'y',
                 'hs', 'fd', 'ty', 'ld', 'pk')

    def __init__(self, comps, x1, x2, y1, y2, step, fixed=False):
        self.comps = comps
        # cached per-FOV derivatives; all invalidated together by _touch()
        self.hs = None      # holds a side component (capture 0.75 vs 0.50)
        self.fd = None      # holds a fiducial (must stay alone)
        self.ty = None      # max component type (decides the visit block)
        self.ld = None      # total inspection load
        self.pk = None      # largest single inspection job
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        self.step = step
        self.fixed = fixed          # big-component tile: centre may not move
        self.x = (x1 + x2) * 0.5
        self.y = (y1 + y2) * 0.5

    def recentre(self):
        self.x = (self.x1 + self.x2) * 0.5
        self.y = (self.y1 + self.y2) * 0.5


def _touch(f):
    f.hs = f.fd = f.ty = f.ld = f.pk = None


def _grow_cache(P, f, c):
    """Incremental cache update for appending one component -- appending can only
    ever raise these values, so nothing has to be recomputed from scratch."""
    if f.hs is not None and P.side[c] == 1:
        f.hs = True
    if f.fd is not None and P.typ[c] == 2:
        f.fd = True
    if f.ty is not None:
        t = int(P.typ[c])
        if t > f.ty:
            f.ty = t
    if f.ld is not None:
        f.ld += float(P.time[c])
    if f.pk is not None:
        t = float(P.time[c])
        if t > f.pk:
            f.pk = t


def _or(a, b):
    return None if (a is None or b is None) else (a or b)


def _fid_in(P, f):
    if f.fd is None:
        f.fd = any(P.typ[c] == 2 for c in f.comps)
    return f.fd


def _ftype(P, f):
    if f.ty is None:
        f.ty = max((int(P.typ[c]) for c in f.comps), default=0)
    return f.ty


def _fload(P, f):
    if f.ld is None:
        f.ld = float(sum(P.time[c] for c in f.comps))
    return f.ld


def _fpeak(P, f):
    if f.pk is None:
        f.pk = float(max((P.time[c] for c in f.comps), default=0.0))
    return f.pk


def _has_side(P, f):
    """Cached: does this FOV contain a side component (capture 0.75 vs 0.5)?"""
    if f.hs is None:
        f.hs = any(P.side[c] == 1 for c in f.comps)
    return f.hs


# ======================================================= integral-image max cov
def _max_coverage_point(x1, x2, y1, y2, wgt, ox, oy, g, nx, ny, topk=0, rng=None):
    """Grid point pierced by the largest total weight of rectangles."""
    i1 = np.ceil((x1 - ox) / g).astype(np.int64)
    i2 = np.floor((x2 - ox) / g).astype(np.int64)
    j1 = np.ceil((y1 - oy) / g).astype(np.int64)
    j2 = np.floor((y2 - oy) / g).astype(np.int64)
    ok = (i1 <= i2) & (j1 <= j2)
    if not ok.any():
        return None
    i1, i2, j1, j2, wv = i1[ok], i2[ok], j1[ok], j2[ok], wgt[ok]
    np.clip(i1, 0, nx - 1, out=i1)
    np.clip(i2, 0, nx - 1, out=i2)
    np.clip(j1, 0, ny - 1, out=j1)
    np.clip(j2, 0, ny - 1, out=j2)

    diff = np.zeros((nx + 1, ny + 1), dtype=np.float32)
    flat = diff.ravel()
    stride = ny + 1
    np.add.at(flat, i1 * stride + j1, wv)
    np.add.at(flat, i1 * stride + (j2 + 1), -wv)
    np.add.at(flat, (i2 + 1) * stride + j1, -wv)
    np.add.at(flat, (i2 + 1) * stride + (j2 + 1), wv)
    np.cumsum(diff, axis=0, out=diff)
    np.cumsum(diff, axis=1, out=diff)
    img = diff[:nx, :ny]
    if topk > 1 and rng is not None:
        flat = img.ravel()
        m = min(topk, flat.size)
        cand = np.argpartition(flat, -m)[-m:]
        cand = cand[flat[cand] > 0]
        if len(cand):
            k = int(cand[rng.integers(len(cand))])
            return ox + (k // ny) * g, oy + (k % ny) * g
    k = int(np.argmax(img))
    return ox + (k // ny) * g, oy + (k % ny) * g


def _cover_exact(P, must, bonus, bonus_w, topk=0, rng=None, cost_aware=False):
    """Cover using extreme-point candidates (corners of the R_i).

    This is the repair used inside LNS, so it runs thousands of times per
    dataset; the greedy loop itself is a numba kernel."""
    if not _FAST or topk > 1 or cost_aware:
        return _cover_exact_py(P, must, bonus, bonus_w, topk, rng, cost_aware)
    idx = np.concatenate([must, bonus]) if len(bonus) else must
    m = len(idx)
    nm = len(must)
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    step = int(P.step[idx[0]])
    cov = ((x2[:, None] >= x1[None, :] - EPS) & (x2[:, None] <= x2[None, :] + EPS) &
           (y2[:, None] >= y1[None, :] - EPS) & (y2[:, None] <= y2[None, :] + EPS))
    wgt = np.empty(m)
    wgt[:nm] = 1.0
    wgt[nm:] = bonus_w
    owner, chosen, ncho = fastops.cover_greedy(
        np.ascontiguousarray(cov, dtype=np.uint8), wgt, nm)

    groups = [[] for _ in range(ncho)]
    orphan = []
    for j in range(m):
        o = owner[j]
        if o >= 0:
            groups[o].append(j)
        elif o == -2:
            orphan.append(j)
    fovs, covered = [], []
    for t, g in enumerate(groups):
        if not g:
            continue
        ga = idx[g]
        bx1 = float(P.rx1[ga].max()); bx2 = float(P.rx2[ga].min())
        by1 = float(P.ry1[ga].max()); by2 = float(P.ry2[ga].min())
        if bx1 > bx2 + EPS or by1 > by2 + EPS:
            c = int(chosen[t])
            bx1 = bx2 = float(x2[c]); by1 = by2 = float(y2[c])
            ga = idx[[c]]
        f = Fov([int(v) for v in ga], bx1, bx2, by1, by2, step)
        f.x, f.y = float(x2[int(chosen[t])]), float(y2[int(chosen[t])])
        fovs.append(f)
        covered.extend(int(v) for v in ga)
    for j in orphan:
        c = int(idx[j])
        f = Fov([c], float(P.rx1[c]), float(P.rx2[c]),
                float(P.ry1[c]), float(P.ry2[c]), step)
        f.recentre()
        fovs.append(f)
        covered.append(c)
    return fovs, np.array(covered, dtype=np.int64)


def _cover_exact_py(P, must, bonus, bonus_w, topk=0, rng=None, cost_aware=False):
    """Cover using *extreme-point* candidates instead of a discretised grid.

    A minimum piercing set always admits an optimal solution whose points are
    corners of intersections of the R_i, so the bottom-right corner of each
    rectangle is a sound (and exact) candidate set.  Gains are maintained
    incrementally, which makes the whole greedy O(m^2) once rather than
    O(picks x gridsize) -- roughly 10x faster than the integral image for the
    small sub-problems produced by LNS."""
    idx = np.concatenate([must, bonus]) if len(bonus) else must
    m = len(idx)
    nm = len(must)
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    step = int(P.step[idx[0]])
    wgt = np.empty(m)
    wgt[:nm] = 1.0
    wgt[nm:] = bonus_w

    # candidate c (= corner of R_c) covers rectangle j  <=>  corner inside R_j
    cov = ((x2[:, None] >= x1[None, :] - EPS) & (x2[:, None] <= x2[None, :] + EPS) &
           (y2[:, None] >= y1[None, :] - EPS) & (y2[:, None] <= y2[None, :] + EPS))
    rem = np.ones(m, dtype=bool)
    gain = cov @ wgt
    if cost_aware:
        # capture is 0.75 s for a FOV holding any side component, else 0.50 s;
        # pick by gain/cost rather than raw coverage
        is_side = (P.side[idx] == 1)
        cst = np.where((cov & is_side[None, :]).any(1), P.side_capture, P.capture)
        cst = cst + P.recon * 0.5
    else:
        cst = None
    fovs, covered = [], []

    while rem[:nm].any():
        g = gain / cst if cst is not None else gain.copy()
        g[~cov[:, :nm][:, rem[:nm]].any(1)] = -1.0 if nm else -1.0
        b = int(np.argmax(g))
        if g[b] <= 0:
            j = int(np.where(rem[:nm])[0][0])
            b = j
        if topk > 1 and rng is not None:
            pool = np.where(g >= g[b] * 0.85)[0]
            if len(pool):
                b = int(pool[rng.integers(len(pool))])
        take = np.where(cov[b] & rem)[0]
        if len(take) == 0:
            take = np.array([b])
        bx1 = float(x1[take].max()); bx2 = float(x2[take].min())
        by1 = float(y1[take].max()); by2 = float(y2[take].min())
        if bx1 > bx2 + EPS or by1 > by2 + EPS:      # numerical guard
            take = np.array([b])
            bx1, bx2, by1, by2 = float(x1[b]), float(x2[b]), float(y1[b]), float(y2[b])
        f = Fov([int(v) for v in idx[take]], bx1, bx2, by1, by2, step)
        f.x, f.y = float(x2[b]), float(y2[b])
        fovs.append(f)
        covered.extend(int(v) for v in idx[take])
        gain -= cov[:, take] @ wgt[take]
        rem[take] = False
    return fovs, np.array(covered, dtype=np.int64)


def _cross_pool(P, idx, nb=None, cap=None):
    """Complete extreme-point candidate pool.

    Pushing an optimal piercing point right and down until it is blocked in both
    axes leaves it at (R_i.x2, R_j.y2) for *possibly different* i and j.  Using
    only the diagonal i == j silently discards most of the optimal positions --
    adding the crossed pairs for spatial neighbours cuts the FOV count by ~6 %
    at essentially no cost."""
    nb = P_CROSS_NB if nb is None else nb
    cap = P_CROSS_CAP if cap is None else cap
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    m = len(idx)
    order = np.argsort(x2)
    pts = {(round(float(x2[i]), 4), round(float(y2[i]), 4)) for i in range(m)}
    for a in range(m):
        i = int(order[a])
        for b in range(max(0, a - nb), min(m, a + nb + 1)):
            j = int(order[b])
            if j == i:
                continue
            px, py = float(x2[i]), float(y2[j])
            if x1[i] - EPS <= px <= x2[i] + EPS and y1[j] - EPS <= py <= y2[j] + EPS:
                pts.add((round(px, 4), round(py, 4)))
    arr = np.array(sorted(pts))
    px = arr[:, 0][:, None]
    py = arr[:, 1][:, None]
    cov = ((px >= x1[None, :] - EPS) & (px <= x2[None, :] + EPS) &
           (py >= y1[None, :] - EPS) & (py <= y2[None, :] + EPS))
    keep = cov.any(1)
    arr, cov = arr[keep], cov[keep]
    if len(arr) > cap:
        # the diagonal candidate (x2_i, y2_i) always lies inside R_i, so keeping
        # every diagonal point guarantees the pool still covers every component
        diag = np.zeros(len(arr), dtype=bool)
        dset = {(round(float(x2[i]), 4), round(float(y2[i]), 4)) for i in range(m)}
        for t in range(len(arr)):
            if (round(float(arr[t, 0]), 4), round(float(arr[t, 1]), 4)) in dset:
                diag[t] = True
        rest = np.where(~diag)[0]
        room = max(0, cap - int(diag.sum()))
        sel = np.concatenate([np.where(diag)[0],
                              rest[np.argsort(-cov[rest].sum(1))[:room]]])
        arr, cov = arr[sel], cov[sel]
    return arr, cov


def _cover_cross(P, must, bonus, bonus_w):
    """Greedy weighted cover over the complete extreme-point pool."""
    idx = np.concatenate([must, bonus]) if len(bonus) else must
    nm = len(must)
    pts, cov = _cross_pool(P, idx)
    wgt = np.empty(cov.shape[1])
    wgt[:nm] = 1.0
    wgt[nm:] = bonus_w
    step = int(P.step[idx[0]])
    rem = np.ones(cov.shape[1], dtype=bool)
    gain = cov @ wgt
    fovs, covered = [], []
    while rem[:nm].any():
        b = int(np.argmax(gain))
        take = np.where(cov[b] & rem)[0]
        if len(take) == 0 or not rem[:nm][cov[b][:nm]].any():
            j = int(np.where(rem[:nm])[0][0])
            o = np.where(cov[:, j])[0]
            if len(o) == 0:
                # no candidate reaches it: give it a FOV of its own
                c = int(idx[j])
                f = Fov([c], float(P.rx1[c]), float(P.rx2[c]),
                        float(P.ry1[c]), float(P.ry2[c]), step)
                f.recentre()
                fovs.append(f)
                covered.append(c)
                rem[j] = False
                continue
            b = int(o[0])
            take = np.where(cov[b] & rem)[0]
        g = idx[take]
        bx1 = float(P.rx1[g].max()); bx2 = float(P.rx2[g].min())
        by1 = float(P.ry1[g].max()); by2 = float(P.ry2[g].min())
        f = Fov([int(v) for v in g], bx1, bx2, by1, by2, step)
        f.x, f.y = float(pts[b][0]), float(pts[b][1])
        fovs.append(f)
        covered.extend(int(v) for v in g)
        gain -= cov[:, take] @ wgt[take]
        rem[take] = False
    return fovs, np.array(covered, dtype=np.int64)


def _localmax_pool(P, idx, cap=None, gridn=None):
    """Candidate pool = local maxima of the coverage field.

    Column generation on these instances keeps pricing out points that are
    locally maximal in coverage; computing them straight from one integral image
    yields the same kind of pool in milliseconds instead of seconds of LP
    solving.  The corner (x2_i, y2_i) of every rectangle is always included, so
    the pool provably reaches every component."""
    cap = P_POOL_CAP if cap is None else cap
    gridn = P_POOL_GRID if gridn is None else gridn
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    m = len(idx)
    ox, oy = float(x1.min()), float(y1.min())
    spx = max(float(x2.max()) - ox, 1e-9)
    spy = max(float(y2.max()) - oy, 1e-9)
    g = max(spx / gridn, spy / gridn, 0.05)
    nx, ny = int(spx / g) + 1, int(spy / g) + 1

    i1 = np.ceil((x1 - ox) / g).astype(np.int64)
    i2 = np.floor((x2 - ox) / g).astype(np.int64)
    j1 = np.ceil((y1 - oy) / g).astype(np.int64)
    j2 = np.floor((y2 - oy) / g).astype(np.int64)
    ok = (i1 <= i2) & (j1 <= j2)
    pts = [(float(x2[i]), float(y2[i])) for i in range(m)]
    if ok.any():
        d = np.zeros((nx + 1, ny + 1), dtype=np.float32)
        fl = d.ravel(); st = ny + 1
        np.add.at(fl, i1[ok] * st + j1[ok], 1.0)
        np.add.at(fl, i1[ok] * st + (j2[ok] + 1), -1.0)
        np.add.at(fl, (i2[ok] + 1) * st + j1[ok], -1.0)
        np.add.at(fl, (i2[ok] + 1) * st + (j2[ok] + 1), 1.0)
        np.cumsum(d, axis=0, out=d)
        np.cumsum(d, axis=1, out=d)
        img = d[:nx, :ny]
        pad = np.full((nx + 2, ny + 2), -1.0, dtype=np.float32)
        pad[1:-1, 1:-1] = img
        loc = img > 0
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                if dx == 1 and dy == 1:
                    continue
                loc &= img >= pad[dx:dx + nx, dy:dy + ny]
        ii, jj = np.where(loc)
        if len(ii) > cap:
            v = img[ii, jj]
            sel = np.argpartition(v, -cap)[-cap:]
            ii, jj = ii[sel], jj[sel]
        pts.extend(zip((ox + ii * g).tolist(), (oy + jj * g).tolist()))

    arr = np.array(sorted({(round(a, 4), round(b, 4)) for a, b in pts}))
    px = arr[:, 0][:, None]
    py = arr[:, 1][:, None]
    cov = ((px >= x1[None, :] - EPS) & (px <= x2[None, :] + EPS) &
           (py >= y1[None, :] - EPS) & (py <= y2[None, :] + EPS))
    keep = cov.any(1)
    return arr[keep], cov[keep]


def _greedy_pool(P, idx, nm, pts, cov, wgt, step):
    rem = np.ones(cov.shape[1], dtype=bool)
    gain = cov @ wgt
    fovs, covered = [], []
    while rem[:nm].any():
        b = int(np.argmax(gain))
        take = np.where(cov[b] & rem)[0]
        if len(take) == 0 or not rem[:nm][cov[b][:nm]].any():
            j = int(np.where(rem[:nm])[0][0])
            o = np.where(cov[:, j])[0]
            if len(o) == 0:
                c = int(idx[j])
                f = Fov([c], float(P.rx1[c]), float(P.rx2[c]),
                        float(P.ry1[c]), float(P.ry2[c]), step)
                f.recentre()
                fovs.append(f); covered.append(c); rem[j] = False
                continue
            b = int(o[0])
            take = np.where(cov[b] & rem)[0]
        g = idx[take]
        bx1 = float(P.rx1[g].max()); bx2 = float(P.rx2[g].min())
        by1 = float(P.ry1[g].max()); by2 = float(P.ry2[g].min())
        f = Fov([int(v) for v in g], bx1, bx2, by1, by2, step)
        f.x, f.y = float(pts[b][0]), float(pts[b][1])
        fovs.append(f)
        covered.extend(int(v) for v in g)
        gain -= cov[:, take] @ wgt[take]
        rem[take] = False
    return fovs, np.array(covered, dtype=np.int64)


def _cover_pool(P, must, bonus, bonus_w):
    idx = np.concatenate([must, bonus]) if len(bonus) else must
    nm = len(must)
    pts, cov = _localmax_pool(P, idx)
    wgt = np.empty(cov.shape[1])
    wgt[:nm] = 1.0
    wgt[nm:] = bonus_w
    return _greedy_pool(P, idx, nm, pts, cov, wgt, int(P.step[idx[0]]))


def _cover(P, must, bonus, bonus_w, deadline, topk=0, rng=None, fast=False):
    if IMP_N and not fast and len(must):
        return _cover_pool(P, must, bonus, bonus_w)
    if IMP_M and not fast and len(must):
        return _cover_cross(P, must, bonus, bonus_w)
    # The integral-image cover gives the better initial construction; the
    # extreme-point cover is ~10x faster and is what LNS uses for its many
    # small repairs.
    if IMP_C and fast and len(must) and (len(must) + len(bonus)) <= P_FAST_N:
        return _cover_exact(P, must, bonus, bonus_w, topk, rng)
    """Weighted set cover of `must` (bonus components ride along for free).

    Returns (list[Fov], covered_index_array).
    """
    fovs = []
    if len(must) == 0:
        return fovs, np.zeros(0, dtype=np.int64)

    idx = np.concatenate([must, bonus]) if len(bonus) else must
    is_must = np.zeros(len(idx), dtype=bool)
    is_must[:len(must)] = True
    wgt = np.where(is_must, 1.0, bonus_w).astype(np.float64)

    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    step = int(P.step[idx[0]])

    ox, oy = float(x1.min()), float(y1.min())
    span_x = max(float(x2.max()) - ox, 1e-6)
    span_y = max(float(y2.max()) - oy, 1e-6)
    g = max(span_x / P_GRID_CELLS, span_y / P_GRID_CELLS, P_GRID_MIN)
    nx = int(span_x / g) + 1
    ny = int(span_y / g) + 1

    alive = np.ones(len(idx), dtype=bool)
    covered = []

    while True:
        if not (alive & is_must).any():
            break
        if time.perf_counter() > deadline:
            # emergency: one FOV per remaining must-component
            for k in np.where(alive & is_must)[0]:
                fovs.append(Fov([int(idx[k])], x1[k], x2[k], y1[k], y2[k], step))
                covered.append(int(idx[k]))
                alive[k] = False
            break

        sel = np.where(alive)[0]
        if IMP_H:
            lox = float(x1[sel].min()); loy = float(y1[sel].min())
            sx = max(float(x2[sel].max()) - lox, 1e-6)
            sy = max(float(y2[sel].max()) - loy, 1e-6)
            cnx = int(sx / g) + 1
            cny = int(sy / g) + 1
            pt = _max_coverage_point(x1[sel], x2[sel], y1[sel], y2[sel],
                                     wgt[sel], lox, loy, g, cnx, cny, topk, rng)
        else:
            pt = _max_coverage_point(x1[sel], x2[sel], y1[sel], y2[sel],
                                     wgt[sel], ox, oy, g, nx, ny, topk, rng)
        take = None
        if pt is not None:
            px, py = pt
            hit = sel[(x1[sel] <= px + EPS) & (x2[sel] >= px - EPS) &
                      (y1[sel] <= py + EPS) & (y2[sel] >= py - EPS)]
            if len(hit) and is_must[hit].any():
                take = hit
        if take is None:
            # grid too coarse for the remaining rectangles -> seed on one of them
            k = np.where(alive & is_must)[0][0]
            px = (x1[k] + x2[k]) * 0.5
            py = (y1[k] + y2[k]) * 0.5
            take = sel[(x1[sel] <= px + EPS) & (x2[sel] >= px - EPS) &
                       (y1[sel] <= py + EPS) & (y2[sel] >= py - EPS)]

        bx1 = float(x1[take].max()); bx2 = float(x2[take].min())
        by1 = float(y1[take].max()); by2 = float(y2[take].min())

        # --- grow: absorb further rectangles that still leave a feasible centre
        chosen = list(take)
        inbox = np.zeros(len(idx), dtype=bool)
        inbox[take] = True
        for _ in range(P_GROW_PASSES):
            cand = np.where(alive & ~inbox &
                            (x1 <= bx2 + EPS) & (x2 >= bx1 - EPS) &
                            (y1 <= by2 + EPS) & (y2 >= by1 - EPS))[0]
            if len(cand) == 0:
                break
            shrink = ((np.maximum(bx1, x1[cand]) - bx1) + (bx2 - np.minimum(bx2, x2[cand])) +
                      (np.maximum(by1, y1[cand]) - by1) + (by2 - np.minimum(by2, y2[cand])))
            grew = False
            for c in cand[np.argsort(shrink)]:
                nx1 = bx1 if x1[c] < bx1 else float(x1[c])
                nx2 = bx2 if x2[c] > bx2 else float(x2[c])
                ny1 = by1 if y1[c] < by1 else float(y1[c])
                ny2 = by2 if y2[c] > by2 else float(y2[c])
                if nx1 > nx2 + EPS or ny1 > ny2 + EPS:
                    continue
                bx1, bx2, by1, by2 = nx1, nx2, ny1, ny2
                inbox[c] = True
                chosen.append(c)
                grew = True
            if not grew:
                break

        take = np.array(chosen, dtype=np.int64)
        fovs.append(Fov([int(v) for v in idx[take]], bx1, bx2, by1, by2, step))
        covered.extend(int(v) for v in idx[take])
        alive[take] = False

    return fovs, np.array(covered, dtype=np.int64)


P_RWCOVER = _env('P_RWCOVER', 1, int)     # exact-pool row-weighting cover
P_RW_ITERS = _env('P_RW_ITERS', 30000, int)
P_RW_TRIES = _env('P_RW_TRIES', 1, int)   # 같은 최소 개수 커버를 여러 개 만들어 이동이 짧은 것 선택
P_RW_MAX = _env('P_RW_MAX', 2500, int)    # fall back above this many components
P_RW_LNS = _env('P_RW_LNS', 1, int)       # exact-pool cover inside the LNS repair
P_RW_LNS_IT = _env('P_RW_LNS_IT', 1500, int)
P_RW_TOP = _env('P_RW_TOP', 0, int)       # 같은 크기 최적해 중 top 커버가 큰 것을 고름
P_RW_TOP_ROUNDS = _env('P_RW_TOP_ROUNDS', 3, int)
P_CPSAT = _env('P_CPSAT', 0, int)          # prove/close the cover exactly
P_CPSAT_TL = _env('P_CPSAT_TL', 0.08)
P_CPSAT_MAX = _env('P_CPSAT_MAX', 20000, int)

try:
    from ortools.sat.python import cp_model as _cpm
except Exception:
    _cpm = None


def _full_pool(P, idx):
    """Complete extreme-point candidate set for minimum rectangle piercing.

    An optimal piercing point can be pushed right until some R_i's right edge
    blocks it and down until some R_j's bottom edge does, so it is (x2_i, y2_j)
    -- and, crucially, that point has to lie in *both* R_i and R_j.  The earlier
    pool skipped that cross-membership test and only paired x-order neighbours,
    which made it 40x larger and still missing the points an optimum needs; with
    the test the pool is a few hundred to a few thousand points and contains an
    optimum by construction.  The diagonal (x2_i, y2_i) is always present, so
    every component keeps at least one candidate."""
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    okx = (x2[:, None] >= x1[None, :] - EPS) & (x2[:, None] <= x2[None, :] + EPS)
    oky = (y2[None, :] >= y1[:, None] - EPS) & (y2[None, :] <= y2[:, None] + EPS)
    ij = np.argwhere(okx & oky)
    if len(ij) == 0:
        return np.stack([x2, y2], 1)
    pts = np.stack([x2[ij[:, 0]], y2[ij[:, 1]]], 1)
    return np.unique(np.round(pts, 6), axis=0)


def _cov_of(P, idx, pts):
    px = pts[:, 0][:, None]
    py = pts[:, 1][:, None]
    return ((px >= P.rx1[idx][None, :] - EPS) & (px <= P.rx2[idx][None, :] + EPS) &
            (py >= P.ry1[idx][None, :] - EPS) & (py <= P.ry2[idx][None, :] + EPS))


def _assign_margin(P, idx, cov, chosen, pts, step):
    """Hand each component to the chosen centre that sits deepest inside its
    feasible box, keeping the 0.75 s capture surcharge on as few FOVs as
    possible.  Deeper means the FOV's own slack shrinks least, which is what the
    centre pass later spends on shortening core 0's tour."""
    chosen = [int(c) for c in chosen]
    taken = {c: [] for c in chosen}
    has_side = {c: False for c in chosen}
    order = np.argsort(-(P.side[idx].astype(int)))
    for j in order:
        opts = [c for c in chosen if cov[c, j]]
        if not opts:
            continue
        g = idx[j]
        best, bv = -1, None
        for c in opts:
            px, py = pts[c]
            mg = min(px - P.rx1[g], P.rx2[g] - px, py - P.ry1[g], P.ry2[g] - py)
            key = (not has_side[c], -mg)
            if bv is None or key < bv:
                bv, best = key, c
        if P.side[g] == 1:
            has_side[best] = True
        taken[best].append(int(j))
    out = []
    for c in chosen:
        if not taken[c]:
            continue
        g = idx[np.array(taken[c])]
        bx1 = float(P.rx1[g].max()); bx2 = float(P.rx2[g].min())
        by1 = float(P.ry1[g].max()); by2 = float(P.ry2[g].min())
        if bx1 > bx2 + EPS or by1 > by2 + EPS:
            bx1 = bx2 = float(pts[c][0]); by1 = by2 = float(pts[c][1])
        f = Fov([int(v) for v in g], bx1, bx2, by1, by2, step)
        f.x, f.y = float(pts[c][0]), float(pts[c][1])
        out.append(f)
    return out


def _top_swap(P, must, bonus, pts, cov, chosen):
    """Among the minimum-size covers, prefer one that also swallows top parts.

    The row-weighting search minimises the number of half-FOV windows needed for
    the side components and is indifferent between the many optima.  Top
    components that none of them happens to contain each need their own full FOV
    afterwards -- 408 of the 5,054 FOVs in the plan.  A candidate may be swapped
    for another one that still covers everything it alone was covering, so the
    size is preserved exactly while top coverage rises."""
    covb = _cov_of(P, bonus, pts)                # candidates x top components
    sel = list(int(c) for c in chosen)
    cnt = cov[sel].sum(0)                        # how many chosen cover each must
    got = covb[sel].any(0)
    for _ in range(max(1, P_RW_TOP_ROUNDS)):
        moved = False
        for pos, c in enumerate(sel):
            uniq = cov[c] & (cnt == 1)
            if uniq.any():
                ok = cov[:, uniq].all(1)
            else:
                ok = np.ones(cov.shape[0], dtype=bool)
            ok[sel] = False
            if not ok.any():
                continue
            others = [q for q in sel if q != c]
            base_got = covb[others].any(0) if others else np.zeros(len(bonus), bool)
            cand = np.where(ok)[0]
            add = (covb[cand] & ~base_got[None, :]).sum(1)
            cur = int((covb[c] & ~base_got).sum())
            b = int(np.argmax(add))
            if add[b] > cur:
                nc = int(cand[b])
                cnt -= cov[c].astype(np.int64)
                cnt += cov[nc].astype(np.int64)
                sel[pos] = nc
                moved = True
        if not moved:
            break
    return np.array(sorted(set(sel)), dtype=np.int64)


P_JOINT = _env('P_JOINT', 0, int)          # side/top 동시 피복 개선
P_JOINT_CELLS = _env('P_JOINT_CELLS', 30_000_000, int)   # |풀| x |부품| 상한
P_JOINT_INC = _env('P_JOINT_INC', 700_000, int)          # 인시던스 상한
P_JOINT_IT = _env('P_JOINT_IT', 12000, int)


def _tour_proxy(P, sel):
    """Cheap nearest-neighbour tour length, used only to compare covers."""
    n = len(sel)
    if n < 3:
        return 0.0
    dx = np.abs(sel[:, 0][:, None] - sel[:, 0][None, :])
    dy = np.abs(sel[:, 1][:, None] - sel[:, 1][None, :])
    C = np.maximum(_axis_time(dx, P.vx, P.ax), _axis_time(dy, P.vy, P.ay))
    np.fill_diagonal(C, np.inf)
    seen = np.zeros(n, dtype=bool)
    cur = 0
    seen[0] = True
    tot = 0.0
    for _ in range(n - 1):
        d = np.where(seen, np.inf, C[cur])
        j = int(np.argmin(d))
        tot += float(d[j])
        seen[j] = True
        cur = j
    return tot


def _cover_joint(P, idx, seed_fovs):
    """Re-solve side and top together, seeded by the two-pass result.

    Covering the side components first and then mopping up the leftover top
    components is a decomposition, not an optimum: a point placed slightly off
    the minimum side cover can swallow a whole cluster of top components and
    save the full FOV they would otherwise need.  Solved jointly on the complete
    pool the answer is 13.2% smaller on the datasets where it is tractable.

    The two-pass centres are themselves extreme points of subsets of `idx`, and
    the extreme-point condition only involves the pair that generates a point,
    so they all live in the joint pool -- which makes the two-pass solution a
    valid warm start and the search monotone."""
    if not P_JOINT or not _FAST or len(idx) < 3 or not seed_fovs:
        return None
    pts = _full_pool(P, idx)
    if len(pts) == 0 or len(pts) * len(idx) > P_JOINT_CELLS:
        return None
    cov = _cov_of(P, idx, pts)
    if not cov.any(0).all():
        return None
    m, n = cov.shape
    inc = int(cov.sum())
    if inc > P_JOINT_INC:
        return None
    # warm start: the two-pass centres, matched into the pool
    key = {(round(float(a), 6), round(float(b), 6)): i for i, (a, b) in enumerate(pts)}
    ins = np.zeros(m, np.bool_)
    for f in seed_fovs:
        i = key.get((round(float(f.x), 6), round(float(f.y), 6)))
        if i is not None:
            ins[i] = True
    chk = np.zeros(n, bool)
    for i in np.where(ins)[0]:
        chk |= cov[i]
    if not chk.all():
        return None                        # warm start not representable: bail
    r, c = np.nonzero(cov)                 # row-major: r is already sorted
    se = np.zeros(m + 1, np.int64)
    se[1:] = np.cumsum(np.bincount(r, minlength=m))
    ct = np.ascontiguousarray(cov.T)
    c2, r2 = np.nonzero(ct)                # column-major: c2 already sorted
    es = np.zeros(n + 1, np.int64)
    es[1:] = np.cumsum(np.bincount(c2, minlength=n))
    best = np.zeros(m, np.bool_)
    fastops.rwls(se, c.astype(np.int64), es, r2.astype(np.int64),
                 ins.copy(), P_JOINT_IT, 20260921, best)
    chosen = np.where(best)[0]
    if len(chosen) == 0 or len(chosen) >= int(ins.sum()):
        return None
    cov2 = _cov_of(P, idx, pts[chosen])
    fovs = _assign_margin(P, idx, cov2, list(range(len(chosen))), pts[chosen],
                          int(P.step[idx[0]]))
    if not fovs or sum(len(f.comps) for f in fovs) != len(idx):
        return None
    return fovs, np.array(idx, dtype=np.int64)


def _cover_rwls(P, must, bonus, bonus_w, iters=None, exact=False):
    """Near-exact minimum cover of `must`, with `bonus` riding along for free.

    On the complete pool this reaches the proven optimum on 8 of 10 sampled
    datasets and lands 7.6% below the integral-image greedy, in 15.6 ms --
    cheaper than the greedy it replaces."""
    if not P_RWCOVER or not _FAST or len(must) < 3 or len(must) > P_RW_MAX:
        return None
    pts = _full_pool(P, must)
    if len(pts) == 0 or len(pts) > 40000:
        return None
    cov = _cov_of(P, must, pts)
    if not cov.any(0).all():
        return None                       # pool incomplete: never risk it
    m, n = cov.shape
    r, c = np.nonzero(cov)
    o = np.argsort(r, kind='stable')
    se = np.zeros(m + 1, np.int64); np.add.at(se, r[o] + 1, 1); se = np.cumsum(se)
    o2 = np.argsort(c, kind='stable')
    es = np.zeros(n + 1, np.int64); np.add.at(es, c[o2] + 1, 1); es = np.cumsum(es)
    # greedy seed
    ins = np.zeros(m, np.bool_)
    rem = np.ones(n, bool)
    gain = cov.sum(1).astype(np.int64)
    while rem.any():
        b = int(np.argmax(gain))
        if gain[b] <= 0:
            break
        ins[b] = True
        got = cov[b] & rem
        rem &= ~cov[b]
        gain -= cov[:, got].sum(1).astype(np.int64)
    if rem.any():
        return None
    it = P_RW_ITERS if iters is None else iters
    tries = P_RW_TRIES if (exact and P_RW_TRIES > 1) else 1
    best = np.zeros(m, np.bool_)
    chosen = None
    bkey = None
    for t in range(tries):
        bs = np.zeros(m, np.bool_)
        fastops.rwls(se, c[o].astype(np.int64), es, r[o2].astype(np.int64),
                     ins.copy(), it, 20260920 + 7919 * t, bs)
        pick = np.where(bs)[0]
        if len(pick) == 0:
            continue
        # a minimum cover is not unique, and the alternatives differ by up to
        # 7.5% in tour length -- same count, so break the tie on travel
        key = (len(pick), _tour_proxy(P, pts[pick]) if tries > 1 else 0.0)
        if bkey is None or key < bkey:
            bkey, chosen, best = key, pick, bs
    if chosen is None or len(chosen) == 0:
        return None
    if exact and P_CPSAT and _cpm is not None and m <= P_CPSAT_MAX:
        # The row-weighting search lands 0.5% above the optimum.  With the
        # complete pool the instance is small enough that CP-SAT closes it in
        # ~33 ms per dataset -- it is handed the RWLS count as an upper bound
        # and a hint, so it only has to prove or improve.
        try:
            mdl = _cpm.CpModel()
            xs = [mdl.NewBoolVar('') for _ in range(m)]
            for j in range(n):
                sup = np.where(cov[:, j])[0]
                if len(sup):
                    mdl.AddBoolOr([xs[int(i)] for i in sup])
            mdl.Minimize(sum(xs))
            mdl.Add(sum(xs) <= len(chosen))
            mdl.AddHint(xs, [int(best[i]) for i in range(m)])
            so = _cpm.CpSolver()
            so.parameters.max_time_in_seconds = P_CPSAT_TL
            so.parameters.num_search_workers = 8
            r = so.Solve(mdl)
            if so.StatusName(r) in ('OPTIMAL', 'FEASIBLE'):
                pick = np.array([i for i in range(m) if so.Value(xs[i])], dtype=np.int64)
                if 0 < len(pick) < len(chosen):
                    cv = np.zeros(n, dtype=bool)
                    for i in pick:
                        cv |= cov[i]
                    if cv.all():
                        chosen = pick
        except Exception:
            pass
    if P_RW_TOP and len(bonus):
        chosen = _top_swap(P, must, bonus, pts, cov, chosen)
    idx = np.concatenate([must, bonus]) if len(bonus) else must
    cov2 = _cov_of(P, idx, pts[chosen])
    remap = {i: i for i in range(len(chosen))}
    fovs = _assign_margin(P, idx, cov2, list(remap), pts[chosen], int(P.step[must[0]]))
    covered = np.concatenate([f.comps for f in fovs]) if fovs else np.zeros(0, np.int64)
    return fovs, np.array(covered, dtype=np.int64)


def _cover_lattice(P, must, bonus, bonus_w, ofx, ofy, pct):
    """Hochbaum-Maass style: pierce R_i with a *regular lattice* of candidate
    centres, offset-swept.  A lattice with pitch <= min rectangle width pierces
    every rectangle, so a globally consistent (rather than myopic) cover is
    obtained.  Returns (fovs, covered, leftover_must)."""
    if len(must) == 0:
        return [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    idx = np.concatenate([must, bonus]) if len(bonus) else must
    nm = len(must)
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    step = int(P.step[idx[0]])

    wx = x2[:nm] - x1[:nm]
    wy = y2[:nm] - y1[:nm]
    px = max(float(np.percentile(wx, pct)), 1e-3)
    py = max(float(np.percentile(wy, pct)), 1e-3)

    lo_x, hi_x = float(x1.min()), float(x2.max())
    lo_y, hi_y = float(y1.min()), float(y2.max())
    gx = np.arange(lo_x + ofx * px, hi_x + px, px)
    gy = np.arange(lo_y + ofy * py, hi_y + py, py)
    if len(gx) == 0 or len(gy) == 0 or len(gx) * len(gy) > 20000:
        return [], np.zeros(0, dtype=np.int64), must

    # coverage matrix: lattice point -> components it can host
    okx = (gx[:, None] >= x1[None, :] - EPS) & (gx[:, None] <= x2[None, :] + EPS)
    oky = (gy[:, None] >= y1[None, :] - EPS) & (gy[:, None] <= y2[None, :] + EPS)
    cov = okx[:, None, :] & oky[None, :, :]
    cov = cov.reshape(len(gx) * len(gy), len(idx))
    pts = np.stack(np.meshgrid(gx, gy, indexing='ij'), -1).reshape(-1, 2)

    live = cov.any(1)
    cov, pts = cov[live], pts[live]
    if len(pts) == 0:
        return [], np.zeros(0, dtype=np.int64), must

    wgt = np.empty(len(idx))
    wgt[:nm] = 1.0
    wgt[nm:] = bonus_w
    rem = np.ones(len(idx), dtype=bool)

    fovs, covered = [], []
    while rem[:nm].any():
        gain = (cov & rem[None, :]) @ wgt
        b = int(np.argmax(gain))
        take = np.where(cov[b] & rem)[0]
        if len(take) == 0 or not rem[:nm][cov[b][:nm]].any():
            break
        bx1 = float(x1[take].max()); bx2 = float(x2[take].min())
        by1 = float(y1[take].max()); by2 = float(y2[take].min())
        f = Fov([int(v) for v in idx[take]], bx1, bx2, by1, by2, step)
        f.x, f.y = float(pts[b][0]), float(pts[b][1])
        fovs.append(f)
        covered.extend(int(v) for v in idx[take])
        rem[take] = False
    leftover = must[rem[:nm]]
    return fovs, np.array(covered, dtype=np.int64), leftover


# ================================================================ big component
def _tile_positions(P, i, sx, sy):
    """Tile centres for big component i; sx, sy in [0,1] slide the grid inside
    the slack that still guarantees full coverage."""
    ew, eh = 2 * P.hw[i], 2 * P.hh[i]
    nx = max(1, math.ceil(P.w[i] / ew - 1e-9))
    ny = max(1, math.ceil(P.h[i] / eh - 1e-9))
    if nx == 1:
        xs = [P.tlx[i] + P.w[i] / 2 + (sx - 0.5) * max(0.0, ew - P.w[i])]
    else:
        xs = [P.tlx[i] + ew / 2 + k * (P.w[i] - ew) / (nx - 1) for k in range(nx)]
    if ny == 1:
        ys = [P.tly[i] + P.h[i] / 2 + (sy - 0.5) * max(0.0, eh - P.h[i])]
    else:
        ys = [P.tly[i] + eh / 2 + k * (P.h[i] - eh) / (ny - 1) for k in range(ny)]
    return xs, ys, ew, eh


def _big_tiles(P):
    fovs = []
    small = np.where(~P.big & (P.typ != 2))[0]
    for i in np.where(P.big)[0]:
        best = None
        offs = ((0.5, 0.5),) if not IMP_G else ((0.5, 0.5), (0.0, 0.0), (1.0, 1.0),
                                                (0.0, 1.0), (1.0, 0.0))
        for sx, sy in offs:
            xs, ys, ew, eh = _tile_positions(P, i, sx, sy)
            if best is None:
                best = (xs, ys, -1)
            if not IMP_G:
                break
            # how many not-yet-placed small components of the same step would
            # such a tile absorb for free?
            got = 0
            cand = small[(P.step[small] == P.step[i])]
            for cx in xs:
                for cy in ys:
                    got += int(((cx - P.hw[cand] <= P.tlx[cand] + EPS) &
                                (cy - P.hh[cand] <= P.tly[cand] + EPS) &
                                (cx + P.hw[cand] >= P.brx[cand] - EPS) &
                                (cy + P.hh[cand] >= P.bry[cand] - EPS)).sum())
            if got > best[2]:
                best = (xs, ys, got)
        xs, ys, _ = best
        for cx in xs:
            for cy in ys:
                fovs.append(Fov([int(i)], cx, cx, cy, cy, int(P.step[i]), fixed=True))
    return fovs


# ================================================================ local search
def _merge(P, fovs, sweeps=P_MERGE_SWEEPS):
    """Merge FOV pairs whose feasible boxes intersect (same step, no fiducial)."""
    # The numba sweep was measured *worse* than the Python one (12,800 vs
    # 12,771): its tie-breaking differs subtly and the Python version is already
    # only ~30 ms per dataset.  Kept behind a flag for reference.
    if not _FAST or not _env('MERGE_NB', 0, int) or len(fovs) < 3:
        return _merge_py(P, fovs, sweeps)
    for _ in range(sweeps):
        F = len(fovs)
        mov = np.array([k for k, f in enumerate(fovs) if not _fid_in(P, f)],
                       dtype=np.int64)
        if len(mov) < 2:
            break
        x1 = np.fromiter((f.x1 for f in fovs), float, F)
        x2 = np.fromiter((f.x2 for f in fovs), float, F)
        y1 = np.fromiter((f.y1 for f in fovs), float, F)
        y2 = np.fromiter((f.y2 for f in fovs), float, F)
        fstep = np.fromiter((f.step for f in fovs), np.int64, F)
        fixed = np.array([f.fixed for f in fovs], dtype=np.bool_)
        cap = np.array([P.side_capture if _has_side(P, f) else P.capture
                        for f in fovs])
        alive = np.ones(F, dtype=np.int64)
        dst, src = fastops.merge_sweep(x1, x2, y1, y2, fstep, fixed, cap,
                                       mov, alive)
        if len(dst) == 0:
            break
        for t in range(len(dst)):
            fa = fovs[int(dst[t])]
            fb = fovs[int(src[t])]
            fa.comps = fa.comps + fb.comps
            fa.hs = _or(fa.hs, fb.hs)
            fa.fd = _or(fa.fd, fb.fd)
            fa.ty = None if fa.ty is None or fb.ty is None else max(fa.ty, fb.ty)
            fa.ld = None if fa.ld is None or fb.ld is None else fa.ld + fb.ld
            fa.pk = None if fa.pk is None or fb.pk is None else max(fa.pk, fb.pk)
            fa.x1 = max(fa.x1, fb.x1); fa.x2 = min(fa.x2, fb.x2)
            fa.y1 = max(fa.y1, fb.y1); fa.y2 = min(fa.y2, fb.y2)
            fa.fixed = fa.fixed or fb.fixed
            fa.recentre()
        fovs = [f for k, f in enumerate(fovs) if alive[k]]
    return fovs


def _merge_py(P, fovs, sweeps=P_MERGE_SWEEPS):
    """Merge FOV pairs whose feasible boxes intersect (same step, no fiducial)."""
    fid = P.typ == 2
    for _ in range(sweeps):
        groups = {}
        for k, f in enumerate(fovs):
            if _fid_in(P, f):
                continue
            groups.setdefault(f.step, []).append(k)

        dead = set()
        merged_any = False
        for _, ks in groups.items():
            if len(ks) < 2:
                continue
            ka = np.array(ks)
            X1 = np.array([fovs[k].x1 for k in ks]); X2 = np.array([fovs[k].x2 for k in ks])
            Y1 = np.array([fovs[k].y1 for k in ks]); Y2 = np.array([fovs[k].y2 for k in ks])
            has_side = np.array([_has_side(P, fovs[k]) for k in ks])
            fixed = np.array([fovs[k].fixed for k in ks])

            inter = ((X1[:, None] <= X2[None, :] + EPS) & (X2[:, None] >= X1[None, :] - EPS) &
                     (Y1[:, None] <= Y2[None, :] + EPS) & (Y2[:, None] >= Y1[None, :] - EPS))
            inter &= ~(fixed[:, None] & fixed[None, :])     # two tiles never merge
            iu, ju = np.triu_indices(len(ks), 1)
            m = inter[iu, ju]
            if not m.any():
                continue
            iu, ju = iu[m], ju[m]
            # saving: 0.75+0.75->0.75 beats 0.75+0.5->0.75 beats 0.5+0.5->0.5
            cap = np.where(has_side, P.side_capture, P.capture)
            save = cap[iu] + cap[ju] - np.maximum(cap[iu], cap[ju])
            for o in np.argsort(-save):
                a, b = int(iu[o]), int(ju[o])
                ga, gb = int(ka[a]), int(ka[b])
                if ga in dead or gb in dead:
                    continue
                fa, fb = fovs[ga], fovs[gb]
                nx1 = max(fa.x1, fb.x1); nx2 = min(fa.x2, fb.x2)
                ny1 = max(fa.y1, fb.y1); ny2 = min(fa.y2, fb.y2)
                if nx1 > nx2 + EPS or ny1 > ny2 + EPS:
                    continue
                fa.comps = fa.comps + fb.comps
                fa.hs = _or(fa.hs, fb.hs)
                fa.fd = _or(fa.fd, fb.fd)
                fa.ty = (None if fa.ty is None or fb.ty is None
                         else max(fa.ty, fb.ty))
                fa.ld = (None if fa.ld is None or fb.ld is None
                         else fa.ld + fb.ld)
                fa.pk = (None if fa.pk is None or fb.pk is None
                         else max(fa.pk, fb.pk))
                fa.x1, fa.x2, fa.y1, fa.y2 = nx1, nx2, ny1, ny2
                fa.fixed = fa.fixed or fb.fixed
                fa.recentre()
                dead.add(gb)
                merged_any = True
        if not merged_any:
            break
        fovs = [f for k, f in enumerate(fovs) if k not in dead]
    return fovs


def _drop_redundant(P, fovs):
    """Delete a FOV when every one of its components fits in some other FOV."""
    F = len(fovs)
    if F < 2:
        return fovs
    if not _FAST:
        return _drop_redundant_py(P, fovs)
    cx = np.fromiter((f.x for f in fovs), float, F)
    cy = np.fromiter((f.y for f in fovs), float, F)
    fstep = np.fromiter((f.step for f in fovs), np.int64, F)
    locked = np.array([f.fixed or _fid_in(P, f) for f in fovs], dtype=np.bool_)
    off = np.zeros(F + 1, dtype=np.int64)
    for i, f in enumerate(fovs):
        off[i + 1] = off[i] + len(f.comps)
    cf = np.empty(off[-1], dtype=np.int64)
    for i, f in enumerate(fovs):
        cf[off[i]:off[i + 1]] = f.comps
    ncomp = np.diff(off)
    order = np.argsort(ncomp).astype(np.int64)
    alive, assign = fastops.drop_kernel(cx, cy, fstep, locked, ncomp, cf, off,
                                        P.tlx, P.tly, P.brx, P.bry, P.hw, P.hh,
                                        order, EPS)
    moved = np.where(assign >= 0)[0]
    for c in moved:
        _add(P, fovs[int(assign[c])], int(c))
    return [f for k, f in enumerate(fovs) if alive[k]]


def _drop_redundant_py(P, fovs):
    """Delete a FOV when every one of its components fits in some other FOV."""
    F = len(fovs)
    if F < 2:
        return fovs
    fid = P.typ == 2
    cx = np.array([f.x for f in fovs]); cy = np.array([f.y for f in fovs])
    fstep = np.array([f.step for f in fovs])
    locked = np.array([f.fixed or _fid_in(P, f) for f in fovs])

    # hosts[c] = FOVs (same step) that geometrically contain component c
    hosts = {}
    for k, f in enumerate(fovs):
        for c in f.comps:
            if c in hosts:
                continue
            m = ((fstep == f.step) &
                 (cx - P.hw[c] <= P.tlx[c] + EPS) & (cy - P.hh[c] <= P.tly[c] + EPS) &
                 (cx + P.hw[c] >= P.brx[c] - EPS) & (cy + P.hh[c] >= P.bry[c] - EPS))
            hosts[c] = np.where(m)[0]

    alive = np.ones(F, dtype=bool)
    for k in sorted(range(F), key=lambda i: len(fovs[i].comps)):
        if locked[k] or not alive[k]:
            continue
        plan = []
        for c in fovs[k].comps:
            h = hosts.get(c)
            if h is None:
                plan = None
                break
            h = h[(h != k) & alive[h]]
            if len(h) == 0:
                plan = None
                break
            plan.append((c, int(h[0])))
        if plan is not None:
            for c, h in plan:
                _add(P, fovs[h], c)
            alive[k] = False
    return [f for k, f in enumerate(fovs) if alive[k]]


def _add(P, fov, c):
    """Attach component c to a FOV, keeping its feasible-centre box valid."""
    fov.comps.append(int(c))
    _grow_cache(P, fov, c)
    if not fov.fixed:
        if P.rx1[c] > fov.x1: fov.x1 = float(P.rx1[c])
        if P.rx2[c] < fov.x2: fov.x2 = float(P.rx2[c])
        if P.ry1[c] > fov.y1: fov.y1 = float(P.ry1[c])
        if P.ry2[c] < fov.y2: fov.y2 = float(P.ry2[c])


def _redistribute(P, fovs):
    """Delete a FOV when its components can be spread over neighbours that are
    allowed to re-centre (box intersection).  Numba kernel when available."""
    F = len(fovs)
    if F < 2:
        return fovs
    if not _FAST:
        return _redistribute_py(P, fovs)
    locked = np.array([f.fixed or _fid_in(P, f) for f in fovs], dtype=np.bool_)
    bx = np.empty((F, 4))
    cx = np.empty(F); cy = np.empty(F)
    fstep = np.empty(F, dtype=np.int64)
    off = np.zeros(F + 1, dtype=np.int64)
    for i, f in enumerate(fovs):
        bx[i] = (f.x1, f.x2, f.y1, f.y2)
        cx[i], cy[i] = f.x, f.y
        fstep[i] = f.step
        off[i + 1] = off[i] + len(f.comps)
    cf = np.empty(off[-1], dtype=np.int64)
    for i, f in enumerate(fovs):
        cf[off[i]:off[i + 1]] = f.comps
    order = np.argsort(np.diff(off)).astype(np.int64)
    alive = np.ones(F, dtype=np.int64)
    lim = (1.6 * P.fov_w, 1.6 * P.fov_h) if IMP_H else (1e18, 1e18)
    assign, alive, tent = fastops.redistribute_kernel(
        bx, fstep, locked, alive, cf, off, P.rx1, P.rx2, P.ry1, P.ry2,
        P.big, order, cx, cy, lim[0], lim[1], EPS)
    moved = np.where(assign >= 0)[0]
    if len(moved) == 0:
        return fovs
    for c in moved:
        j = int(assign[c])
        fovs[j].comps.append(int(c))
        _grow_cache(P, fovs[j], int(c))
    for j in range(F):
        if alive[j] and not fovs[j].fixed:
            fovs[j].x1, fovs[j].x2, fovs[j].y1, fovs[j].y2 = tent[j]
            fovs[j].recentre()
    return [f for k, f in enumerate(fovs) if alive[k]]


def _redistribute_py(P, fovs):
    """Stronger redundancy removal: a FOV is deleted when its components can be
    spread over neighbouring FOVs *after letting those FOVs re-centre* (box
    intersection), not merely where they already sit."""
    F = len(fovs)
    if F < 2:
        return fovs
    fid = P.typ == 2
    locked = [f.fixed or _fid_in(P, f) for f in fovs]
    alive = [True] * F
    box = [[f.x1, f.x2, f.y1, f.y2] for f in fovs]
    fstep = np.array([f.step for f in fovs])

    for k in sorted(range(F), key=lambda i: len(fovs[i].comps)):
        if locked[k] or not alive[k]:
            continue
        if IMP_H:
            rx, ry = fovs[k].x, fovs[k].y
            lim_x, lim_y = 1.6 * P.fov_w, 1.6 * P.fov_h
            hosts = [j for j in range(F)
                     if alive[j] and j != k and fstep[j] == fstep[k]
                     and abs(fovs[j].x - rx) <= lim_x and abs(fovs[j].y - ry) <= lim_y]
        else:
            hosts = [j for j in range(F)
                     if alive[j] and j != k and fstep[j] == fstep[k]]
        if not hosts:
            continue
        tent, plan, ok = {}, [], True
        for c in fovs[k].comps:
            if P.big[c]:
                ok = False
                break
            placed = False
            for j in hosts:
                bx = tent.get(j, box[j])
                nx1 = bx[0] if P.rx1[c] < bx[0] else float(P.rx1[c])
                nx2 = bx[1] if P.rx2[c] > bx[1] else float(P.rx2[c])
                ny1 = bx[2] if P.ry1[c] < bx[2] else float(P.ry1[c])
                ny2 = bx[3] if P.ry2[c] > bx[3] else float(P.ry2[c])
                if nx1 <= nx2 + EPS and ny1 <= ny2 + EPS:
                    tent[j] = [nx1, nx2, ny1, ny2]
                    plan.append((c, j))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok and plan:
            for c, j in plan:
                fovs[j].comps.append(int(c))
                _grow_cache(P, fovs[j], int(c))
            for j, bx in tent.items():
                box[j] = bx
                fovs[j].x1, fovs[j].x2, fovs[j].y1, fovs[j].y2 = bx
                fovs[j].recentre()
            alive[k] = False
    return [f for k, f in enumerate(fovs) if alive[k]]


def _bottleneck(P, fovs):
    """Rough (core0, best-inspection) estimate for the plan, before sequencing.
    Used to decide where the remaining time budget is worth spending."""
    n = len(fovs)
    cap = sum(P.side_capture if _has_side(P, f) else P.capture for f in fovs)
    hop = max(_axis_time(np.array([P.fov_w]), P.vx, P.ax)[0],
              _axis_time(np.array([P.fov_h]), P.vy, P.ay)[0])
    c0 = cap + max(0, n - 1) * float(hop) * 0.8 + 2 * Z_PENALTY
    rec = n * P.recon * 0.5
    st = sum(float(P.time[c] for c in ()) for f in ()) if False else \
        sum(_fload(P, f) for f in fovs)
    bestk, bestv = 4, None
    for k in range(4, P.max_core + 1):
        v = max(c0, rec, st / (k - 3)) + BETA_CORE * k
        if bestv is None or v < bestv:
            bestv, bestk = v, k
    return max(c0, rec), st / (bestk - 3)


def _cost_proxy(P, fovs):
    """Capture + reconstruction load -- what FOV count actually costs."""
    c = 0.0
    for f in fovs:
        c += P.side_capture if _has_side(P, f) else P.capture
    return c + len(fovs) * P.recon * 0.5


def _absorb(P, fovs, freed):
    """Put freed components into existing FOVs where they geometrically fit.

    The fiducial FOV is excluded as a host: the contest rules reserve it for the
    fiducial alone.  The Fov objects are shared, so filtering the host list here
    still mutates the real plan."""
    if len(fovs) == 0 or len(freed) == 0:
        return list(freed)
    sub = [f for f in fovs if not _fid_in(P, f)]
    if len(sub) != len(fovs):
        fovs = sub
        if len(fovs) == 0:
            return list(freed)
    if not _FAST:
        return _absorb_py(P, fovs, freed)
    F = len(fovs)
    cx = np.fromiter((f.x for f in fovs), float, F)
    cy = np.fromiter((f.y for f in fovs), float, F)
    fstep = np.fromiter((f.step for f in fovs), np.int64, F)
    hasside = np.array([_has_side(P, f) for f in fovs], dtype=np.bool_)
    ncomp = np.fromiter((len(f.comps) for f in fovs), np.int64, F)
    bx = np.empty((F, 4))
    for i, f in enumerate(fovs):
        bx[i] = (f.x1, f.x2, f.y1, f.y2)
    cids = np.asarray(freed, dtype=np.int64)
    got = fastops.absorb_kernel(cx, cy, fstep, hasside, ncomp, bx, cids,
                                P.step[cids].astype(np.int64),
                                P.side[cids].astype(np.int64),
                                P.tlx, P.tly, P.brx, P.bry, P.hw, P.hh, EPS)
    left = []
    for t, c in enumerate(cids):
        j = int(got[t])
        if j < 0:
            left.append(int(c))
        else:
            fovs[j].comps.append(int(c))
            _grow_cache(P, fovs[j], int(c))
    for j, f in enumerate(fovs):
        if not f.fixed:
            f.x1, f.x2, f.y1, f.y2 = bx[j]
    return left


def _absorb_py(P, fovs, freed):
    """Put freed components into existing FOVs where they geometrically fit."""
    if len(fovs) == 0:
        return list(freed)
    cx = np.array([f.x for f in fovs]); cy = np.array([f.y for f in fovs])
    fstep = np.array([f.step for f in fovs])
    hasside = np.array([_has_side(P, f) for f in fovs])
    left = []
    for c in freed:
        m = ((fstep == P.step[c]) &
             (cx - P.hw[c] <= P.tlx[c] + EPS) & (cy - P.hh[c] <= P.tly[c] + EPS) &
             (cx + P.hw[c] >= P.brx[c] - EPS) & (cy + P.hh[c] >= P.bry[c] - EPS))
        w = np.where(m)[0]
        if len(w) == 0:
            left.append(c)
            continue
        if P.side[c] == 1 and hasside[w].any():
            w = w[hasside[w]]
        j = int(w[np.argmin([len(fovs[i].comps) for i in w])])
        _add(P, fovs[j], c)
        hasside[j] = hasside[j] or P.side[c] == 1
    return left


def _repair_free(P, keep, freed, deadline, topk=0, rng=None, fast=True):
    """Re-cover components released by a destroy step."""
    left = _absorb(P, keep, freed)
    for st in (0, 1):
        sub = np.array([c for c in left if P.step[c] == st], dtype=np.int64)
        if len(sub) == 0:
            continue
        side = sub[P.side[sub] == 1]
        top = sub[P.side[sub] == 0]
        if len(side):
            got = _cover_rwls(P, side, top, P_BONUS, P_RW_LNS_IT) if P_RW_LNS else None
            fs, cov = got if got is not None else \
                _cover(P, side, top, P_BONUS, deadline, topk, rng, fast)
            keep.extend(fs)
            cs = set(int(v) for v in cov)
            top = np.array([t for t in top if int(t) not in cs], dtype=np.int64)
        if len(top):
            empty = np.zeros(0, dtype=np.int64)
            got = _cover_rwls(P, top, empty, 0.0, P_RW_LNS_IT) if P_RW_LNS else None
            fs, _ = got if got is not None else \
                _cover(P, top, empty, 0.0, deadline, topk, rng, fast)
            keep.extend(fs)
    return keep


def _clone(fovs):
    out = []
    for f in fovs:
        g = Fov(list(f.comps), f.x1, f.x2, f.y1, f.y2, f.step, f.fixed)
        g.x, g.y, g.hs, g.fd, g.ty, g.ld, g.pk = f.x, f.y, f.hs, f.fd, f.ty, f.ld, f.pk
        out.append(g)
    return out


def _cap_of(P, fs):
    c = sum(P.side_capture if _has_side(P, f) else P.capture for f in fs) \
        + len(fs) * P.recon * 0.5
    if P_SLACKW:
        # Two covers of the same size are not equally good: the one whose FOVs
        # have more room to slide lets the centre pass shorten the tour later.
        room = 0.0
        for f in fs:
            room += min(f.x2 - f.x1, P.fov_w) + min(f.y2 - f.y1, P.fov_h)
        c -= P_SLACKW * room
    return c


def _lns(P, fovs, deadline, rng):
    """Localised large-neighbourhood search over the FOV set.

    Each iteration touches only the destroyed window and its immediate
    neighbours -- cloning, re-covering, merging and costing are all local, which
    is what makes the iteration count high enough for a metaheuristic to work.

    IMP_E additionally enables several destroy operators, a noisy repair and
    record-to-record acceptance."""
    cur = list(fovs)
    curc = _cap_of(P, cur)
    best, bestc = list(cur), curc
    fid = P.typ == 2
    lock = [f.fixed or _fid_in(P, f) for f in cur]
    dev = P_LNS_DEV if IMP_E else 0.0
    ops = 4 if IMP_E else 1
    noimp = 0
    stale = 0
    it = 0

    while time.perf_counter() < deadline:
        it += 1
        F = len(cur)
        if len(lock) != F:
            lock = [f.fixed or _fid_in(P, f) for f in cur]
        movable = [k for k in range(F) if not lock[k]]
        if len(movable) < 3:
            break
        cx = np.fromiter((f.x for f in cur), float, F)
        cy = np.fromiter((f.y for f in cur), float, F)

        # ---------------- destroy -------------------------------------------
        op = int(rng.integers(ops))
        if op == 0:
            seed = movable[int(rng.integers(len(movable)))]
            rad = P.fov_w * float(rng.uniform(0.6, 1.9))
            rm = [k for k in movable
                  if abs(cx[k] - cx[seed]) <= rad and abs(cy[k] - cy[seed]) <= rad]
            if len(rm) > 14:
                rm = list(rng.choice(np.array(rm), size=14, replace=False))
        elif op == 1:
            q = min(len(movable), int(rng.integers(3, 10)))
            rm = sorted(movable, key=lambda k: len(cur[k].comps))[:q]
        elif op == 2:
            q = min(len(movable), int(rng.integers(3, 10)))
            rm = sorted(movable, key=lambda k: -(P.side_capture if _has_side(P, cur[k])
                                                 else P.capture)
                        / max(len(cur[k].comps), 1))[:q]
        else:
            q = min(len(movable), int(rng.integers(3, 12)))
            rm = list(rng.choice(np.array(movable), size=q, replace=False))
        rm = [int(k) for k in rm]
        if len(rm) < 2:
            continue
        rms = set(rm)

        # ---------------- local neighbourhood -------------------------------
        rx0 = min(cx[k] for k in rm) - 1.1 * P.fov_w
        rx1 = max(cx[k] for k in rm) + 1.1 * P.fov_w
        ry0 = min(cy[k] for k in rm) - 1.1 * P.fov_h
        ry1 = max(cy[k] for k in rm) + 1.1 * P.fov_h
        steps_rm = {cur[k].step for k in rm}
        nb = [k for k in range(F)
              if k not in rms and cur[k].step in steps_rm
              and rx0 <= cx[k] <= rx1 and ry0 <= cy[k] <= ry1]
        if len(nb) > 40:
            nb = nb[:40]
        nbs = set(nb)

        old = [cur[k] for k in rm] + [cur[k] for k in nb]
        oldc = _cap_of(P, old)
        freed = [c for k in rm for c in cur[k].comps]

        keep = _clone([cur[k] for k in nb])
        keep = _repair_free(P, keep, freed, deadline,
                            topk=(4 if IMP_E else 0), rng=rng)
        keep = _merge(P, keep, sweeps=4)
        for f in keep:
            f.recentre()
        newc = _cap_of(P, keep)

        if newc < oldc + dev - 1e-9:
            cur = [f for k, f in enumerate(cur) if k not in rms and k not in nbs] + keep
            curc = curc - oldc + newc
            lock = []
            if curc < bestc - 1e-9:
                best, bestc, noimp, stale = list(cur), curc, 0, 0
            else:
                noimp += 1
                stale += 1
        else:
            noimp += 1
            stale += 1
        if IMP_E and noimp >= P_LNS_RESET:
            cur, curc, noimp, lock = list(best), bestc, 0, []
        # On roughly half the boards LNS never finds anything; stopping early
        # hands that time back instead of burning it as RT (1 s = 1 point).
        if stale >= P_LNS_STOP:
            break
    return best


def _kexchange(P, fovs, deadline, rng):
    """Deterministic k-exchange sweep: take the smallest FOV together with its
    nearest same-step neighbour (and optionally a third), release their
    components and re-cover the local area.  Accept when the capture+recon cost
    drops.  This is the systematic counterpart to the randomised LNS -- it
    guarantees every small FOV gets at least one removal attempt."""
    fid = P.typ == 2
    rounds = 0
    while rounds < 3 and time.perf_counter() < deadline:
        rounds += 1
        F = len(fovs)
        lock = [f.fixed or _fid_in(P, f) for f in fovs]
        cx = np.fromiter((f.x for f in fovs), float, F)
        cy = np.fromiter((f.y for f in fovs), float, F)
        order = sorted((k for k in range(F) if not lock[k]),
                       key=lambda k: len(fovs[k].comps))
        changed = False
        for k in order:
            if time.perf_counter() > deadline:
                break
            if k >= len(fovs) or lock[k]:
                continue
            cand = [j for j in range(F)
                    if j != k and not lock[j] and fovs[j].step == fovs[k].step
                    and abs(cx[j] - cx[k]) <= 1.5 * P.fov_w
                    and abs(cy[j] - cy[k]) <= 1.5 * P.fov_h]
            if not cand:
                continue
            cand.sort(key=lambda j: max(abs(cx[j] - cx[k]), abs(cy[j] - cy[k])))
            for width in (2, 3):
                rm = set([k] + cand[:width - 1])
                if len(rm) < width:
                    continue
                rx0 = min(cx[t] for t in rm) - 1.1 * P.fov_w
                rx1 = max(cx[t] for t in rm) + 1.1 * P.fov_w
                ry0 = min(cy[t] for t in rm) - 1.1 * P.fov_h
                ry1 = max(cy[t] for t in rm) + 1.1 * P.fov_h
                st = {fovs[t].step for t in rm}
                nb = [t for t in range(F)
                      if t not in rm and fovs[t].step in st
                      and rx0 <= cx[t] <= rx1 and ry0 <= cy[t] <= ry1][:40]
                old = [fovs[t] for t in rm] + [fovs[t] for t in nb]
                oldc = _cap_of(P, old)
                freed = [c for t in rm for c in fovs[t].comps]
                keep = _clone([fovs[t] for t in nb])
                keep = _repair_free(P, keep, freed, deadline, fast=True)
                keep = _merge(P, keep, sweeps=4)
                for f in keep:
                    f.recentre()
                if _cap_of(P, keep) < oldc - 1e-9:
                    nbs = set(nb)
                    fovs = [f for t, f in enumerate(fovs)
                            if t not in rm and t not in nbs] + keep
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    return fovs


# =================================================================== sequencing
def _axis_time(d, v, a):
    th = v * v / a
    return np.where(d >= th, 2 * v / a + (d - th) / v, 2 * np.sqrt(np.maximum(d, 0) / a))


def _travel(P, x, y):
    dx = np.abs(np.diff(x)); dy = np.abs(np.diff(y))
    return np.maximum(_axis_time(dx, P.vx, P.ax), _axis_time(dy, P.vy, P.ay))


def _serpentine(P, fovs, horizontal=True):
    if not fovs:
        return []
    band = P.fov_h if horizontal else P.fov_w
    key = []
    for k, f in enumerate(fovs):
        primary = (f.y if horizontal else f.x)
        secondary = (f.x if horizontal else f.y)
        row = int(primary // band)
        key.append((row, secondary if row % 2 == 0 else -secondary, k))
    key.sort()
    return [fovs[k] for _, _, k in key]


def _nn_2opt(P, fovs, start):
    if len(fovs) <= 2:
        return list(fovs)
    x = np.array([f.x for f in fovs]); y = np.array([f.y for f in fovs])
    cost = np.maximum(_axis_time(np.abs(x[:, None] - x[None, :]), P.vx, P.ax),
                      _axis_time(np.abs(y[:, None] - y[None, :]), P.vy, P.ay))
    n = len(fovs)
    left = list(range(n))
    d0 = np.maximum(_axis_time(np.abs(x - start[0]), P.vx, P.ax),
                    _axis_time(np.abs(y - start[1]), P.vy, P.ay))
    cur = int(np.argmin(d0))
    left.remove(cur)
    tour = [cur]
    while left:
        la = np.array(left)
        cur = int(la[np.argmin(cost[cur, la])])
        left.remove(cur)
        tour.append(cur)
    # Or-opt / 2-opt sweep (bounded).  _travel_ids runs a compiled 2-opt/Or-opt
    # to convergence on every candidate order, so this Python sweep is pure
    # duplicated work when the travel seed is enabled.
    improved = P_TSEED == 0 and _FAST
    guard = 0
    while improved and guard < 6:
        improved = False
        guard += 1
        for i in range(n - 2):
            for j in range(i + 2, min(i + 24, n)):
                nxt = tour[j + 1] if j + 1 < n else None
                # classic 2-opt on the open path
                d_old = cost[tour[i], tour[i + 1]] + (cost[tour[j], tour[j + 1]] if nxt is not None else 0)
                d_new = cost[tour[i], tour[j]] + (cost[tour[i + 1], tour[j + 1]] if nxt is not None else 0)
                if d_new < d_old - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    return [fovs[k] for k in tour]


def _load_banded(P, part, bands):
    """Front-load inspection work: split by inspection load, serpentine each band."""
    if len(part) <= bands:
        return _serpentine(P, part, True)
    load = np.array([_fload(P, f) for f in part])
    order = np.argsort(-load)
    out = []
    for g in np.array_split(order, bands):
        if len(g):
            out.extend(_serpentine(P, [part[i] for i in g], True))
    return out


def _serp_band(P, part, band_x, band_y, horizontal):
    band = band_y if horizontal else band_x
    key = []
    for k, f in enumerate(part):
        prim = f.y if horizontal else f.x
        sec = f.x if horizontal else f.y
        row = int(prim // band)
        key.append((row, sec if row % 2 == 0 else -sec, k))
    key.sort()
    return [part[k] for _, _, k in key]


def _band_nn(P, part, bands, pos):
    if len(part) <= bands:
        return _nn_2opt(P, part, pos)
    load = np.array([_fload(P, f) for f in part])
    out = []
    for g in np.array_split(np.argsort(-load), bands):
        if len(g):
            sub = [part[i] for i in g]
            sub = _nn_2opt(P, sub, pos)
            out.extend(sub)
            pos = (sub[-1].x, sub[-1].y)
    return out


_POS_FREE = frozenset(('serp_h', 'serp_v', 'band2', 'band3', 'band4', 'band6',
                       'serp_h2', 'serp_v2', 'serp_h15'))


def _sequence(P, fovs, mode, step_order=None, cache=None):
    """Order FOVs: type descending, step blocks merged, spatial order inside."""
    fid = P.typ == 2
    ftype, fstep = [], []
    for f in fovs:
        ftype.append(_ftype(P, f))
        fstep.append(f.step)
    ftype = np.array(ftype); fstep = np.array(fstep)

    out = []
    pos = (0.0, 0.0)
    cur_step = None
    for t in (2, 1, 0):
        grp = [f for k, f in enumerate(fovs) if ftype[k] == t]
        if not grp:
            continue
        steps = sorted({f.step for f in grp})
        so_t = step_order.get(t) if isinstance(step_order, dict) else step_order
        if so_t is not None:
            # explicit block order: paying one 5 s Z-transition can be far
            # cheaper than letting a 25 s inspection job arrive last
            steps = [s for s in so_t if s in steps] + \
                    [s for s in steps if s not in so_t]
        elif cur_step is not None and cur_step in steps:
            steps = [cur_step] + [s for s in steps if s != cur_step]
        for s in steps:
            part = [f for f in grp if f.step == s]
            if not part:
                continue
            ck = (mode, t, s)
            if cache is not None and mode in _POS_FREE and ck in cache:
                part = cache[ck]
            elif t == 2 or len(part) <= 2:
                part = sorted(part, key=lambda f: (f.x, f.y))
            elif mode == 'serp_h':
                part = _serpentine(P, part, True)
            elif mode == 'serp_v':
                part = _serpentine(P, part, False)
            elif mode == 'band2':
                part = _load_banded(P, part, 2)
            elif mode == 'band3':
                part = _load_banded(P, part, 3)
            elif mode == 'band4':
                part = _load_banded(P, part, 4)
            elif mode == 'band6':
                part = _load_banded(P, part, 6)
            elif mode == 'serp_h2':
                part = _serp_band(P, part, P.fov_w * 0.5, P.fov_h * 0.5, True)
            elif mode == 'serp_v2':
                part = _serp_band(P, part, P.fov_w * 0.5, P.fov_h * 0.5, False)
            elif mode == 'serp_h15':
                part = _serp_band(P, part, P.fov_w * 1.5, P.fov_h * 1.5, True)
            elif mode == 'bandnn2':
                part = _band_nn(P, part, 2, pos)
            elif mode == 'bandnn3':
                part = _band_nn(P, part, 3, pos)
            elif mode == 'nn_rev':
                part = _nn_2opt(P, part, pos)[::-1]
            else:
                part = _nn_2opt(P, part, pos)
            if cache is not None and mode in _POS_FREE:
                cache[ck] = part
            out.extend(part)
            pos = (part[-1].x, part[-1].y)
            cur_step = s
    return out


# ==================================================================== simulator
P_TSEED   = _env('P_TSEED', 1, int)      # converged travel-only 2-opt/Or-opt seed
P_TROUNDS = _env('P_TROUNDS', 15, int)
P_TPASS   = _env('P_TPASS', 2, int)
P_TMAXL   = _env('P_TMAXL', 3, int)
P_TKICKS  = _env('P_TKICKS', 0, int)     # 이동 시드에 더블브리지 재시작
P_TKICKS_TOP = _env('P_TKICKS_TOP', 0, int)  # 우승 후보에만 강한 재시작
P_TSEED_TOP = _env('P_TSEED_TOP', 0, int)  # 0 = seed every mode candidate
P_TBLOCK  = _env('P_TBLOCK', 0, int)     # travel seed may cross step boundaries
P_PBLOCK  = _env('P_PBLOCK', 1, int)
P_NBPOL   = _env('P_NBPOL', 1, int)        # compiled order search
P_NBCHUNK = _env('P_NBCHUNK', 500, int)   # proposals per kernel call
P_NBREV   = _env('P_NBREV', 0.35)
P_NBSEG   = _env('P_NBSEG', 0.45)
P_NBSWP   = _env('P_NBSWP', 0.0)
P_NBSTALL = _env('P_NBSTALL', 300, int)
P_NBKICK  = _env('P_NBKICK', 1, int)
P_NBDEV   = _env('P_NBDEV', 0.0)
P_NBONLY  = _env('P_NBONLY', 0, int)     # skip the Python pre-passes in _polish
P_NBK2    = _env('P_NBK2', 1, int)       # also descend on the runner-up core count
P_NBK2_MARGIN = _env('P_NBK2_MARGIN', 8.0)
P_NBRESTART = _env('P_NBRESTART', 2, int)
P_NBSEED  = _env('P_NBSEED', 12345, int)   # -1 leaves the RNG free-running
P_NBGIVE  = _env('P_NBGIVE', 6, int)       # stop after N chunks with no gain    # polish blocks: -1 = follow P_TBLOCK


def _pblock():
    return P_TBLOCK if P_PBLOCK < 0 else P_PBLOCK


def _travel_seed(P, seq):
    """Run a *travel-only* 2-opt/Or-opt to convergence on the visit order.

    The pipeline polish evaluates every candidate move with the full simulation,
    which costs ~1000x more per move than scoring the move+Z term alone, so it
    never gets near a locally optimal tour inside its budget.  Core 0 (travel +
    Z-transition + capture) does not bind CT directly -- every practice dataset
    is inspection-bound -- but its *cumulative* time sets each FOV's release
    time, so shortening it still moves the score (zeroing travel is worth -225
    points, zeroing capture -528).
    Converging the cheap surrogate first and handing the result to _polish as a
    seed recovers most of that for ~1 ms per dataset.

    Moves stay inside a (type, step) block, so both the type-descending order
    and the Z-transition structure are preserved -- and inside a block the Z
    term is identically zero, making the surrogate exactly the travel time."""
    n = len(seq)
    if not _FAST or not P_TSEED or n < 5:
        return seq
    x = np.array([f.x for f in seq], dtype=np.float64)
    y = np.array([f.y for f in seq], dtype=np.float64)
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    C = np.ascontiguousarray(np.maximum(_axis_time(dx, P.vx, P.ax),
                                        _axis_time(dy, P.vy, P.ay)))
    if P_TBLOCK:
        # Only the type order is a hard constraint; step-contiguity is merely
        # the cheapest Z structure, not a required one.  Pricing the 5 s
        # transition into the tour lets the search pay it where the detour it
        # avoids is worth more -- exp/tsp_head.py measured 221.7 s available
        # this way against 57 s with step-contiguous blocks.
        st = np.array([f.step for f in seq], dtype=np.int64)
        C = C + Z_PENALTY * (st[:, None] != st[None, :])
        C = np.ascontiguousarray(C)
        keys = [_ftype(P, f) for f in seq]
    else:
        keys = [(_ftype(P, f), f.step) for f in seq]
    order = np.arange(n, dtype=np.int64)
    bounds, a = [], 0
    for i in range(1, n + 1):
        if i == n or keys[i] != keys[a]:
            bounds.append((a, i))
            a = i
    if not any(b - a >= 3 for a, b in bounds):
        return seq
    for _ in range(max(1, P_TPASS)):
        for (a, b) in bounds:
            if b - a < 3:
                continue
            o = np.ascontiguousarray(order[a:b])
            pred = np.int64(order[a - 1]) if a > 0 else np.int64(-1)
            succ = np.int64(order[b]) if b < n else np.int64(-1)
            fastops.path_opt(o, C, pred, succ, P_TROUNDS, P_TMAXL)
            order[a:b] = o
    return [seq[i] for i in order]


def _travel_ids(P, ev, ftyp, ids, kicks=None):
    """_travel_seed on an index order, reusing one prebuilt SeqEval.

    Building a SeqEval costs O(n^2) for the move matrix; the mode search scores
    24 candidate orders, so sharing one instance across all of them (and with
    the polish) removes 24 matrix builds per dataset."""
    n = len(ids)
    if kicks is None:
        kicks = P_TKICKS
    if not _FAST or not P_TSEED or n < 5:
        return np.asarray(ids, dtype=np.int64)
    C = ev.mvf
    st = ev.stpi
    if P_TBLOCK:
        C = np.ascontiguousarray(C + Z_PENALTY * (st[:, None] != st[None, :]))
        keys = [int(ftyp[i]) for i in ids]
    else:
        keys = [(int(ftyp[i]), int(st[i])) for i in ids]
    order = np.array(ids, dtype=np.int64)
    bounds, a = [], 0
    for i in range(1, n + 1):
        if i == n or keys[i] != keys[a]:
            bounds.append((a, i))
            a = i
    for _ in range(max(1, P_TPASS)):
        for (a, b) in bounds:
            if b - a < 3:
                continue
            o = np.ascontiguousarray(order[a:b])
            pred = np.int64(order[a - 1]) if a > 0 else np.int64(-1)
            succ = np.int64(order[b]) if b < n else np.int64(-1)
            if kicks:
                fastops.path_opt_ils(o, C, pred, succ, P_TROUNDS, P_TMAXL,
                                     kicks, 20260921 + a)
            else:
                fastops.path_opt(o, C, pred, succ, P_TROUNDS, P_TMAXL)
            order[a:b] = o
    return order


def blocks_from(ftyp, stp, ids):
    out, cur, run = [], None, 0
    for i in ids:
        key = int(ftyp[i]) if _pblock() else (int(ftyp[i]), int(stp[i]))
        if key != cur:
            if run:
                out.append(run)
            cur, run = key, 1
        else:
            run += 1
    if run:
        out.append(run)
    return out


P_CENTRE = _env('P_CENTRE', 1, int)      # slide FOV centres inside their slack
P_CENTRE_PASS = _env('P_CENTRE_PASS', 3, int)
P_CENTRE_REPOL = _env('P_CENTRE_REPOL', 1, int)
P_CENTRE_ROUNDS = _env('P_CENTRE_ROUNDS', 3, int)


def _centre_opt(P, seq, passes=None):
    """Slide every FOV centre inside its feasible box to shorten the tour.

    A FOV centre is free anywhere in [x1,x2] x [y1,y2] -- the box is exactly the
    set of positions that still cover the FOV's components -- but the pipeline
    parks it at the box centre.  Travel is 875 s of core 0's load; core 0 never
    binds CT on its own, but its cumulative time sets the inspection release
    times, so the slack is still worth spending.  Axis
    travel time is monotone in |d|, so the optimal coordinate is one of the box
    ends or a clamped neighbour coordinate; 3 coordinate-descent passes over
    those candidates converge."""
    if not P_CENTRE:
        return 0.0
    n = len(seq)
    if n < 3:
        return 0.0
    passes = P_CENTRE_PASS if passes is None else passes
    # the true slack is the intersection of R_c over the components the FOV
    # actually holds -- recomputing it here also picks up FOVs whose stored box
    # was never updated (fiducial FOVs are created degenerate) and is the exact
    # feasibility test, so no move can ever uncover a component
    x = np.empty(n); y = np.empty(n)
    bx1 = np.empty(n); bx2 = np.empty(n)
    by1 = np.empty(n); by2 = np.empty(n)
    ok = np.zeros(n, dtype=np.bool_)
    for t, f in enumerate(seq):
        x[t] = f.x; y[t] = f.y
        bx1[t] = bx2[t] = f.x
        by1[t] = by2[t] = f.y
        cs = f.comps
        if not cs or P.big[cs].any():
            continue                      # big-component tile: position is forced
        a1 = float(P.rx1[cs].max()); a2 = float(P.rx2[cs].min())
        b1 = float(P.ry1[cs].max()); b2 = float(P.ry2[cs].min())
        if a2 - a1 > 1e-9 or b2 - b1 > 1e-9:
            bx1[t], bx2[t], by1[t], by2[t] = a1, a2, b1, b2
            ok[t] = True
    if not ok.any():
        return 0.0
    gain = float(fastops.centre_kernel(x, y, bx1, bx2, by1, by2, ok,
                                       P.vx, P.ax, P.vy, P.ay, max(1, passes)))
    if gain > 1e-9:
        for t, f in enumerate(seq):
            if ok[t]:
                f.x = float(x[t]); f.y = float(y[t])
    return gain


P_REASSIGN = _env('P_REASSIGN', 1, int)
P_RA_TAIL  = _env('P_RA_TAIL', 1.0)     # fraction of the sequence to scan
P_RA_TRIES = _env('P_RA_TRIES', 4000, int)
P_RA_HOSTS = _env('P_RA_HOSTS', 4, int)
P_FIN_ROUNDS = _env('P_FIN_ROUNDS', 10, int)


def _reassign_seq(P, seq, deadline, rng):
    """Relabel components onto other FOVs that already cover them.

    Which FOV *lists* a component is free whenever more than one FOV's window
    contains it, and the total inspection work is unchanged either way (the
    checker charges the component once per listing).  What does change is when
    that work reaches the inspection cores: work sitting in the last FOV is pure
    tail, and work sitting too late starves the cores.  Only relabelling is
    attempted -- no FOV centre moves -- so feasibility is immediate, and every
    candidate move is scored with the exact pipeline simulation."""
    n = len(seq)
    if not P_REASSIGN or n < 4:
        return seq, False
    ev = SeqEval(P, seq)
    ids = np.arange(n, dtype=np.int64)
    base, k = ev.score(ids)
    _, states = ev.run(ids, k, keep=True)
    cx = np.array([f.x for f in seq])
    cy = np.array([f.y for f in seq])
    fst = np.array([f.step for f in seq])
    lo = int(n * (1.0 - min(max(P_RA_TAIL, 0.0), 1.0)))
    tries = 0
    changed = False
    for t in range(n - 1, lo - 1, -1):
        if time.perf_counter() > deadline or tries > P_RA_TRIES:
            break
        f = seq[t]
        for c in list(f.comps):
            if len(f.comps) <= 0 or P.big[c] or P.typ[c] == 2:
                continue
            m = ((fst == P.step[c]) &
                 (cx - P.hw[c] <= P.tlx[c] + EPS) & (cy - P.hh[c] <= P.tly[c] + EPS) &
                 (cx + P.hw[c] >= P.brx[c] - EPS) & (cy + P.hh[c] >= P.bry[c] - EPS))
            m[t] = False
            hosts = np.where(m)[0]
            if len(hosts) == 0:
                continue
            # earliest hosts first: that is where the work is most useful
            for h in hosts[np.argsort(hosts)][:P_RA_HOSTS]:
                h = int(h)
                tries += 1
                if tries > P_RA_TRIES or time.perf_counter() > deadline:
                    break
                oldf = list(seq[t].comps)
                oldh = list(seq[h].comps)
                newf = [q for q in oldf if q != c]
                # a FOV's type is the max type of what it lists, and the visit
                # order must stay type-descending, so a move may not change
                # either FOV's type
                if int(P.typ[c]) > _ftype(P, seq[h]):
                    continue
                if newf and max(int(P.typ[q]) for q in newf) != _ftype(P, seq[t]):
                    continue
                newh = sorted(oldh + [int(c)], key=lambda q: -P.time[q])
                ev.set_fov(t, newf)
                ev.set_fov(h, newh)
                # only positions from min(t, h) on can change, so the simulation
                # restarts from the cached prefix state instead of position 0
                m = t if t < h else h
                if m == 0:
                    v = ev.run(ids, k)[0]
                else:
                    v = ev.run(ids, k, start=m, state=states[m - 1])[0]
                v += BETA_CORE * k
                if v < base - 1e-9:
                    seq[t].comps = newf
                    seq[h].comps = newh
                    _touch(seq[t]); _touch(seq[h])
                    base = v
                    changed = True
                    _, states = ev.run(ids, k, keep=True)
                    break
                ev.set_fov(t, oldf)
                ev.set_fov(h, oldh)
    if changed:
        seq = [f for f in seq if f.comps]
    return seq, changed


P_EVAC = _env('P_EVAC', 0, int)
P_EVAC_N = _env('P_EVAC_N', 10, int)     # how many FOVs to try to dissolve


def _evacuate(P, seq, deadline):
    """Try to dissolve whole FOVs by relabelling *all* of their components.

    Emptying a FOV is a compound move: no single component transfer improves the
    schedule, but removing the FOV altogether saves its capture (0.5-0.75 s), a
    hop, possibly a 5 s Z-transition, and -- when it is at the end -- the whole
    inspection tail that follows the last reconstruction.  Candidates are tried
    from the back, where the tail is, and from the lightest FOVs, which are the
    easiest to place."""
    n = len(seq)
    if not P_EVAC or n < 4:
        return seq, False
    ev = SeqEval(P, seq)
    ids = np.arange(n, dtype=np.int64)
    base, k = ev.score(ids)
    cx = np.array([f.x for f in seq])
    cy = np.array([f.y for f in seq])
    fst = np.array([f.step for f in seq])
    load = np.array([_fload(P, f) for f in seq])
    ncomp = np.array([len(f.comps) for f in seq])
    cand = list(range(n - 1, max(0, n - 4) - 1, -1))
    for t in np.argsort(ncomp):
        t = int(t)
        if t not in cand:
            cand.append(t)
        if len(cand) >= P_EVAC_N:
            break
    changed = False
    for t in cand:
        if time.perf_counter() > deadline:
            break
        f = seq[t]
        if not f.comps or _fid_in(P, f) or any(P.big[c] for c in f.comps):
            continue
        plan = []
        ok = True
        for c in f.comps:
            m = ((fst == P.step[c]) &
                 (cx - P.hw[c] <= P.tlx[c] + EPS) & (cy - P.hh[c] <= P.tly[c] + EPS) &
                 (cx + P.hw[c] >= P.brx[c] - EPS) & (cy + P.hh[c] >= P.bry[c] - EPS))
            m[t] = False
            hosts = np.where(m)[0]
            if len(hosts) == 0:
                ok = False
                break
            # cheapest host: one that already carries a side component when c is
            # a side component, else the least loaded
            best, bv = -1, None
            for h in hosts:
                h = int(h)
                if int(P.typ[c]) > _ftype(P, seq[h]):
                    continue
                v = (0 if (P.side[c] == 0 or _has_side(P, seq[h])) else 1, load[h])
                if bv is None or v < bv:
                    bv, best = v, h
            if best < 0:
                ok = False
                break
            plan.append((int(c), best))
        if not ok:
            continue
        saved = [list(g.comps) for g in seq]
        for c, h in plan:
            seq[h].comps = sorted(seq[h].comps + [c], key=lambda q: -P.time[q])
            _touch(seq[h])
        seq[t].comps = []
        _touch(seq[t])
        trial = [g for g in seq if g.comps]
        ty = [_ftype(P, g) for g in trial]
        good = not any(ty[q] < ty[q + 1] for q in range(len(ty) - 1))
        v = evaluate(P, trial)[0] if good else None
        if good and v < base - 1e-9:
            base = v
            changed = True
            seq = trial
            n = len(seq)
            ev = SeqEval(P, seq)
            ids = np.arange(n, dtype=np.int64)
            cx = np.array([g.x for g in seq])
            cy = np.array([g.y for g in seq])
            fst = np.array([g.step for g in seq])
            load = np.array([_fload(P, g) for g in seq])
            break
        for g, cs in zip(seq, saved):
            g.comps = cs
            _touch(g)
    return seq, changed


P_FLOOR_EPS = _env('P_FLOOR_EPS', 0.0)   # stop polishing this close to the bound


def _seq_bound(P, ev, ids, k):
    """Lower bound on CT for this FOV set and visit order.

    Core 0 must move, switch step level and capture every FOV in turn, and the
    inspection cores cannot finish before the total work divided by k-3; the
    last FOV's reconstruction and its longest job come after either of those.
    No re-ordering can beat this, so once CT reaches it the search is done."""
    a = np.asarray(ids, dtype=np.int64)
    c0 = float(ev.timgf[a].sum())
    if len(a) > 1:
        c0 += float(ev.mvf[a[:-1], a[1:]].sum())
        c0 += Z_PENALTY * float((ev.stpi[a[:-1]] != ev.stpi[a[1:]]).sum())
    work = float(ev.load.sum())
    insp = work / max(k - 3, 1)
    tail = P.recon + float(ev.peak[a[-1]]) if len(a) else 0.0
    return max(c0 + tail, insp + ev.timgf[a[0]] + P.recon if len(a) else 0.0)


P_JOBORD = _env('P_JOBORD', 0, int)
P_JOB_N = _env('P_JOB_N', 12, int)       # how many trailing FOVs to re-order
P_JOB_TRIES = _env('P_JOB_TRIES', 900, int)
P_TILEFILL = _env('P_TILEFILL', 1, int)   # let big-component tiles carry ordinary components


def _job_order(P, seq, deadline, rng):
    """Re-order the components *inside* a FOV's comp_idx list.

    The checker dispatches a FOV's components to inspection cores strictly in
    the order they are listed, each to whichever core is free first.  Longest-
    processing-time first is a good rule but not an optimal one, and for the
    FOVs at the end of the run the resulting imbalance is the tail that decides
    the cycle time.  Trailing FOVs are re-ordered first because a change there
    re-simulates almost nothing."""
    n = len(seq)
    if not P_JOBORD or n < 2:
        return False
    ev = SeqEval(P, seq)
    ids = np.arange(n, dtype=np.int64)
    base, k = ev.score(ids)
    _, states = ev.run(ids, k, keep=True)
    changed = False
    tries = 0
    for t in range(n - 1, max(0, n - P_JOB_N) - 1, -1):
        cs = seq[t].comps
        m = len(cs)
        if m < 3 or time.perf_counter() > deadline or tries > P_JOB_TRIES:
            break
        cur = list(cs)
        improved = True
        while improved and tries <= P_JOB_TRIES and time.perf_counter() <= deadline:
            improved = False
            for _ in range(min(60, m * 4)):
                tries += 1
                i = int(rng.integers(m))
                j = int(rng.integers(m))
                if i == j:
                    continue
                trial = list(cur)
                trial.insert(j, trial.pop(i))
                ev.set_fov(t, trial)
                if t == 0:
                    v = ev.run(ids, k)[0]
                else:
                    v = ev.run(ids, k, start=t, state=states[t - 1])[0]
                v += BETA_CORE * k
                if v < base - 1e-9:
                    base = v
                    cur = trial
                    changed = True
                    improved = True
                    _, states = ev.run(ids, k, keep=True)
                else:
                    ev.set_fov(t, cur)
        if cur != list(cs):
            seq[t].comps = cur
            _touch(seq[t])
    return changed


P_RUNMOVE = _env('P_RUNMOVE', 1, int)
P_RUN_TRIES = _env('P_RUN_TRIES', 1200, int)
P_ROUND_EPS = _env('P_ROUND_EPS', 0.1)   # stop the improvement loop below this gain
P_SLACKW = _env('P_SLACKW', 0.03)         # tie-break the cover toward roomier FOVs


def _run_move(P, seq, deadline):
    """Relocate whole runs of same-step FOVs.

    Each place where the step level changes costs core 0 a flat 5 s, and core 0
    is the bottleneck on most datasets.  The random relocate/Or-opt moves shift
    at most three FOVs at a time, so they can never merge two long same-step
    runs -- that takes moving the whole run at once, which also removes two
    transitions instead of creating one."""
    n = len(seq)
    if not P_RUNMOVE or n < 4:
        return False
    ev = SeqEval(P, seq)
    ids = list(range(n))
    base, k = ev.score(np.asarray(ids, np.int64))
    _, states = ev.run(np.asarray(ids, np.int64), k, keep=True)
    ftyp = [_ftype(P, f) for f in seq]
    spans, a = [], 0
    for i in range(1, n + 1):
        if i == n or ftyp[i] != ftyp[a]:
            spans.append((a, i))
            a = i
    tries = 0
    changed = True
    any_change = False
    while changed and tries < P_RUN_TRIES and time.perf_counter() < deadline:
        changed = False
        stp = [seq[q].step for q in ids]
        runs = []
        for (lo, hi) in spans:
            a = lo
            for i in range(lo + 1, hi + 1):
                if i == hi or stp[i] != stp[a]:
                    runs.append((a, i, lo, hi))
                    a = i
        for (ra, rb, lo, hi) in runs:
            if tries >= P_RUN_TRIES or time.perf_counter() > deadline:
                break
            L = rb - ra
            if L >= hi - lo:
                continue
            seg = ids[ra:rb]
            rest = ids[:ra] + ids[rb:]
            stride = 1 if hi - lo - L <= 24 else max(1, (hi - lo - L) // 24)
            for p in range(lo, hi - L + 1, stride):
                if p == ra:
                    continue
                tries += 1
                for sg in (seg, seg[::-1]):
                    trial = rest[:p] + sg + rest[p:]
                    m = min(ra, p)
                    if m == 0:
                        v = ev.run(np.asarray(trial, np.int64), k)[0]
                    else:
                        v = ev.run(np.asarray(trial, np.int64), k, start=m,
                                   state=states[m - 1])[0]
                    v += BETA_CORE * k
                    if v < base - 1e-9:
                        base = v
                        ids = trial
                        _, states = ev.run(np.asarray(ids, np.int64), k, keep=True)
                        changed = True
                        any_change = True
                        break
                if changed:
                    break
            if changed:
                break
    if any_change:
        seq[:] = [seq[q] for q in ids]
    return any_change


P_SLACKPASS = _env('P_SLACKPASS', 0, int)


def _slack_pass(P, seq):
    """Hand away the components that pin a FOV's centre in place.

    A FOV may sit anywhere in the intersection of R_c over the components it
    lists, and that intersection is decided by at most four of them -- the ones
    touching each edge.  Re-listing such a component on a neighbouring FOV whose
    centre already covers it costs nothing (total inspection work is unchanged)
    and can free a lot of travel slack for the centre pass.  Only the binding
    components are examined, so the pass is a few hundred tests per dataset."""
    n = len(seq)
    if not P_SLACKPASS or n < 2:
        return False
    cx = np.array([f.x for f in seq])
    cy = np.array([f.y for f in seq])
    fst = np.array([f.step for f in seq])
    ftp = np.array([_ftype(P, f) for f in seq])

    def box(cs):
        if not cs:
            return None
        a = np.asarray(cs, dtype=np.int64)
        return (float(P.rx1[a].max()), float(P.rx2[a].min()),
                float(P.ry1[a].max()), float(P.ry2[a].min()))

    moved = False
    for t in range(n):
        f = seq[t]
        cs = f.comps
        if len(cs) < 2 or f.fixed:
            continue
        b = box(cs)
        if b is None:
            continue
        a = np.asarray(cs, dtype=np.int64)
        if P.big[a].any():
            continue
        span = (b[1] - b[0]) + (b[3] - b[2])
        binding = set()
        for arr, val, lo in ((P.rx1[a], b[0], True), (P.rx2[a], b[1], False),
                             (P.ry1[a], b[2], True), (P.ry2[a], b[3], False)):
            binding.add(int(a[int(np.argmax(arr)) if lo else int(np.argmin(arr))]))
        for c in binding:
            rest = [q for q in seq[t].comps if q != c]
            if not rest:
                continue
            if max(int(P.typ[q]) for q in rest) != int(ftp[t]):
                continue
            nb_ = box(rest)
            newspan = (nb_[1] - nb_[0]) + (nb_[3] - nb_[2])
            if newspan <= span + 1e-9:
                continue
            m = ((fst == P.step[c]) & (ftp >= int(P.typ[c])) &
                 (cx - P.hw[c] <= P.tlx[c] + EPS) & (cy - P.hh[c] <= P.tly[c] + EPS) &
                 (cx + P.hw[c] >= P.brx[c] - EPS) & (cy + P.hh[c] >= P.bry[c] - EPS))
            m[t] = False
            best, bgain = -1, 1e-9
            for h in np.where(m)[0]:
                h = int(h)
                if seq[h].fixed:
                    continue
                hb = box(seq[h].comps)
                if hb is None:
                    continue
                hs = (hb[1] - hb[0]) + (hb[3] - hb[2])
                nh = box(seq[h].comps + [c])
                if nh[1] < nh[0] - 1e-9 or nh[3] < nh[2] - 1e-9:
                    continue
                nhs = (nh[1] - nh[0]) + (nh[3] - nh[2])
                # a side component must not turn a 0.50 s capture into 0.75 s
                if P.side[c] == 1 and not _has_side(P, seq[h]):
                    continue
                g = (newspan - span) + (nhs - hs)
                if g > bgain:
                    bgain, best = g, h
            if best >= 0:
                seq[t].comps = rest
                seq[best].comps = sorted(seq[best].comps + [int(c)],
                                         key=lambda q: -P.time[q])
                _touch(seq[t]); _touch(seq[best])
                moved = True
                break
    return moved


P_MARGIN_ASSIGN = _env('P_MARGIN_ASSIGN', 0, int)


def _margin_assign(P, fovs):
    """Re-list every component on the FOV whose centre covers it most deeply.

    Which FOV lists a component is free whenever several cover it, and the total
    inspection work is identical either way.  What differs is each FOV's own
    slack -- the intersection of R over what it lists -- and that slack is what
    the centre pass spends on shortening core 0's tour.  The same rule inside
    assign_sidewise was worth 7 score points, so it is applied to the whole plan
    once the cover is final.  Side components only move to FOVs that already
    hold one, so no 0.50 s capture becomes a 0.75 s capture."""
    F = len(fovs)
    if not P_MARGIN_ASSIGN or F < 2:
        return False
    cx = np.fromiter((f.x for f in fovs), float, F)
    cy = np.fromiter((f.y for f in fovs), float, F)
    fst = np.fromiter((f.step for f in fovs), np.int64, F)
    hasside = np.array([_has_side(P, f) for f in fovs], dtype=bool)
    fixed = np.array([f.fixed or _fid_in(P, f) for f in fovs], dtype=bool)
    owner, pool = [], []
    for i, f in enumerate(fovs):
        for c in f.comps:
            if P.big[c] or P.typ[c] == 2:
                continue
            pool.append(int(c))
            owner.append(i)
    if not pool:
        return False
    cid = np.asarray(pool, dtype=np.int64)
    own = np.asarray(owner, dtype=np.int64)
    dx1 = cx[:, None] - (P.brx[cid] - P.hw[cid])[None, :]
    dx2 = (P.tlx[cid] + P.hw[cid])[None, :] - cx[:, None]
    dy1 = cy[:, None] - (P.bry[cid] - P.hh[cid])[None, :]
    dy2 = (P.tly[cid] + P.hh[cid])[None, :] - cy[:, None]
    mg = np.minimum(np.minimum(dx1, dx2), np.minimum(dy1, dy2))
    ok = (mg >= -EPS) & (fst[:, None] == P.step[cid][None, :])
    side = P.side[cid] == 1
    if side.any():
        ok[:, side] &= hasside[:, None]
    ok[fixed, :] = False                      # a fixed FOV cannot grow its box
    ok[own, np.arange(len(cid))] = True       # the current owner always stays legal
    mg = np.where(ok, mg, -np.inf)
    pick = mg.argmax(0)
    if (pick == own).all():
        return False
    groups = [[] for _ in range(F)]
    for i, f in enumerate(fovs):
        for c in f.comps:
            if P.big[c] or P.typ[c] == 2:
                groups[i].append(int(c))
    for i in range(F):
        groups[i] = [c for c in groups[i] if P.big[c] or P.typ[c] == 2]
    for t in range(len(cid)):
        groups[int(pick[t])].append(int(cid[t]))
    for i, f in enumerate(fovs):
        g = groups[i]
        if not g:
            f.comps = []
            _touch(f)
            continue
        f.comps = sorted(set(g), key=lambda q: -P.time[q])
        _touch(f)
        a = np.asarray(f.comps, dtype=np.int64)
        if not P.big[a].any():
            f.x1 = float(P.rx1[a].max()); f.x2 = float(P.rx2[a].min())
            f.y1 = float(P.ry1[a].max()); f.y2 = float(P.ry2[a].min())
    return True


def _sim(P, seq, t_img, t_comp, steps, xs, ys, k):
    """Exact re-implementation of Checker.get_real_cost (single k)."""
    n = len(seq)
    tl = [t_img[0]] * k
    rc = 1 if tl[1] <= tl[2] else 2
    tl[rc] += P.recon
    sel = tl[rc]
    for d in t_comp[0]:
        best, bv = 3, max(tl[3], sel)
        for c in range(4, k):
            v = tl[c] if tl[c] > sel else sel
            if v < bv:
                bv, best = v, c
        tl[best] = bv + d
    if n > 1:
        mv = _travel(P, xs, ys)
        for i in range(n - 1):
            t0 = tl[0] + mv[i]
            if steps[i] != steps[i + 1]:
                t0 += Z_PENALTY
            t0 += t_img[i + 1]
            tl[0] = t0
            for c in range(1, k):
                if tl[c] < t0:
                    tl[c] = t0
            rc = 1 if tl[1] <= tl[2] else 2
            tl[rc] += P.recon
            sel = tl[rc]
            for d in t_comp[i + 1]:
                best, bv = 3, (tl[3] if tl[3] > sel else sel)
                for c in range(4, k):
                    v = tl[c] if tl[c] > sel else sel
                    if v < bv:
                        bv, best = v, c
                tl[best] = bv + d
    return max(tl)


class SeqEval:
    """Pre-computes per-FOV timing data and caches simulation prefixes so that
    re-ordering can be re-scored incrementally."""

    def __init__(self, P, fovs):
        self.P = P
        self.x = np.array([f.x for f in fovs])
        self.y = np.array([f.y for f in fovs])
        self.timg = np.array([P.side_capture if _has_side(P, f) else P.capture
                              for f in fovs])
        self.tcomp = [np.asarray([P.time[c] for c in f.comps], dtype=np.float64)
                      for f in fovs]
        self.step = np.array([f.step for f in fovs])
        self.load = np.array([float(t.sum()) if len(t) else 0.0 for t in self.tcomp])
        self.peak = np.array([float(t.max()) if len(t) else 0.0 for t in self.tcomp])
        dx = np.abs(self.x[:, None] - self.x[None, :])
        dy = np.abs(self.y[:, None] - self.y[None, :])
        self.mv = np.maximum(_axis_time(dx, P.vx, P.ax), _axis_time(dy, P.vy, P.ay))
        self._flat()

    def _flat(self):
        """Per-FOV component times in a padded 2-D array.

        A flat CSR has to be rebuilt whenever a component moves between FOVs,
        which the relabelling pass does thousands of times per dataset; a padded
        row layout makes that an O(row) in-place edit and gives the simulator
        better locality as well."""
        m = len(self.tcomp)
        lens = np.fromiter((len(t) for t in self.tcomp), np.int64, m) if m \
            else np.zeros(0, np.int64)
        cap = int(lens.max()) + 4 if m and lens.max() else 4
        self.tc2 = np.zeros((max(m, 1), cap))
        for i, t in enumerate(self.tcomp):
            if len(t):
                self.tc2[i, :len(t)] = t
        self.tcn = lens
        self.stpi = self.step.astype(np.int64)
        self.timgf = self.timg.astype(np.float64)
        self.mvf = np.ascontiguousarray(self.mv, dtype=np.float64)

    def set_fov(self, i, comps):
        P = self.P
        ix = np.asarray(comps, dtype=np.int64)
        t = P.time[ix] if len(ix) else np.empty(0)
        self.timg[i] = (P.side_capture if len(ix) and (P.side[ix] == 1).any()
                        else P.capture)
        self.timgf[i] = self.timg[i]
        self.tcomp[i] = t
        self.load[i] = float(t.sum()) if len(t) else 0.0
        self.peak[i] = float(t.max()) if len(t) else 0.0
        if len(t) > self.tc2.shape[1]:
            g = np.zeros((self.tc2.shape[0], len(t) + 4))
            g[:, :self.tc2.shape[1]] = self.tc2
            self.tc2 = g
        self.tc2[i, :] = 0.0
        if len(t):
            self.tc2[i, :len(t)] = t
        self.tcn[i] = len(t)

    def _inspect(self, tl, k, sel, jobs):
        for d in jobs:
            best, bv = 3, (tl[3] if tl[3] > sel else sel)
            for c in range(4, k):
                v = tl[c] if tl[c] > sel else sel
                if v < bv:
                    bv, best = v, c
            tl[best] = bv + d

    def run(self, ids, k, start=0, state=None, keep=False):
        if _FAST:
            a = ids if isinstance(ids, np.ndarray) else np.asarray(ids, dtype=np.int64)
            st = (np.empty((len(a), k)) if keep else self._SC)
            tl0 = (state if isinstance(state, np.ndarray)
                   else (np.asarray(state, dtype=np.float64) if state is not None
                         else np.zeros(k)))
            mx = fastops.sim_kernel(a, k, self.timgf, self.stpi, self.mvf,
                                    self.tc2, self.tcn, self.P.recon, Z_PENALTY,
                                    start, tl0, st, keep)
            return float(mx), (st if keep else None)
        return self._run_py(ids, k, start, state, keep)

    _SC = np.zeros((1, 1))

    def _run_py(self, ids, k, start=0, state=None, keep=False):
        """Simulate from position `start` (state = t_last after position start-1).
        Returns (ct, states) where states[p] is the t_last list after position p."""
        P = self.P
        n = len(ids)
        states = [None] * n if keep else None
        if start == 0:
            f0 = ids[0]
            tl = [self.timg[f0]] * k
            rc = 1 if tl[1] <= tl[2] else 2
            tl[rc] += P.recon
            self._inspect(tl, k, tl[rc], self.tcomp[f0])
            if keep:
                states[0] = list(tl)
            p = 1
        else:
            tl = list(state)
            p = start
        mv = self.mv
        timg = self.timg
        st = self.step
        for i in range(p, n):
            a, b = ids[i - 1], ids[i]
            t0 = tl[0] + mv[a, b]
            if st[a] != st[b]:
                t0 += Z_PENALTY
            t0 += timg[b]
            tl[0] = t0
            for c in range(1, k):
                if tl[c] < t0:
                    tl[c] = t0
            rc = 1 if tl[1] <= tl[2] else 2
            tl[rc] += P.recon
            self._inspect(tl, k, tl[rc], self.tcomp[b])
            if keep:
                states[i] = list(tl)
        return max(tl), states

    def ct(self, ids, k):
        return self.run(ids, k)[0]

    def score(self, ids):
        """min_k (CT + 10k) -- identical to Checker.compute_ct, with an exact
        bound so the k-sweep can stop early."""
        mc = self.P.max_core
        hi = self.ct(ids, mc)             # best attainable CT (most cores)
        best = (hi + BETA_CORE * mc, mc)
        for k in range(4, mc):
            if hi + BETA_CORE * k >= best[0]:
                break
            v = self.ct(ids, k) + BETA_CORE * k
            if v < best[0]:
                best = (v, k)
        return best


def _polish(ev, ids, blocks, deadline, rng, seeds=True, alt=None):
    """Hill-climb the visit order.  Moves stay inside a type/step block so the
    type-descending order and the minimal Z-transition structure are preserved."""
    best_s, k = ev.score(ids)
    since_k = 0
    cur = list(ids)
    n = len(cur)
    spans, o = [], 0
    for b in blocks:
        spans.append((o, o + b))
        o += b
    cur_ct, states = ev.run(cur, k, keep=True)
    best_c = cur_ct + BETA_CORE * k

    def _accept(trial, m):
        nonlocal cur, best_c, states, k, since_k
        if m == 0:
            c = ev.run(trial, k)[0]
        else:
            c = ev.run(trial, k, start=m, state=states[m - 1])[0]
        c += BETA_CORE * k
        if c < best_c - 1e-9:
            cur = trial
            best_c = c
            since_k += 1
            if P_KREFRESH and since_k >= P_KREFRESH:
                # CT has moved; the core count that minimises CT+10k may have
                # moved with it, and polishing against a stale k optimises the
                # wrong objective.
                since_k = 0
                v2, k2 = ev.score(cur)
                if k2 != k:
                    k = k2
                    best_c = v2
            _, states = ev.run(cur, k, keep=True)
            return True
        return False

    def try_move(i, j):
        if i == j:
            return False
        trial = list(cur)
        trial.insert(j, trial.pop(i))
        return _accept(trial, min(i, j))

    def try_seg(i, L, j, lo, hi):
        """Or-opt: relocate a run of L consecutive FOVs, staying inside [lo, hi).

        The insertion index has to be clamped to the block: after deleting the
        run, every position above i shifts down by L, and an unclamped index can
        land in the previous block and break the type-descending order."""
        if i + L > hi or j == i:
            return False
        trial = list(cur)
        seg = trial[i:i + L]
        del trial[i:i + L]
        k2 = j if j < i else j - L
        k2 = max(lo, min(k2, hi - L))
        if k2 == i:
            return False
        trial[k2:k2] = seg
        return _accept(trial, min(i, k2))

    def try_rev(i, j):
        """2-opt segment reversal -- the classic move for the travel term."""
        if j <= i + 1:
            return False
        trial = list(cur)
        trial[i:j + 1] = trial[i:j + 1][::-1]
        return _accept(trial, i)

    # --- deterministic seeds: light FOVs to the tail, heavy ones to the front
    for (a, b) in ([] if (not seeds or (P_NBONLY and _FAST and P_NBPOL)) else spans):
        if b - a < 3 or time.perf_counter() > deadline:
            continue
        load = ev.load[[cur[t] for t in range(a, b)]]
        peak = ev.peak[[cur[t] for t in range(a, b)]]
        for t in np.argsort(load)[:6]:                 # lightest to the tail
            if try_move(a + int(t), b - 1):
                load = ev.load[[cur[q] for q in range(a, b)]]
        for t in np.argsort(-load)[:6]:                # heaviest to the front
            if try_move(a + int(t), a):
                load = ev.load[[cur[q] for q in range(a, b)]]
        # the single longest component decides the tail of a saturated core,
        # so FOVs holding a big job get their own attempt at the front
        for t in np.argsort(-peak)[:8]:
            if try_move(a + int(t), a):
                peak = ev.peak[[cur[q] for q in range(a, b)]]

    # --- 2-opt sweep over each block (bounded segment length): travel term
    for (a, b) in ([] if (not seeds or (P_NBONLY and _FAST and P_NBPOL)) else spans):
        w = b - a
        if w < 4:
            continue
        for i in range(a, b - 2):
            if time.perf_counter() > deadline:
                break
            for j in range(i + 2, min(i + 14, b)):
                try_rev(i, j)

    # --- compiled randomised descent + double-bridge restarts ---------------
    if _FAST and P_NBPOL:
        start = np.asarray(cur, dtype=np.int64)
        gb = np.empty_like(start)
        bs = np.array([a for a, _ in spans], dtype=np.int64)
        be = np.array([b for _, b in spans], dtype=np.int64)
        blockof = np.zeros(n, dtype=np.int64)
        for _bi, (_a, _b) in enumerate(spans):
            blockof[_a:_b] = _bi
        # The checker keeps min_k (CT_k + 10k).  Descending on CT at one k can
        # settle into an order that is good there and mediocre at the runner-up
        # core count, and the runner-up is within a few seconds of winning on
        # 23 of 96 datasets -- so when it is close, part of the budget descends
        # on it instead and the two candidates are compared on the true score.
        # Two descents from the same seed, at the best core count and at its
        # nearest rival, measured better than one long descent: the split acts
        # as a restart and the two orders are compared on the true min_k score.
        # Only k1-1 / k1+1 are probed -- ranking every core count costs one full
        # simulation each and bought nothing.
        _, k = ev.score(start)
        plan_k = [k]
        if P_NBK2:
            mc = ev.P.max_core
            v0 = ev.ct(start, k) + BETA_CORE * k
            altk, av = -1, None
            for q in (k - 1, k + 1):
                if 4 <= q <= mc:
                    vq = ev.ct(start, q) + BETA_CORE * q
                    if av is None or vq < av:
                        av, altk = vq, q
            if altk > 0 and av - v0 < P_NBK2_MARGIN:
                plan_k.append(altk)
        if P_NBRESTART > len(plan_k):
            # more independent descents from the same seed, cycling the core
            # counts; the split is a restart mechanism, and restarts beat one
            # long descent because the walk saturates early
            base_k = list(plan_k)
            while len(plan_k) < P_NBRESTART:
                plan_k.append(base_k[len(plan_k) % len(base_k)])
        bestv, bestarr = None, start.copy()
        t_now = time.perf_counter()
        share = (deadline - t_now) / len(plan_k)
        for qi, kk in enumerate(plan_k):
            # the second descent may start from a different candidate order --
            # same FOV set and the same type blocks, so the spans still hold
            arr = (np.asarray(alt, dtype=np.int64).copy()
                   if (qi == 1 and alt is not None and len(alt) == n)
                   else start.copy())
            sub = min(deadline, t_now + share * (qi + 1))
            k = kk
            idle = 0
            _nbseed = -1 if P_NBSEED < 0 else (P_NBSEED + 7919 * qi + 104729 * n)
            while time.perf_counter() < sub:
                stx = np.zeros((n, k))
                _nbseed = -1 if P_NBSEED < 0 else (_nbseed + 1)
                fastops.polish_kernel(arr, gb, k, ev.timgf, ev.stpi, ev.mvf,
                                      ev.tc2, ev.tcn, ev.P.recon, Z_PENALTY,
                                      bs, be, blockof, stx, P_NBCHUNK,
                                      P_NBREV, P_NBSEG, P_NBSWP, P_NBSTALL,
                                      P_NBKICK, P_NBDEV, _nbseed)
                arr[:] = gb
                v2, k2 = ev.score(arr)
                if bestv is None or v2 < bestv - 1e-9:
                    bestv, bestarr = v2, arr.copy()
                    idle = 0
                else:
                    idle += 1
                    if P_NBGIVE and idle >= P_NBGIVE:
                        break
                if P_KREFRESH and k2 != k and len(plan_k) == 1:
                    k = k2
        cur = [int(t) for t in bestarr]
        return cur, ev.score(cur)[0]

    # --- randomised descent, biased towards the tail of each block
    while time.perf_counter() < deadline:
        moved = False
        for (a, b) in spans:
            w = b - a
            if w < 3:
                continue
            for _ in range(min(40, w * 3)):
                if time.perf_counter() > deadline:
                    break
                i = a + int(rng.integers(w))
                if rng.random() < 0.35 and w > 4:
                    j = min(b - 1, i + 2 + int(rng.integers(max(1, w // 2))))
                    if try_rev(min(i, j), max(i, j)):
                        moved = True
                    continue
                if IMP_O and rng.random() < 0.30 and w > 6:
                    L = 2 + int(rng.integers(2))
                    if try_seg(i, L, a + int(rng.integers(w)), a, b):
                        moved = True
                    continue
                j = a + int(min(w - 1, rng.integers(w) if rng.random() < 0.5
                                else w - 1 - rng.integers(max(1, w // 3))))
                if try_move(i, j):
                    moved = True
        if not moved:
            break

    # --- iterated local search: perturb and re-descend while budget remains ---
    if IMP_I:
        best_ids, best_v = list(cur), best_c
        while time.perf_counter() < deadline:
            cur = list(best_ids)
            _, states = ev.run(cur, k, keep=True)
            best_c = best_v
            # double-bridge style perturbation inside one random block
            big = [sp for sp in spans if sp[1] - sp[0] >= 6]
            if not big:
                break
            a, b = big[int(rng.integers(len(big)))]
            w = b - a
            cuts = sorted(int(rng.integers(1, w)) for _ in range(3))
            p1, p2, p3 = (a + c for c in cuts)
            seg = cur[a:b]
            o1, o2, o3 = p1 - a, p2 - a, p3 - a
            cur[a:b] = seg[:o1] + seg[o2:o3] + seg[o1:o2] + seg[o3:]
            _, states = ev.run(cur, k, keep=True)
            best_c = ev.run(cur, k)[0] + BETA_CORE * k
            moved = True
            rounds = 0
            while moved and rounds < 4 and time.perf_counter() < deadline:
                moved = False
                rounds += 1
                for (aa, bb) in spans:
                    ww = bb - aa
                    if ww < 3:
                        continue
                    for _ in range(min(30, ww * 2)):
                        if time.perf_counter() > deadline:
                            break
                        i = aa + int(rng.integers(ww))
                        if rng.random() < 0.35 and ww > 4:
                            j = min(bb - 1, i + 2 + int(rng.integers(max(1, ww // 2))))
                            if try_rev(min(i, j), max(i, j)):
                                moved = True
                            continue
                        j = aa + int(rng.integers(ww))
                        if try_move(i, j):
                            moved = True
            if best_c < best_v - 1e-9:
                best_ids, best_v = list(cur), best_c
        cur = best_ids
    return cur, ev.score(cur)[0]


def _reassign(P, seq, deadline, rng):
    """Move components between FOVs where geometry allows.

    Two payoffs: a lighter final FOV shortens the pipeline tail (the inspection
    that can only start after the very last capture), and a FOV that is emptied
    out disappears entirely (capture + reconstruction saved)."""
    n = len(seq)
    if n < 3:
        return seq
    fid = P.typ == 2
    cx = np.array([f.x for f in seq])
    cy = np.array([f.y for f in seq])
    fstep = np.array([f.step for f in seq])
    locked = np.array([f.fixed or _fid_in(P, f) for f in seq])

    ev = SeqEval(P, seq)
    ids = list(range(n))
    best, k = ev.score(ids)
    comps = [list(f.comps) for f in seq]

    def hosts_of(c):
        m = ((fstep == P.step[c]) & ~locked &
             (cx - P.hw[c] <= P.tlx[c] + EPS) & (cy - P.hh[c] <= P.tly[c] + EPS) &
             (cx + P.hw[c] >= P.brx[c] - EPS) & (cy + P.hh[c] >= P.bry[c] - EPS))
        return np.where(m)[0]

    # candidates: components sitting in the tail of the sequence, heaviest first
    tail_from = max(1, int(n * 0.55))
    cand = []
    for i in range(tail_from, n):
        if locked[i]:
            continue
        for c in comps[i]:
            if not P.big[c]:
                cand.append((float(P.time[c]), i, int(c)))
    cand.sort(reverse=True)

    tried = 0
    for _, i, c in cand:
        if tried >= 140 or time.perf_counter() > deadline:
            break
        if c not in comps[i]:
            continue
        hs = [int(j) for j in hosts_of(c) if j != i and j < i]
        if not hs:
            continue
        # prefer the lightest earlier host
        hs.sort(key=lambda j: ev.load[j])
        for j in hs[:3]:
            tried += 1
            comps[i].remove(c)
            comps[j].append(c)
            ev.set_fov(i, comps[i])
            ev.set_fov(j, comps[j])
            v = ev.ct(ids, k) + BETA_CORE * k
            if v < best - 1e-9:
                best = v
                break
            comps[j].remove(c)
            comps[i].append(c)
            ev.set_fov(i, comps[i])
            ev.set_fov(j, comps[j])

    out = []
    for i, f in enumerate(seq):
        if not comps[i]:
            continue          # FOV emptied out -> drop it (capture + recon saved)
        f.comps = sorted(comps[i], key=lambda c: -P.time[c])
        _touch(f)
        out.append(f)
    return out


def evaluate(P, seq):
    """Return (score = min_k CT+10k, ct, k) -- same rule as Checker.compute_ct."""
    ev = SeqEval(P, seq)
    ids = np.arange(len(seq), dtype=np.int64)
    mc = P.max_core
    hi = ev.ct(ids, mc)
    best = (hi + BETA_CORE * mc, hi, mc)
    for k in range(4, mc):
        if hi + BETA_CORE * k >= best[0]:
            break
        ct = ev.ct(ids, k)
        v = ct + BETA_CORE * k
        if v < best[0]:
            best = (v, ct, k)
    return best


# ================================================================= construction
def _groups_global(P, idx):
    return [idx] if len(idx) else []


def _groups_hm(P, idx, L, sx, sy):
    """Hochbaum-Maass shifted block partition (block = L*fov_w x L*fov_h)."""
    if len(idx) == 0:
        return []
    bw, bh = L * P.fov_w, L * P.fov_h
    cx = (P.rx1[idx] + P.rx2[idx]) * 0.5
    cy = (P.ry1[idx] + P.ry2[idx]) * 0.5
    gx = np.floor((cx - sx) / bw).astype(np.int64)
    gy = np.floor((cy - sy) / bh).astype(np.int64)
    key = gx * 100003 + gy
    out = []
    for k in np.unique(key):
        out.append(idx[key == k])
    return out


def _build(P, partition, deadline, lat=None):
    """One full construction: big tiles -> fiducials -> side pass -> top pass."""
    fovs = _big_tiles(P)
    done = np.zeros(P.n, dtype=bool)
    done[P.big] = True

    for i in np.where((P.typ == 2) & ~P.big)[0]:
        cx = (P.rx1[i] + P.rx2[i]) * 0.5
        cy = (P.ry1[i] + P.ry2[i]) * 0.5
        fovs.append(Fov([int(i)], cx, cx, cy, cy, int(P.step[i]), fixed=True))
        done[i] = True

    ordinary = ~P.big & (P.typ != 2)
    if P_TILEFILL and fovs:
        # A big-component tile is a full FOV that is captured anyway, so every
        # ordinary component inside its window rides along for free.  Handing
        # them to the tiles before the cover runs takes them out of the covering
        # instance entirely -- worth 4-14% of the components on big layouts.
        tiles = [f for f in fovs if f.fixed and any(P.big[c] for c in f.comps)]
        if tiles:
            tx = np.array([f.x for f in tiles])
            ty = np.array([f.y for f in tiles])
            ts = np.array([f.step for f in tiles])
            tside = np.array([_has_side(P, f) for f in tiles])
            for i in np.where(ordinary & ~done)[0]:
                m = ((ts == P.step[i]) &
                     (tx - P.hw[i] <= P.tlx[i] + EPS) & (ty - P.hh[i] <= P.tly[i] + EPS) &
                     (tx + P.hw[i] >= P.brx[i] - EPS) & (ty + P.hh[i] >= P.bry[i] - EPS))
                if P.side[i] == 1:
                    m &= tside          # never turn a 0.50 s capture into 0.75 s
                w = np.where(m)[0]
                if len(w):
                    j = int(w[np.argmin([len(tiles[q].comps) for q in w])])
                    tiles[j].comps.append(int(i))
                    _touch(tiles[j])
                    done[i] = True
    if IMP_D:
        for st in (0, 1):
            base = np.where(ordinary & (P.step == st) & ~done)[0]
            if len(base) == 0:
                continue
            fs, cov = _cover_exact(P, base, np.zeros(0, dtype=np.int64), 0.0,
                                   cost_aware=True)
            fovs.extend(fs)
            done[cov] = True
        return fovs
    for st in (0, 1):
        base = np.where(ordinary & (P.step == st) & ~done)[0]
        if len(base) == 0:
            continue
        mark = len(fovs)
        side_all = base[P.side[base] == 1]
        top_all = base[P.side[base] == 0]

        # --- pass 1: side components (half FOV), top components ride along
        for grp in partition(P, side_all):
            gs = grp[~done[grp]]
            if len(gs) == 0:
                continue
            lo_x, hi_x = P.rx1[gs].min() - P.fov_w, P.rx2[gs].max() + P.fov_w
            lo_y, hi_y = P.ry1[gs].min() - P.fov_h, P.ry2[gs].max() + P.fov_h
            near = top_all[~done[top_all]]
            near = near[(P.rx2[near] >= lo_x) & (P.rx1[near] <= hi_x) &
                        (P.ry2[near] >= lo_y) & (P.ry1[near] <= hi_y)]
            if lat is not None:
                fs, cov, gs2 = _cover_lattice(P, gs, near, P_BONUS, lat[0], lat[1], lat[2])
                fovs.extend(fs); done[cov] = True
                gs = gs2[~done[gs2]] if len(gs2) else gs2
                near = near[~done[near]]
            if len(gs):
                got = _cover_rwls(P, gs, near, P_BONUS, exact=True)
                fs, cov = got if got is not None else \
                    _cover(P, gs, near, P_BONUS, deadline)
                fovs.extend(fs)
                done[cov] = True

        # --- pass 2: remaining top components (full FOV)
        for grp in partition(P, top_all):
            gs = grp[~done[grp]]
            if len(gs) == 0:
                continue
            empty = np.zeros(0, dtype=np.int64)
            if lat is not None:
                fs, cov, gs2 = _cover_lattice(P, gs, empty, 0.0, lat[0], lat[1], lat[2])
                fovs.extend(fs); done[cov] = True
                gs = gs2[~done[gs2]] if len(gs2) else gs2
            if len(gs):
                got = _cover_rwls(P, gs, empty, 0.0, exact=True)
                fs, cov = got if got is not None else \
                    _cover(P, gs, empty, 0.0, deadline)
                fovs.extend(fs)
                done[cov] = True

        # the two passes are a decomposition, not an optimum: re-solve the step
        # jointly from the result they just produced
        if P_JOINT and len(fovs) > mark:
            jg = _cover_joint(P, base, fovs[mark:])
            if jg is not None:
                # Fewer FOVs is not automatically cheaper: a joint cover trades
                # 0.50 s top-only FOVs for 0.75 s side-carrying ones, so the
                # capture+reconstruction cost decides, not the count.
                if _cap_of(P, jg[0]) < _cap_of(P, fovs[mark:]) - 1e-9:
                    fovs[mark:] = jg[0]
                    done[jg[1]] = True

    return fovs


def _finalise_multi(P, fovs, tries, rng, deadline):
    """`_merge` and `_redistribute` both scan FOVs in a fixed order and commit
    greedily, so the result depends on that order.  Re-running the cleanup from
    shuffled orders is extremely cheap (~5 ms) and reliably removes another FOV
    or two."""
    best, bestc = None, None
    for t in range(max(1, tries)):
        if t and time.perf_counter() > deadline:
            break
        work = _clone(fovs)
        if t:
            idxs = rng.permutation(len(work))
            work = [work[i] for i in idxs]
        got = _finalise(P, work)
        c = _cap_of(P, got)
        if bestc is None or c < bestc - 1e-12:
            best, bestc = got, c
    return best


P_FIN_CLEAN = _env('P_FIN_CLEAN', 1, int)


def _finalise(P, fovs):
    if P_FIN_CLEAN:
        fovs = _merge(P, fovs)
        fovs = _redistribute(P, fovs)
        fovs = _merge(P, fovs, sweeps=6)
        fovs = _drop_redundant(P, fovs)
    for f in fovs:
        f.recentre()
        n0 = len(f.comps)
        f.comps = sorted(set(f.comps), key=lambda c: -P.time[c])   # LPT
        if len(f.comps) != n0:
            _touch(f)
    return [f for f in fovs if f.comps]


# ======================================================================= driver
def _repair(P, fovs):
    """Guarantee feasibility: every component sits inside the FOV it is listed in,
    no FOV mixes steps, and nothing is left uncovered."""
    kept, orphan = [], []
    for f in fovs:
        f.recentre()
        good = []
        for c in f.comps:
            if P.big[c]:
                good.append(c)
                continue
            if (f.x - P.hw[c] <= P.tlx[c] + 1e-7 and f.y - P.hh[c] <= P.tly[c] + 1e-7 and
                    f.x + P.hw[c] >= P.brx[c] - 1e-7 and f.y + P.hh[c] >= P.bry[c] - 1e-7 and
                    int(P.step[c]) == f.step):
                good.append(c)
            else:
                orphan.append(c)
        f.comps = good
        _touch(f)
        if good:
            kept.append(f)
    seen = np.zeros(P.n, dtype=bool)
    for f in kept:
        seen[f.comps] = True
    missing = [int(c) for c in np.where(~seen)[0] if not any(c in f.comps for f in kept)]
    orphan.extend(m for m in missing if m not in orphan)
    if orphan:
        orphan = _absorb(P, kept, orphan)
        for c in orphan:
            cx = (P.rx1[c] + P.rx2[c]) * 0.5
            cy = (P.ry1[c] + P.ry2[c]) * 0.5
            g = Fov([int(c)], P.rx1[c], P.rx2[c], P.ry1[c], P.ry2[c], int(P.step[c]))
            g.x, g.y = cx, cy
            kept.append(g)
    for f in kept:
        f.comps = sorted(set(f.comps), key=lambda c: -P.time[c])
    return kept


def _hold_fid(fn):
    """Run a plan-level pass with the fiducial FOVs held out of the list.

    Contest rule 2.3 reserves the fiducial FOV for the fiducial alone, but the
    redistribute / drop / relabel kernels treat *any* FOV whose window covers a
    component as a valid host, so each of them quietly parks ordinary parts
    there -- and utils/checker.py never complains.  Guarding each kernel
    separately means touching their numba signatures; holding the FOVs out of
    the list is the one guard that works for all of them.  The Fov objects are
    shared, so the pass still operates on the real plan, and the held-out FOVs
    go back at the index they came from so the visit order is preserved."""
    @functools.wraps(fn)
    def g(P, fovs, *a, **k):
        at = [i for i, f in enumerate(fovs) if _fid_in(P, f)]
        if not at:
            return fn(P, fovs, *a, **k)
        keep = set(at)
        held = [fovs[i] for i in at]
        rest = [f for i, f in enumerate(fovs) if i not in keep]
        r = fn(P, rest, *a, **k)
        tup = isinstance(r, tuple)
        out = list(r[0]) if tup else list(r)
        for i, f in zip(at, held):
            out.insert(min(i, len(out)), f)
        return ((out,) + tuple(r[1:])) if tup else out
    return g


_redistribute = _hold_fid(_redistribute)
_drop_redundant = _hold_fid(_drop_redundant)
_reassign_seq = _hold_fid(_reassign_seq)


def solve(data_folder: str) -> int:
    t_start = time.perf_counter()
    P = Problem(data_folder)
    deadline = t_start + P_TIME_BUDGET
    rng = np.random.default_rng(20260918)

    # ---- 1. construction (global weighted geometric set cover) --------------
    raw = _build(P, _groups_global, deadline)
    if IMP_J:
        plans = [_finalise_multi(P, raw, P_J_TRIES, rng,
                                 t_start + 0.45 * P_TIME_BUDGET)]
    else:
        plans = [_finalise(P, raw)]
    if P_PORTFOLIO:
        # Hochbaum-Maass variants: shifted block decomposition and offset-swept
        # lattice piercing.  Measured on the practice set they lose to the
        # global greedy, so they are opt-in (HM_P_PORTFOLIO=1).
        cut = t_start + P_CONSTRUCT_FRAC * P_TIME_BUDGET
        for sh in range(P_HM_L):
            if time.perf_counter() > cut:
                break
            part = (lambda pp, ii, _x=sh * P.fov_w, _y=sh * P.fov_h:
                    _groups_hm(pp, ii, P_HM_L, _x, _y))
            plans.append(_finalise(P, _build(P, part, deadline)))
        for ofx, ofy, pct in ((0.0, 0.0, 2.0), (0.5, 0.5, 2.0), (0.0, 0.5, 12.0)):
            if time.perf_counter() > cut:
                break
            plans.append(_finalise(P, _build(P, _groups_global, deadline,
                                             lat=(ofx, ofy, pct))))
    plan = min((pl for pl in plans if pl), key=lambda pl: _cost_proxy(P, pl))

    # ---- 2. LNS: destroy a spatial window, re-cover, keep improvements ------
    lns_frac = P_LNS_END
    if IMP_A:
        # Reducing FOV count only pays off while core 0 (capture) is the
        # bottleneck.  When inspection dominates, CT = sum(t)/(k-3) no matter how
        # few FOVs there are -- so hand the whole budget to the sequencer.
        c0, insp = _bottleneck(P, plan)
        if insp > c0 * 1.05:
            lns_frac = 0.04
        elif c0 > insp * 1.05:
            lns_frac = 0.65
    plan = _lns(P, plan, t_start + lns_frac * P_TIME_BUDGET, rng)
    if IMP_J:
        plan = _finalise_multi(P, plan, P_J_TRIES, rng,
                               t_start + (lns_frac + 0.15) * P_TIME_BUDGET)
    if IMP_F:
        plan = _kexchange(P, plan, t_start + (lns_frac + 0.20) * P_TIME_BUDGET, rng)
    plan = _repair(P, _finalise(P, plan))

    # ---- 3. sequencing: block structure + several spatial heuristics --------
    # Stage 1: the block order (which step goes first, per FOV type) decides
    # both the 5 s Z-transitions and how early the heavy inspection work lands.
    # Search it with one cheap ordering heuristic ...
    opts = (None, (0, 1), (1, 0))
    scored = []
    if P_SO_MODE == 'global':
        for so in opts:
            for mode in P_MODES:
                seq = _sequence(P, plan, mode, so)
                if not P_TSEED_TOP:
                    seq = _travel_seed(P, seq)
                scored.append((evaluate(P, seq)[0], (mode, so), seq))
    else:
        probe = P_MODES[0]
        best_so, best_v = None, None
        for t1 in opts:
            for t0 in opts:
                so = {1: t1, 0: t0}
                v = evaluate(P, _sequence(P, plan, probe, so))[0]
                if best_v is None or v < best_v:
                    best_v, best_so = v, so
        # ... then spend the mode budget on the winning block order.
        for mode in P_MODES:
            seq = _sequence(P, plan, mode, best_so)
            if not P_TSEED_TOP:
                seq = _travel_seed(P, seq)
            scored.append((evaluate(P, seq)[0], (mode, 'so'), seq))
    scored.sort(key=lambda z: z[0])
    if P_TSEED_TOP and P_TSEED:
        # The travel surrogate costs ~3 ms per candidate, so converge it only on
        # the few orderings that could still win.
        head = []
        for v, tag, sq in scored[:P_TSEED_TOP]:
            s2 = _travel_seed(P, sq)
            head.append((evaluate(P, s2)[0], tag, s2))
        scored = sorted(head + scored[P_TSEED_TOP:], key=lambda z: z[0])

    # ---- 4. pipeline polish: hill-climb inside each type/step block ---------
    # Picking the winning mode *before* polishing is myopic, so IMP_K polishes
    # the best two candidates and keeps whichever ends up lower.
    def _blocks_of(seq):
        out, cur, run = [], None, 0
        for f in seq:
            key = _ftype(P, f) if _pblock() else (_ftype(P, f), f.step)
            if key != cur:
                if run:
                    out.append(run)
                cur, run = key, 1
            else:
                run += 1
        if run:
            out.append(run)
        return out

    cands = scored[:2] if IMP_K else scored[:1]
    share = (deadline - time.perf_counter()) / max(1, len(cands))
    bestseq, bestval = None, None
    for _, _, seq0 in cands:
        ev = SeqEval(P, seq0)
        ids0 = list(range(len(seq0)))
        ids0, val = _polish(ev, ids0, _blocks_of(seq0),
                            min(deadline, time.perf_counter() + share), rng)
        cand = [seq0[i] for i in ids0]
        ty = [_ftype(P, f) for f in cand]
        if any(ty[q] < ty[q + 1] for q in range(len(ty) - 1)):
            cand, val = seq0, evaluate(P, seq0)[0]      # never ship a bad order
        if bestval is None or val < bestval:
            bestval, bestseq = val, cand
    seq = bestseq

    if IMP_B:
        seq = _reassign(P, seq, deadline + 0.25 * P_TIME_BUDGET, rng)

    out = os.path.join(data_folder, 'output_fov.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['', 'x', 'y', 'comp_idx'])
        for i, fv in enumerate(seq):
            w.writerow([i, f'{fv.x:.6f}', f'{fv.y:.6f}',
                        json.dumps([int(c) for c in fv.comps])])
    return len(seq)
