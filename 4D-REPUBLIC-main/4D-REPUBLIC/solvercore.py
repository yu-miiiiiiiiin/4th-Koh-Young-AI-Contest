"""Shared back-end for every solver variant.

Each solver only has to produce a FOV set; sequencing, pipeline polish,
feasibility repair and output writing are identical so that the *construction*
algorithms can be compared on equal terms.
"""
import os
import csv
import json
import time

import numpy as np

from . import fastops
from . import core

Problem = core.Problem
Fov = core.Fov
EPS = core.EPS


# ----------------------------------------------------------------- geometry --
def base_plan(P):
    """Big-component tiles + one FOV per fiducial, plus the 'done' mask."""
    fovs = core._big_tiles(P)
    done = np.zeros(P.n, dtype=bool)
    done[P.big] = True
    for i in np.where((P.typ == 2) & ~P.big)[0]:
        cx = (P.rx1[i] + P.rx2[i]) * 0.5
        cy = (P.ry1[i] + P.ry2[i]) * 0.5
        f = Fov([int(i)], cx, cx, cy, cy, int(P.step[i]), fixed=True)
        fovs.append(f)
        done[i] = True
    return fovs, done


def candidates(P, idx):
    """Extreme-point candidate centres + the coverage matrix (cand x comp).

    The bottom-right corner of each feasible-centre rectangle R_i is a sound
    candidate set: an optimal piercing always has a point at such a corner."""
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    cov = ((x2[:, None] >= x1[None, :] - EPS) & (x2[:, None] <= x2[None, :] + EPS) &
           (y2[:, None] >= y1[None, :] - EPS) & (y2[:, None] <= y2[None, :] + EPS))
    return np.stack([x2, y2], 1), cov


def rich_candidates(P, idx, extra_rounds=3):
    """Extreme-point candidates *plus* the positions the integral-image greedy
    would choose.  Giving every model the same pool isolates the selection
    algorithm being compared from the candidate generator."""
    pts, cov = candidates(P, idx)
    extra = []
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    ox, oy = float(x1.min()), float(y1.min())
    spx = max(float(x2.max()) - ox, 1e-6)
    spy = max(float(y2.max()) - oy, 1e-6)
    g = max(spx / 300, spy / 300, 0.30)
    nx, ny = int(spx / g) + 1, int(spy / g) + 1
    rem = np.ones(len(idx), dtype=bool)
    for _ in range(4000):
        if not rem.any():
            break
        sel = np.where(rem)[0]
        pt = core._max_coverage_point(x1[sel], x2[sel], y1[sel], y2[sel],
                                      np.ones(len(sel)), ox, oy, g, nx, ny)
        if pt is None:
            break
        px, py = pt
        hit = sel[(x1[sel] <= px + EPS) & (x2[sel] >= px - EPS) &
                  (y1[sel] <= py + EPS) & (y2[sel] >= py - EPS)]
        if len(hit) == 0:
            break
        extra.append((px, py))
        rem[hit] = False
    if not extra:
        return pts, cov
    ep = np.array(extra)
    allp = np.vstack([pts, ep])
    px = allp[:, 0][:, None]
    py = allp[:, 1][:, None]
    cov2 = ((px >= x1[None, :] - EPS) & (px <= x2[None, :] + EPS) &
            (py >= y1[None, :] - EPS) & (py <= y2[None, :] + EPS))
    return allp, cov2


def cross_candidates(P, idx, cap=9000, nb=14):
    """Complete extreme-point pool.

    Pushing an optimal piercing point right and down until it is blocked makes
    it (R_i.x2, R_j.y2) for *possibly different* i and j -- using only i == j
    (the cheap pool) can miss the optimum entirely.  Cross pairs are generated
    for spatial neighbours only, deduplicated, and reduced by column dominance
    (a candidate whose coverage is contained in another's is never needed)."""
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    m = len(idx)
    ox = np.argsort(x2)
    pts = [(float(x2[i]), float(y2[i])) for i in range(m)]
    # neighbours in x-order: their R can overlap, so the crossed corner is feasible
    for a in range(m):
        i = int(ox[a])
        lo = max(0, a - nb)
        hi = min(m, a + nb + 1)
        for b in range(lo, hi):
            j = int(ox[b])
            if j == i:
                continue
            px, py = float(x2[i]), float(y2[j])
            if x1[i] - EPS <= px <= x2[i] + EPS and y1[j] - EPS <= py <= y2[j] + EPS:
                pts.append((round(px, 4), round(py, 4)))
    pts = np.array(sorted(set(pts)))
    if len(pts) == 0:
        return candidates(P, idx)
    px = pts[:, 0][:, None]
    py = pts[:, 1][:, None]
    cov = ((px >= x1[None, :] - EPS) & (px <= x2[None, :] + EPS) &
           (py >= y1[None, :] - EPS) & (py <= y2[None, :] + EPS))
    keep = cov.any(1)
    pts, cov = pts[keep], cov[keep]
    if len(pts) > cap:
        dset = {(round(float(x2[i]), 4), round(float(y2[i]), 4)) for i in range(m)}
        diag = np.array([(round(float(a), 4), round(float(b), 4)) in dset
                         for a, b in pts])
        rest = np.where(~diag)[0]
        room = max(0, cap - int(diag.sum()))
        sel = np.concatenate([np.where(diag)[0],
                              rest[np.argsort(-cov[rest].sum(1))[:room]]])
        pts, cov = pts[sel], cov[sel]
    assert cov.any(0).all(), 'candidate pool misses a component'
    return pts, cov


def blocks_of(P, idx):
    """Split components into independent sub-problems: two components can only
    share a FOV when their R rectangles intersect, so connected components of
    that intersection graph are solved separately."""
    n = len(idx)
    if n == 0:
        return []
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    seen = np.zeros(n, dtype=bool)
    out = []
    for s in range(n):
        if seen[s]:
            continue
        comp = [s]
        seen[s] = True
        head = 0
        while head < len(comp):
            k = comp[head]
            head += 1
            m = (~seen & (x1 <= x2[k] + EPS) & (x2 >= x1[k] - EPS) &
                 (y1 <= y2[k] + EPS) & (y2 >= y1[k] - EPS))
            w = np.where(m)[0]
            if len(w):
                seen[w] = True
                comp.extend(int(v) for v in w)
        out.append(np.array(comp, dtype=np.int64))
    return out


def two_pass(P, select, bonus_w=0.08):
    """Side-first decomposition shared by every model.

    A side component must sit in the central quarter of a FOV, so side
    components are the binding constraint; covering them first and letting the
    resulting FOVs absorb nearby top components for free is worth far more than
    any refinement of the selection rule itself.

    `select(must, bonus, weights) -> (points, cov, chosen)` is the model's own
    selection algorithm."""
    fovs, done = base_plan(P)
    for st in (0, 1):
        base = np.where(~P.big & (P.typ != 2) & (P.step == st) & ~done)[0]
        if len(base) == 0:
            continue
        side = base[P.side[base] == 1]
        top = base[P.side[base] == 0]
        if len(side):
            w = np.concatenate([np.ones(len(side)), np.full(len(top), bonus_w)])
            pts, cov, chosen = select(side, top, w)
            # a model may return coverage over `must` only or over must+bonus
            idx = (np.concatenate([side, top]) if (len(top) and cov.shape[1] > len(side))
                   else side)
            fovs.extend(assign_sidewise(P, idx, cov, chosen, pts))
            for c in chosen:
                done[idx[cov[c]]] = True
            top = top[~done[top]]
        if len(top):
            w = np.ones(len(top))
            pts, cov, chosen = select(top, np.zeros(0, dtype=np.int64), w)
            fovs.extend(assign_sidewise(P, top, cov, chosen, pts))
            for c in chosen:
                done[top[cov[c]]] = True
    return fovs


def make_fov(P, idx, take, cx, cy):
    """Build a FOV from a candidate centre and the components it takes."""
    g = idx[take]
    bx1 = float(P.rx1[g].max()); bx2 = float(P.rx2[g].min())
    by1 = float(P.ry1[g].max()); by2 = float(P.ry2[g].min())
    if bx1 > bx2 + EPS or by1 > by2 + EPS:
        bx1 = bx2 = float(cx); by1 = by2 = float(cy)
    f = Fov([int(v) for v in g], bx1, bx2, by1, by2, int(P.step[g[0]]))
    f.x, f.y = float(cx), float(cy)
    return f


def assign_sidewise(P, idx, cov, chosen, pts):
    """Turn a chosen candidate set into FOVs, assigning each component so that
    as few FOVs as possible end up holding a side component (0.75 s vs 0.50 s)."""
    chosen = list(chosen)
    taken = {c: [] for c in chosen}
    order = np.argsort(-(P.side[idx].astype(int)))        # side components first
    has_side = {c: False for c in chosen}
    def _margin(c, j):
        """How deep inside the component's feasible box this centre sits.

        The FOV's own slack is the intersection of R over everything it lists,
        so giving a component to the candidate that is furthest from the edge of
        its R keeps the most room for the centre pass to slide the FOV later."""
        g = idx[j]
        px, py = pts[c]
        mx = min(px - P.rx1[g], P.rx2[g] - px)
        my = min(py - P.ry1[g], P.ry2[g] - py)
        if ASSIGN_REL:
            mx /= max(P.rx2[g] - P.rx1[g], 1e-9)
            my /= max(P.ry2[g] - P.ry1[g], 1e-9)
        return min(mx, my)

    for j in order:
        opts = [c for c in chosen if cov[c, j]]
        if not opts:
            continue
        if ASSIGN_MARGIN:
            c = min(opts, key=lambda c: (not has_side[c],
                                         ASSIGN_BAL * len(taken[c]) - _margin(c, j)))
        else:
            c = min(opts, key=lambda c: (not has_side[c], len(taken[c])))
        if P.side[idx[j]] == 1:
            has_side[c] = True
        taken[c].append(int(j))
    out = []
    for c in chosen:
        if taken[c]:
            out.append(make_fov(P, idx, np.array(taken[c]), pts[c][0], pts[c][1]))
    return out


# ------------------------------------------------------------------ back end --
CLEAN_ROUNDS = int(os.environ.get('SC_CLEAN_ROUNDS', '3'))




def _fid_exclusive(P, seq, deadline=None):
    """Enforce contest rule 2.3 (Fiducial FOV Restriction).

    "Fiducial components (type=2) must be placed in the first FOV exclusively.
     No other components allowed in the Fiducial FOV.  The Fiducial must be
     centered in the FOV."

    utils/checker.py tests neither half -- cover_all_components only checks
    containment, mixed steps and the type-descending order -- so the merge /
    redistribute / reassign passes quietly park ordinary components in the
    fiducial FOV and slide its centre away.  Freed components are re-covered
    properly (not as singletons), which costs far less than one FOV each.
    """
    freed = []
    for f in seq:
        if not core._fid_in(P, f):
            continue
        fid = [int(c) for c in f.comps if P.typ[c] == 2]
        if not fid:
            continue
        if len(fid) != len(f.comps):
            freed.extend(int(c) for c in f.comps if P.typ[c] != 2)
            f.comps = fid
        c0 = fid[0]
        cx = (P.tlx[c0] + P.brx[c0]) * 0.5
        cy = (P.tly[c0] + P.bry[c0]) * 0.5
        f.x1 = f.x2 = f.x = cx
        f.y1 = f.y2 = f.y = cy
        f.fixed = True
        core._touch(f)
    if not freed:
        return seq
    fids = [f for f in seq if core._fid_in(P, f)]
    hosts = [f for f in seq if not core._fid_in(P, f)]
    if deadline is None:
        deadline = time.perf_counter() + 0.05
    core._repair_free(P, hosts, freed, deadline)
    # _repair_free can drop a component when its sub-cover falls back past the
    # deadline; _repair is the guaranteed-feasibility backstop (and its _absorb
    # no longer treats the fiducial FOV as a host).
    # _repair must see the fiducial FOVs too, or it treats the fiducials as
    # uncovered and re-homes them into a host FOV.  Its _absorb no longer uses
    # a fiducial FOV as a host, so nothing can drift back in.
    return core._repair(P, fids + hosts)


def finish(P, plan, deadline, rng, data_folder, polish=True):
    """Clean up, sequence, polish and write output_fov.csv.  Returns FOV count.

    merge / redistribute / drop are order-dependent greedy passes, so running the
    chain more than once keeps finding FOVs to delete -- measured as the single
    best value-per-second step in the whole pipeline."""
    for _ in range(max(1, CLEAN_ROUNDS) - 1):
        plan = core._merge(P, plan)
        plan = core._redistribute(P, plan)
        plan = core._merge(P, plan, sweeps=6)
        plan = core._drop_redundant(P, plan)
    plan = core._repair(P, core._finalise(P, plan))
    plan = _fid_exclusive(P, plan, deadline)
    for _rw in range(RW_ON):
        got = rwls_improve(P, plan)
        if got is plan:
            break
        got = core._merge(P, got, sweeps=6)
        got = core._redistribute(P, got)
        got = core._drop_redundant(P, got)
        got = core._repair(P, core._finalise(P, got))
        if core._cap_of(P, got) >= core._cap_of(P, plan):
            break
        plan = got

    # same block-order + mode search the tuned solver uses: which step goes
    # first decides both the 5 s Z-transitions and how early the heavy
    # inspection work reaches the inspection cores.  One SeqEval is built for
    # the FOV set and every candidate order is scored as a permutation of it.
    if core.P_MARGIN_ASSIGN and core._margin_assign(P, plan):
        plan = core._repair(P, [f for f in plan if f.comps])
    ev = core.SeqEval(P, plan)
    ftyp = np.array([core._ftype(P, f) for f in plan], dtype=np.int64)
    pos = {id(f): i for i, f in enumerate(plan)}
    # Ranking the candidate orders needs only a consistent yardstick, so the
    # k-sweep runs once and the rest are compared at that core count; the few
    # best are then re-scored properly.
    scache = {}
    cands = []
    kref = None
    for so in (SO_OPTS if SO_OPTS else (None,)):
        for mode in core.P_MODES:
            seq = core._sequence(P, plan, mode, so, scache)
            ids = np.fromiter((pos[id(f)] for f in seq), np.int64, len(seq))
            ids = core._travel_ids(P, ev, ftyp, ids)
            if kref is None or not MODE_FASTK:
                v, kref = ev.score(ids)
            else:
                v = ev.ct(ids, kref) + core.BETA_CORE * kref
            cands.append((v, ids))
    cands.sort(key=lambda z: z[0])
    scored2 = []
    for v, ids in cands[:(MODE_TOP if MODE_FASTK else 1)]:
        scored2.append((ev.score(ids)[0], ids))
    scored2.sort(key=lambda z: z[0])
    ids = scored2[0][1]
    if core.P_TKICKS_TOP:
        # a travel-ILS start for the winner only: the polish trades travel for
        # pipeline from whatever basin it is handed, so the basin matters
        cand = core._travel_ids(P, ev, ftyp, ids, kicks=core.P_TKICKS_TOP)
        if ev.score(cand)[0] < scored2[0][0] - 1e-9:
            ids = cand
    seq = [plan[int(i)] for i in ids]
    alt0 = None
    if ALT_START and len(scored2) > 1:
        # express the runner-up order in the new seq positions
        inv = np.empty(len(plan), dtype=np.int64)
        inv[ids] = np.arange(len(ids), dtype=np.int64)
        alt0 = inv[scored2[1][1]]

    # ---- improvement loop over the three coupled decisions -----------------
    # visit order (pipeline hill climb) -> FOV centres inside their slack ->
    # which FOV lists which component.  Each one changes the input of the next,
    # so they are cycled until nothing moves or the runtime budget runs out
    # (runtime costs 1 score point per second, so the loop is deadline-driven).
    prev_val = None
    for _round in range(max(1, core.P_FIN_ROUNDS)):
        if time.perf_counter() >= deadline:
            break
        # drop / reassign put ordinary components back into the fiducial FOV
        # every round; clean before the polish so the hill climb is always
        # optimising a rule-compliant plan rather than one that has to be
        # patched up after the fact.
        seq = _fid_exclusive(P, seq, deadline)
        ev2 = core.SeqEval(P, seq)
        ft2 = np.array([core._ftype(P, f) for f in seq], dtype=np.int64)
        ids2 = np.arange(len(seq), dtype=np.int64)
        if polish:
            ids2, val = core._polish(ev2, list(ids2),
                                     core.blocks_from(ft2, ev2.stpi, ids2),
                                     deadline, rng, seeds=(_round == 0),
                                     alt=(alt0 if _round == 0 else None))
            seq = [seq[int(i)] for i in ids2]
        if core.P_SLACKPASS and time.perf_counter() < deadline:
            core._slack_pass(P, seq)
        g = core._centre_opt(P, seq) if core.P_CENTRE else 0.0
        ch = False
        if core.P_EVAC and time.perf_counter() < deadline:
            seq, ce = core._evacuate(P, seq, deadline)
            ch = ch or ce
        if CLEAN_IN_LOOP and g > 1e-9:
            order = {id(f): q for q, f in enumerate(seq)}
            got = core._merge(P, list(seq), sweeps=4)
            got = core._redistribute(P, got)
            got = core._drop_redundant(P, got)
            if len(got) < len(seq):
                seq = sorted([f for f in got if f.comps],
                             key=lambda f: order.get(id(f), 1 << 30))
                ch = True
        elif DROP_IN_LOOP and g > 1e-9:
            # the centres just moved, so windows that were not redundant before
            # may be now.  Deleting a FOV hands its components to others, which
            # can lengthen the schedule, so the exact score decides.
            before = [list(f.comps) for f in seq]
            v_before = core.evaluate(P, seq)[0]
            pruned = core._drop_redundant(P, list(seq))
            if len(pruned) < len(seq):
                order = {id(f): q for q, f in enumerate(seq)}
                cand = sorted(pruned, key=lambda f: order[id(f)])
                if core.evaluate(P, cand)[0] < v_before - 1e-9:
                    seq = cand
                    ch = True
                else:
                    for f, cs in zip(seq, before):
                        f.comps = cs
                        core._touch(f)
        if core.P_REASSIGN and time.perf_counter() < deadline:
            keep = list(seq)
            seq, ch = core._reassign_seq(P, seq, deadline, rng)
            ty = [core._ftype(P, f) for f in seq]
            if any(ty[q] < ty[q + 1] for q in range(len(ty) - 1)):
                seq, ch = keep, False      # never ship a broken type order
            if ch and core.P_CENTRE:
                core._centre_opt(P, seq)
        if core.P_RUNMOVE and time.perf_counter() < deadline:
            ch = core._run_move(P, seq, deadline) or ch
        if core.P_JOBORD and time.perf_counter() < deadline:
            ch = core._job_order(P, seq, deadline, rng) or ch
        if g < 1e-9 and not ch:
            break
        # a round that cannot move the true score is the signal to stop: every
        # further second is charged at 1 score point and buys nothing
        cur_val = core.evaluate(P, seq)[0]
        if prev_val is not None and prev_val - cur_val < core.P_ROUND_EPS:
            break
        prev_val = cur_val
        if core.P_FLOOR_EPS:
            ev3 = core.SeqEval(P, seq)
            ev3._ftyp = np.array([core._ftype(P, f) for f in seq], dtype=np.int64)
            i3 = np.arange(len(seq), dtype=np.int64)
            _, k3 = ev3.score(i3)
            if ev3.ct(i3, k3) - core._seq_bound(P, ev3, i3, k3) < core.P_FLOOR_EPS:
                break

    seq = _fid_exclusive(P, seq)

    # Final safety net.  The in-loop cleanup passes may hand a component to a
    # FOV that sits later in the sequence and raise that FOV's type, which
    # breaks the type-descending rule and costs the whole dataset a penalty.
    # A stable sort by type repairs it while preserving everything else.
    ty = [core._ftype(P, f) for f in seq]
    if any(ty[q] < ty[q + 1] for q in range(len(ty) - 1)):
        seq = [seq[q] for q in sorted(range(len(seq)), key=lambda q: (-ty[q], q))]

    with open(os.path.join(data_folder, 'output_fov.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['', 'x', 'y', 'comp_idx'])
        for i, fv in enumerate(seq):
            w.writerow([i, f'{fv.x:.6f}', f'{fv.y:.6f}',
                        json.dumps([int(c) for c in fv.comps])])
    return len(seq)


# ----------------------------------------------------- set-cover improvement --
RW_ITERS = int(os.environ.get('SC_RW_ITERS', '6000'))
RW_NB = int(os.environ.get('SC_RW_NB', '2'))
SO_OPTS = ((None, (0, 1), (1, 0)) if int(os.environ.get('SC_SO', '1')) else (None,))
MODE_FASTK = int(os.environ.get('SC_MODE_FASTK', '1'))
MODE_TOP = int(os.environ.get('SC_MODE_TOP', '3'))
ALT_START = int(os.environ.get('SC_ALT_START', '0'))
ASSIGN_MARGIN = int(os.environ.get('SC_ASSIGN_MARGIN', '1'))
ASSIGN_REL = int(os.environ.get('SC_ASSIGN_REL', '0'))
ASSIGN_BAL = float(os.environ.get('SC_ASSIGN_BAL', '0'))
DROP_IN_LOOP = int(os.environ.get('SC_DROP_LOOP', '1'))
CLEAN_IN_LOOP = int(os.environ.get('SC_CLEAN_LOOP', '0'))
RW_ON = int(os.environ.get('SC_RW_ON', '0'))   # 구성 단계가 완전풀 RWLS로 대체 (R99)   # measured: -29 FOVs for +5 s RT, net negative


def _cand_points(P, idx, nb):
    """Complete extreme-point candidate set (see hm_solver._full_pool)."""
    return core._full_pool(P, idx)


def _csr(P, idx, pts, chunk=512):
    x1, x2 = P.rx1[idx], P.rx2[idx]
    y1, y2 = P.ry1[idx], P.ry2[idx]
    rr, cc = [], []
    for s in range(0, len(pts), chunk):
        px = pts[s:s + chunk, 0][:, None]
        py = pts[s:s + chunk, 1][:, None]
        c = ((px >= x1[None, :] - EPS) & (px <= x2[None, :] + EPS) &
             (py >= y1[None, :] - EPS) & (py <= y2[None, :] + EPS))
        a, b = np.nonzero(c)
        rr.append(a + s)
        cc.append(b)
    r = np.concatenate(rr) if rr else np.zeros(0, np.int64)
    c = np.concatenate(cc) if cc else np.zeros(0, np.int64)
    m, n = len(pts), len(idx)
    o = np.argsort(r, kind='stable')
    se_ptr = np.zeros(m + 1, np.int64)
    np.add.at(se_ptr, r[o] + 1, 1)
    se_ptr = np.cumsum(se_ptr)
    o2 = np.argsort(c, kind='stable')
    es_ptr = np.zeros(n + 1, np.int64)
    np.add.at(es_ptr, c[o2] + 1, 1)
    es_ptr = np.cumsum(es_ptr)
    return se_ptr, c[o].astype(np.int64), es_ptr, r[o2].astype(np.int64)


def rwls_improve(P, plan, iters=None, nb=None):
    """Re-solve the FOV cover with row-weighting local search, seeded by `plan`.

    The construction+cleanup+LNS chain lands 2-3% above the minimum cover, and
    on core-0-bound datasets each FOV is worth ~0.93 s of cycle time (0.75 s of
    capture plus a hop).  RWLS starts from the incumbent, so it can only move
    down.  Big-component tiles and fiducial FOVs are held exactly as they are --
    including the components they already carry -- so the sub-problem is a clean
    unicost set cover over the remaining components, one instance per step
    (FOVs may not mix step levels)."""
    if not RW_ON or not core._FAST:
        return plan
    iters = RW_ITERS if iters is None else iters
    nb = RW_NB if nb is None else nb
    fixed, movable = [], []
    for f in plan:
        (fixed if (f.fixed or core._fid_in(P, f)) else movable).append(f)
    if not movable:
        return plan
    held = np.zeros(P.n, dtype=bool)
    for f in fixed:
        held[f.comps] = True
    free = np.where(~held & ~P.big & (P.typ != 2))[0]
    if len(free) == 0:
        return plan
    out = list(fixed)
    changed = False
    for st in np.unique(P.step[free]):
        idx = free[P.step[free] == st]
        seedf = [f for f in movable if f.step == st]
        pts = _cand_points(P, idx, nb)
        if seedf:
            sp = np.array([[f.x, f.y] for f in seedf])
            allp = np.vstack([pts, sp]) if len(pts) else sp
        else:
            allp = pts
        ok = len(allp) > 0 and len(seedf) > 1
        if ok:
            se_ptr, se_idx, es_ptr, es_idx = _csr(P, idx, allp)
            m, n = len(allp), len(idx)
            ins = np.zeros(m, np.bool_)
            ins[len(pts):] = True
            cov = np.zeros(n, np.bool_)
            for sx in np.where(ins)[0]:
                cov[se_idx[se_ptr[sx]:se_ptr[sx + 1]]] = True
            ok = bool(cov.all())
        if ok:
            bs = np.zeros(m, np.bool_)
            fastops.rwls(se_ptr, se_idx, es_ptr, es_idx, ins.copy(), iters,
                         20260920, bs)
            chosen = np.where(bs)[0]
            if 0 < len(chosen) < len(seedf):
                covm = np.zeros((m, n), dtype=bool)
                for sx in chosen:
                    covm[sx, se_idx[se_ptr[sx]:se_ptr[sx + 1]]] = True
                got = assign_sidewise(P, idx, covm, chosen, allp)
                if got and sum(len(g.comps) for g in got) == len(idx):
                    out.extend(got)
                    changed = True
                    continue
        out.extend(seedf)
        if os.environ.get('SC_RW_DEBUG'):
            print(f'      step {st}: keep {len(seedf)} (ok={ok})', flush=True)
    if os.environ.get('SC_RW_DEBUG'):
        print(f'    RWLS {len(plan)} -> {len(out)}  fixed={len(fixed)} free={len(free)}',
              flush=True)
    if not changed:
        return plan
    return out

PICK_PLAN = int(os.environ.get('SC_PICK_PLAN', '0'))   # 측정: 빠른 랭킹이 최종 점수를 오판해 -5점


def pick_plan(P, cands):
    """Choose between candidate FOV sets on the *real* objective.

    The LNS accepts on a capture proxy (capture + reconstruction), which is what
    makes its iteration cheap, but the proxy is not the score: on 17 of the 96
    practice datasets the LNS plan scores worse than the plan it started from,
    by 31.3 points in total, sometimes without even removing a FOV.  One cheap
    sequencing pass per candidate settles it -- about 10 ms each against an
    average dataset budget of 480 ms."""
    best = None
    for plan in cands:
        if not plan:
            continue
        p = core._repair(P, core._finalise(P, plan))
        if not p:
            continue
        ev = core.SeqEval(P, p)
        ftyp = np.array([core._ftype(P, f) for f in p], dtype=np.int64)
        pos = {id(f): i for i, f in enumerate(p)}
        v = None
        for so in (None, (0, 1)):
            seq = core._sequence(P, p, core.P_MODES[0], so)
            ids = np.fromiter((pos[id(f)] for f in seq), np.int64, len(seq))
            ids = core._travel_ids(P, ev, ftyp, ids)
            s2 = ev.score(ids)[0]
            if v is None or s2 < v:
                v = s2
        if best is None or v < best[0] - 1e-9:
            best = (v, p)
    return best[1] if best else cands[0]
