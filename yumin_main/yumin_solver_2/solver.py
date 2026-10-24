"""
Solver for ICCAS 2026 KohYoung AI Contest  --  cluster-and-cover strategy.

Difference from yumin_solver (Riemann tiling)
---------------------------------------------
yumin_solver took each group's bounding area and filled it with a regular grid
of FOVs. On a sparse PCB most of that area is empty, so a lot of the grid was
spent imaging bare laminate -- every wasted tile still costs a capture plus a
stage move, which is what drove CT up.

This solver never reasons about an area. It reasons about where the components
actually are, clusters them, and covers each cluster with as few FOVs as it can.

Algorithm
---------
1. SCAN ORDER.  Components are split into pure (type, step) groups and visited
   in this order:

       type=2 (Fiducial)  -- one dedicated FOV each, centred on the part,
                             in registration order (rules 2.3 / 3.2)
       type=1 (Barcode)   -- step-major; may also carry ordinary parts (2.4)
       type=0 (Normal)    -- step=0 then step=1

   Groups are pure in (type, step), so FOV type is non-increasing (order rule)
   and no FOV ever mixes step levels (step rule) -- both hold by construction
   rather than by checking.

   Top and side are captured together at one position (rule 1.3), so a group is
   *not* split by side; `SPLIT_SIDE` above restores the split. Each component
   still keeps its own window, half FOV for side=1 (rule 2.6).

   The step order within type=1 is not fixed by the scan rule, so it is chosen
   to chain with the step the fiducials ended on and with type=0's fixed
   step=0 -> step=1: the 5s Z-transition is then paid once per type instead of
   at every group boundary. See `_group_sequence`.

2. CLUSTERING.  Inside a group, components are bucketed into cells the size of
   one effective FOV and 8-connected cells are unioned (`_clusters`). Each
   connected blob is one cluster; empty regions simply break the chain, so no
   FOV is ever generated over a gap.

3. COVERING.  A FOV at centre p covers component c iff c fits inside p's
   window, which for a fixed window size is exactly

       p.x in [c.br_x - ew/2, c.tl_x + ew/2]
       p.y in [c.br_y - eh/2, c.tl_y + eh/2]

   i.e. every component becomes an axis-aligned *feasible-centre box*, and
   "cover this cluster with fewest FOVs" becomes "stab all these boxes with
   fewest points" -- the minimum piercing problem. Axis-aligned boxes have
   Helly number 2, so a set of components shares one FOV exactly when their
   boxes intersect *pairwise*, which makes this a minimum clique cover and
   makes every step below a cheap box-intersection test:

     a. `_strip_cover` -- sweep strips along one axis and run the exact 1-D
        interval-piercing greedy inside each strip. Run all 8 axis/direction
        restarts, polish each, keep the best.
     b. `_merge`       -- two FOVs fuse iff their feasible boxes still
        intersect; greedily grow each FOV until nothing else fits.
     c. `_dissolve`    -- delete a FOV by handing each of its components to a
        FOV that can still *shift* onto it (not just one already sitting on
        it), then re-merge into the slack that frees up.
     d. `_absorb`      -- drop any FOV whose every component also fits a
        surviving FOV where it already stands.

   Oversized components are tiled separately (`_tile_big`); since the checker
   re-inspects such a component once per tile, those tiles are a sunk cost and
   are refilled with whatever else already fits inside them.

4. ROUTING.  Clusters are chained nearest-neighbour under the checker's own
   `max(x-axis, y-axis)` travel model; FOVs inside a cluster run in boustrophedon
   (snake) order, entered from whichever corner is closest. Finally each FOV is
   nudged inside its feasible box toward its neighbours to shave travel.

Usage:
    python 2_run.py --solver yumin_solver_2
"""
import os
import json
import math

import numpy as np
import pandas as pd

TOL = 1e-9
# Coverage tests in utils/checker.py run at 1e-5; stay strictly inside that.
SAFE = 1e-7

# One FOV holds both sides, because rule 1.3 captures Top and Side together at
# the same position -- scanning side=1 as its own phase revisits nearly the same
# spots a second time. Rule 4.6 charges such a FOV max(capture, side_capture),
# i.e. 0.75s instead of 0.5s, not a second shot, so the ~26% drop in FOV count
# wins. Set True to scan side=0 and side=1 as separate phases instead.
SPLIT_SIDE = False
LPT = True


# --------------------------------------------------------------------------
# travel model (mirrors Checker._compute_axis_time / _compute_cost_map)
# --------------------------------------------------------------------------

def _axis_time(dist, vel, acc):
    """Time to move `dist` along one axis under a trapezoidal profile."""
    th = vel ** 2 / acc
    dist = np.asarray(dist, dtype=float)
    over = 2 * vel / acc + (dist - th) / vel
    under = 2 * np.sqrt(np.maximum(dist, 0.0) / acc)
    return np.where(dist >= th, over, under)


def _make_travel(param):
    """Return travel(px, py, qx, qy) -> stage time, as the checker computes it."""
    vel_x, vel_y, acc_x, acc_y = param

    def travel(px, py, qx, qy):
        return np.maximum(_axis_time(np.abs(np.asarray(px) - qx), vel_x, acc_x),
                          _axis_time(np.abs(np.asarray(py) - qy), vel_y, acc_y))

    return travel


# --------------------------------------------------------------------------
# 1. scan order
# --------------------------------------------------------------------------

def _group_sequence(present, start_step):
    """(type, step, side) groups in scan order; `present` holds non-empty keys.

    Fiducials are not here -- rule 3.2 pins them to registration order, so they
    are emitted separately and `start_step` carries the step the last one left
    the Z axis on (None when the board has no fiducial).

    Type is descending, so the inspection-order rule holds by construction.
    Inside a type the two step levels run back to back, and for type=1 -- whose
    step order the scan rule leaves free -- the order is picked so the steps
    chain across the type boundaries. That pays the 5s Z-transition once per
    type instead of once per boundary. type=0 keeps the specified
    step=0 -> step=1 order.
    """
    def groups_of(t, order):
        out = []
        for s in order:
            for side in (0, 1):            # side=0 (top) before side=1
                if (t, s, side) in present:
                    out.append((t, s, side))
        return out

    best, best_cost = None, None
    for order1 in ((0, 1), (1, 0)):        # type=1 step order is free
        seq = groups_of(1, order1) + groups_of(0, (0, 1))
        cost = sum(seq[i][1] != seq[i + 1][1] for i in range(len(seq) - 1))
        if seq and start_step is not None and seq[0][1] != start_step:
            cost += 1
        if best_cost is None or cost < best_cost:
            best, best_cost = seq, cost
    return best


# --------------------------------------------------------------------------
# 2. clustering
# --------------------------------------------------------------------------

def _clusters(cx, cy, ew, eh):
    """8-connected blobs of FOV-sized occupied cells; returns lists of positions."""
    gi = np.floor(cx / ew).astype(np.int64)
    gj = np.floor(cy / eh).astype(np.int64)

    cells = {}
    for k in range(len(gi)):
        cells.setdefault((int(gi[k]), int(gj[k])), []).append(k)

    keys = list(cells)
    at = {key: i for i, key in enumerate(keys)}
    parent = list(range(len(keys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (i, j) in keys:
        a = at[(i, j)]
        # only the "forward" half of the 8-neighbourhood; the rest is symmetric
        for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
            b = at.get((i + di, j + dj))
            if b is not None:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    blobs = {}
    for key in keys:
        blobs.setdefault(find(at[key]), []).extend(cells[key])
    return list(blobs.values())


# --------------------------------------------------------------------------
# 3. covering -- minimum piercing of the feasible-centre boxes
# --------------------------------------------------------------------------

def _strip_cover(lo_x, hi_x, lo_y, hi_y, flip_x, flip_y, swap):
    """Piercing set: sweep strips on one axis, exact 1-D greedy on the other.

    `swap` transposes the axes and `flip_*` reverses a sweep direction (negating
    an interval just exchanges its bounds), giving 8 cheap restarts of the same
    sweep. They disagree often enough that keeping the best is worth the time.
    """
    if flip_x:
        lo_x, hi_x = -hi_x, -lo_x
    if flip_y:
        lo_y, hi_y = -hi_y, -lo_y
    if swap:
        lo_x, hi_x, lo_y, hi_y = lo_y, hi_y, lo_x, hi_x

    live = np.ones(len(lo_x), dtype=bool)
    out = []
    while live.any():
        idx = np.flatnonzero(live)
        # the strip line must sit at the smallest upper bound still open,
        # which is what makes the 1-D sweep optimal
        sy = hi_y[idx].min()
        strip = idx[lo_y[idx] <= sy + TOL]

        open_ = np.ones(len(strip), dtype=bool)
        while open_.any():
            rest = strip[open_]
            sx = hi_x[rest].min()
            sel = lo_x[rest] <= sx + TOL
            take = rest[sel]
            out.append(list(take))
            live[take] = False
            open_[np.flatnonzero(open_)[sel]] = False
    return out


def _boxes_of(members, lo_x, hi_x, lo_y, hi_y):
    """Feasible-centre box of each FOV = intersection over its components."""
    n = len(members)
    b = (np.empty(n), np.empty(n), np.empty(n), np.empty(n))
    for i, ms in enumerate(members):
        b[0][i] = lo_x[ms].max()
        b[1][i] = hi_x[ms].min()
        b[2][i] = lo_y[ms].max()
        b[3][i] = hi_y[ms].min()
    return b


def _merge(box, members, alive):
    """Fuse FOVs whose feasible boxes still intersect (greedy clique growth).

    Two FOVs can share a single shot exactly when the intersection of all their
    components' feasible boxes is non-empty -- and since each FOV's box is
    already that intersection, the test is just "do the two boxes overlap".
    """
    b_lo_x, b_hi_x, b_lo_y, b_hi_y = box
    for i in range(len(b_lo_x)):
        if not alive[i]:
            continue
        while True:
            cand = (alive
                    & (b_lo_x <= b_hi_x[i] + TOL) & (b_hi_x >= b_lo_x[i] - TOL)
                    & (b_lo_y <= b_hi_y[i] + TOL) & (b_hi_y >= b_lo_y[i] - TOL))
            cand[i] = False
            js = np.flatnonzero(cand)
            if js.size == 0:
                break
            n_lo_x = np.maximum(b_lo_x[i], b_lo_x[js])
            n_hi_x = np.minimum(b_hi_x[i], b_hi_x[js])
            n_lo_y = np.maximum(b_lo_y[i], b_lo_y[js])
            n_hi_y = np.minimum(b_hi_y[i], b_hi_y[js])
            # keep whichever merge leaves the most slack, so i can keep growing
            k = int(np.argmax((n_hi_x - n_lo_x) + (n_hi_y - n_lo_y)))
            j = int(js[k])
            b_lo_x[i], b_hi_x[i] = n_lo_x[k], n_hi_x[k]
            b_lo_y[i], b_hi_y[i] = n_lo_y[k], n_hi_y[k]
            members[i].extend(members[j])
            members[j] = []
            alive[j] = False


def _absorb(px, py, members, alive, lo_x, hi_x, lo_y, hi_y):
    """Drop a FOV when every one of its components also fits a surviving FOV."""
    order = sorted(np.flatnonzero(alive), key=lambda i: len(members[i]))
    for i in order:
        if not alive[i] or alive.sum() <= 1:
            continue
        others = np.flatnonzero(alive)
        others = others[others != i]
        ox, oy = px[others], py[others]

        dest = []
        for m in members[i]:
            hit = ((ox >= lo_x[m] - SAFE) & (ox <= hi_x[m] + SAFE)
                   & (oy >= lo_y[m] - SAFE) & (oy <= hi_y[m] + SAFE))
            w = np.flatnonzero(hit)
            if w.size == 0:
                dest = None
                break
            dest.append(int(others[w[0]]))
        if dest is None:
            continue

        for m, d in zip(members[i], dest):
            members[d].append(m)
        members[i] = []
        alive[i] = False


def _dissolve(box, members, alive, lo_x, hi_x, lo_y, hi_y):
    """Delete a FOV when its components can be handed to FOVs that still have slack.

    Stronger than `_absorb`: a receiving FOV does not have to already sit on the
    component, it only has to be *able* to shift onto it, i.e. its feasible box
    still meets the component's. Boxes are tightened as the plan is built so the
    receivers stay feasible for everything they take on.
    """
    dropped = False
    changed = True
    while changed:
        changed = False
        for i in sorted(np.flatnonzero(alive), key=lambda z: len(members[z])):
            if not alive[i] or alive.sum() <= 1:
                continue
            others = np.flatnonzero(alive)
            others = others[others != i]
            trial = {int(j): (box[0][j], box[1][j], box[2][j], box[3][j]) for j in others}

            plan = []
            for m in members[i]:
                fit = None
                for j in others:
                    a, b, c, d = trial[int(j)]
                    n_a, n_b = max(a, lo_x[m]), min(b, hi_x[m])
                    n_c, n_d = max(c, lo_y[m]), min(d, hi_y[m])
                    if n_a <= n_b + TOL and n_c <= n_d + TOL:
                        fit = (int(j), n_a, n_b, n_c, n_d)
                        break
                if fit is None:
                    plan = None
                    break
                trial[fit[0]] = fit[1:]
                plan.append(fit[0])
            if plan is None:
                continue

            for m, j in zip(members[i], plan):
                members[j].append(m)
                box[0][j], box[1][j], box[2][j], box[3][j] = trial[j]
            members[i] = []
            alive[i] = False
            changed = dropped = True
    return dropped


def _cover_cluster(pos, lo_x, hi_x, lo_y, hi_y):
    """Cover one cluster with as few FOVs as possible.

    Returns [{box, comp}, ...] where comp indexes into the group arrays.
    """
    sub = (lo_x[pos], hi_x[pos], lo_y[pos], hi_y[pos])

    best_alive = best_members = best_box = None
    for flip_x in (False, True):
        for flip_y in (False, True):
            for swap in (False, True):
                members = _strip_cover(*sub, flip_x, flip_y, swap)
                box = _boxes_of(members, *sub)
                alive = np.ones(len(members), dtype=bool)

                # fuse everything that still shares a feasible centre, then hand
                # off whole FOVs into the slack that fusing left behind
                _merge(box, members, alive)
                if _dissolve(box, members, alive, *sub):
                    _merge(box, members, alive)
                px = (box[0] + box[1]) / 2.0
                py = (box[2] + box[3]) / 2.0
                _absorb(px, py, members, alive, *sub)

                if best_alive is None or alive.sum() < best_alive.sum():
                    best_alive, best_members, best_box = alive, members, box

    out = []
    for i in np.flatnonzero(best_alive):
        ms = best_members[i]
        out.append({'lo_x': sub[0][ms].max(), 'hi_x': sub[1][ms].min(),
                    'lo_y': sub[2][ms].max(), 'hi_y': sub[3][ms].min(),
                    'comp': [int(pos[m]) for m in ms]})
    return out


# --------------------------------------------------------------------------
# oversized components -- covered by a union of FOVs (checker allows this)
# --------------------------------------------------------------------------

def _tile_big(idx, tl_x, tl_y, br_x, br_y, ew, eh):
    n_x = max(1, math.ceil((br_x - tl_x) / ew))
    n_y = max(1, math.ceil((br_y - tl_y) / eh))
    out = []
    for gx in range(n_x):
        for gy in range(n_y):
            fx = min(tl_x + ew / 2 + gx * ew, br_x - ew / 2)
            fy = min(tl_y + eh / 2 + gy * eh, br_y - eh / 2)
            out.append({'lo_x': fx, 'hi_x': fx, 'lo_y': fy, 'hi_y': fy,
                        'comp': [int(idx)]})
    return out


# --------------------------------------------------------------------------
# 4. routing
# --------------------------------------------------------------------------

def _snake(fovs, eh, entry, travel):
    """Boustrophedon order over FOVs, entered from the corner nearest `entry`."""
    if len(fovs) <= 1:
        return list(fovs)

    cx = np.array([(f['lo_x'] + f['hi_x']) / 2.0 for f in fovs])
    cy = np.array([(f['lo_y'] + f['hi_y']) / 2.0 for f in fovs])
    row = np.floor((cy - cy.min()) / max(eh, 1e-9)).astype(int)

    best, best_cost = None, None
    for top_down in (False, True):
        for first_right in (False, True):
            seq = []
            rows = sorted(set(row.tolist()), reverse=top_down)
            for k, r in enumerate(rows):
                band = np.flatnonzero(row == r)
                rightward = first_right if k % 2 == 0 else not first_right
                seq.extend(band[np.argsort(cx[band] * (1 if rightward else -1),
                                           kind='stable')])
            cost = float(travel(entry[0], entry[1], cx[seq[0]], cy[seq[0]]))
            if best_cost is None or cost < best_cost:
                best, best_cost = seq, cost
    return [fovs[i] for i in best]


def _chain_clusters(clusters, entry, travel):
    """Nearest-neighbour chain over clusters, by centroid, under the travel model."""
    rest = list(range(len(clusters)))
    cent = [(float(np.mean([(f['lo_x'] + f['hi_x']) / 2.0 for f in c])),
             float(np.mean([(f['lo_y'] + f['hi_y']) / 2.0 for f in c]))) for c in clusters]
    out, at = [], entry
    while rest:
        cx = np.array([cent[i][0] for i in rest])
        cy = np.array([cent[i][1] for i in rest])
        k = int(np.argmin(travel(at[0], at[1], cx, cy)))
        i = rest.pop(k)
        out.append(i)
        at = cent[i]
    return out


def _relax(fovs, entry, travel):
    """Nudge each FOV inside its feasible box toward its neighbours."""
    for _ in range(2):
        for i, f in enumerate(fovs):
            prev = (fovs[i - 1]['x'], fovs[i - 1]['y']) if i else entry
            nxt = (fovs[i + 1]['x'], fovs[i + 1]['y']) if i + 1 < len(fovs) else prev
            tx = (prev[0] + nxt[0]) / 2.0
            ty = (prev[1] + nxt[1]) / 2.0
            f['x'] = min(max(tx, f['lo_x']), f['hi_x'])
            f['y'] = min(max(ty, f['lo_y']), f['hi_y'])
    return fovs


# --------------------------------------------------------------------------
# group driver
# --------------------------------------------------------------------------

def _extend(fovs, pool, lo_x, hi_x, lo_y, hi_y):
    """Rule 2.4: a Barcode FOV may also carry ordinary parts, and fov.type stays
    max(type) = Barcode. Hand it every same-step part it can still reach, which
    drops those parts out of the type=0 covering problem for free.

    Returns the components taken."""
    taken = []
    for f in fovs:
        if pool.size == 0:
            break
        a, b = f['lo_x'], f['hi_x']
        c, d = f['lo_y'], f['hi_y']
        keep = np.ones(pool.size, dtype=bool)
        for k, m in enumerate(pool):
            n_a, n_b = max(a, lo_x[m]), min(b, hi_x[m])
            n_c, n_d = max(c, lo_y[m]), min(d, hi_y[m])
            if n_a <= n_b + TOL and n_c <= n_d + TOL:
                a, b, c, d = n_a, n_b, n_c, n_d
                f['comp'].append(int(m))
                taken.append(int(m))
                keep[k] = False
        f['lo_x'], f['hi_x'], f['lo_y'], f['hi_y'] = a, b, c, d
        f['x'] = (a + b) / 2.0
        f['y'] = (c + d) / 2.0
        pool = pool[keep]
    return taken


def _fiducial_fovs(idx, geom):
    """Rule 2.3/3.2: one dedicated FOV per fiducial, centred on it, in
    registration order. The box is pinned to a point so `_relax` cannot drift
    it off centre, and the components stay in index order."""
    tl_x, tl_y, br_x, br_y = geom
    out = []
    for i in sorted(int(v) for v in idx):
        fx = (tl_x[i] + br_x[i]) / 2.0
        fy = (tl_y[i] + br_y[i]) / 2.0
        out.append({'lo_x': fx, 'hi_x': fx, 'lo_y': fy, 'hi_y': fy,
                    'x': fx, 'y': fy, 'comp': [i]})
    return out


def _cover_group(idx, geom, ew, eh, grid, entry, travel):
    """Cluster one group and cover each cluster.

    `ew`/`eh` are per-component effective windows (side=1 parts get the half
    FOV), so a group may legally mix sides; `grid` is the cell size used for
    clustering and for snake rows.
    """
    tl_x, tl_y, br_x, br_y = geom

    lo_x = br_x - ew / 2.0
    hi_x = tl_x + ew / 2.0
    lo_y = br_y - eh / 2.0
    hi_y = tl_y + eh / 2.0

    big = (lo_x[idx] > hi_x[idx] + 1e-9) | (lo_y[idx] > hi_y[idx] + 1e-9)
    big_idx = idx[big]
    fit_idx = idx[~big]

    fovs = []
    for i in big_idx:
        fovs.extend(_tile_big(i, tl_x[i], tl_y[i], br_x[i], br_y[i], ew[i], eh[i]))

    # An oversized component is re-inspected once per tile, so its tiles are a
    # sunk cost -- let them carry any same-group component that already fits,
    # which costs nothing and removes that component from the covering problem.
    if fovs and len(fit_idx):
        take = np.zeros(len(fit_idx), dtype=bool)
        for f in fovs:
            free = np.flatnonzero(~take)
            if free.size == 0:
                break
            m = fit_idx[free]
            hit = ((lo_x[m] <= f['lo_x']) & (hi_x[m] >= f['hi_x'])
                   & (lo_y[m] <= f['lo_y']) & (hi_y[m] >= f['hi_y']))
            f['comp'].extend(int(v) for v in m[hit])
            take[free[hit]] = True
        fit_idx = fit_idx[~take]

    if len(fit_idx):
        g_lo_x, g_hi_x = lo_x[fit_idx], hi_x[fit_idx]
        g_lo_y, g_hi_y = lo_y[fit_idx], hi_y[fit_idx]
        cx = (tl_x[fit_idx] + br_x[fit_idx]) / 2.0
        cy = (tl_y[fit_idx] + br_y[fit_idx]) / 2.0

        blobs = [_cover_cluster(np.array(b), g_lo_x, g_hi_x, g_lo_y, g_hi_y)
                 for b in _clusters(cx, cy, *grid)]
        # positions are local to fit_idx; map back to component indices
        for blob in blobs:
            for f in blob:
                f['comp'] = [int(fit_idx[p]) for p in f['comp']]

        at = entry
        for k in _chain_clusters(blobs, entry, travel):
            ordered = _snake(blobs[k], grid[1], at, travel)
            fovs.extend(ordered)
            last = ordered[-1]
            at = ((last['lo_x'] + last['hi_x']) / 2.0,
                  (last['lo_y'] + last['hi_y']) / 2.0)

    for f in fovs:
        f['x'] = (f['lo_x'] + f['hi_x']) / 2.0
        f['y'] = (f['lo_y'] + f['hi_y']) / 2.0
    return fovs


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def solve(data_folder: str) -> int:
    comp = pd.read_csv(os.path.join(data_folder, 'input_component.csv'))
    size = pd.read_csv(os.path.join(data_folder, 'input_size.csv'))
    param = pd.read_csv(os.path.join(data_folder, 'input_parameter.csv'))

    fov_w = float(size['fov_size'].iloc[0])
    fov_h = float(size['fov_size'].iloc[1])

    n = len(comp)
    tl_x = comp['tl_x'].values.astype(float)
    tl_y = comp['tl_y'].values.astype(float)
    br_x = comp['br_x'].values.astype(float)
    br_y = comp['br_y'].values.astype(float)
    geom = (tl_x, tl_y, br_x, br_y)

    c_type = comp['type'].astype(int).values
    c_time = comp['time'].values.astype(float)
    c_side = comp['side'].astype(int).values if 'side' in comp.columns else np.zeros(n, int)
    c_step = comp['step'].astype(int).values if 'step' in comp.columns else np.zeros(n, int)

    travel = _make_travel([float(param['v_x'].iloc[0]), float(param['v_y'].iloc[0]),
                           float(param['a_x'].iloc[0]), float(param['a_y'].iloc[0])])

    # per-component window: side=1 (side view) is inspected through the half FOV
    ew = np.where(c_side == 1, fov_w / 2.0, fov_w)
    eh = np.where(c_side == 1, fov_h / 2.0, fov_h)

    # Fiducials come first, one per FOV, centred, in registration order.
    fovs = _fiducial_fovs(np.flatnonzero(c_type == 2), geom)
    if fovs:
        at = (fovs[-1]['x'], fovs[-1]['y'])
        start_step = int(c_step[fovs[-1]['comp'][0]])
    else:
        at, start_step = (0.0, 0.0), None

    keys = {}
    for i in range(n):
        if c_type[i] == 2:
            continue
        side = c_side[i] if SPLIT_SIDE else 0
        keys.setdefault((c_type[i], c_step[i], side), []).append(i)

    lo_x, hi_x = br_x - ew / 2.0, tl_x + ew / 2.0
    lo_y, hi_y = br_y - eh / 2.0, tl_y + eh / 2.0
    done = np.zeros(n, dtype=bool)

    for t, step, side in _group_sequence(set(keys), start_step):
        idx = np.array([i for i in keys[(t, step, side)] if not done[i]])
        if idx.size == 0:
            continue
        done[idx] = True
        # clustering grid: the group's coarsest window, so blobs stay connected
        grid = (fov_w / 2.0, fov_h / 2.0) if (SPLIT_SIDE and side == 1) else (fov_w, fov_h)
        group = _cover_group(idx, geom, ew, eh, grid, at, travel)

        if t == 1 and group:
            pool = np.array([i for i in keys.get((0, step, side), ())
                             if not done[i] and lo_x[i] <= hi_x[i] and lo_y[i] <= hi_y[i]],
                            dtype=int)
            if pool.size:
                done[_extend(group, pool, lo_x, hi_x, lo_y, hi_y)] = True

        if group:
            _relax(group, at, travel)
            fovs.extend(group)
            at = (group[-1]['x'], group[-1]['y'])

    rows = [{'x': f['x'], 'y': f['y'],
             'comp_idx': json.dumps(sorted((int(i) for i in f['comp']),
                                          key=lambda i: -c_time[i] if LPT else i))}
            for f in fovs if f['comp']]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_folder, 'output_fov.csv'), index=True)
    return len(df)
