"""FOV placement: cover every component with as few shots as possible.

A FOV has a fixed size, so "FOV centred at p covers component c" is just

    p.x in [c.br_x - ew/2, c.tl_x + ew/2]        ew = fov_w   (side=0, rule 2.2)
    p.y in [c.br_y - eh/2, c.tl_y + eh/2]        ew = fov_w/2 (side=1, rule 2.6)

so each component is one axis-aligned *feasible-centre box* and the covering
problem is minimum piercing: stab all boxes with the fewest points.  Axis-
aligned boxes have Helly number 2, so a set of components fits one shot exactly
when their boxes intersect pairwise -- every routine below is therefore a box
intersection test and nothing more.

Construction per piece is strip-sweep -> merge -> dissolve -> absorb, run from
all eight axis/direction frames, best kept.  `shrink` then runs a ruin-and-
recreate pass over the result; see its docstring for why that is the pass that
matters on these boards.

The only partition imposed from outside is by step level, which rule 2.7
forces.  Type is deliberately *not* partitioned, because rule 2.4 lets one FOV
carry a Barcode and ordinary parts at once.

Output FOVs carry their *box*, not just a point, so the router downstream can
still slide them to shave stage travel (route.relax).

Measured on the 96 practice boards: covering the side=1 parts alone needs
within a few percent of the FOVs the whole board needs (39.8 vs 45.8 on
S00, 45.9 vs 46.5 on S04).  The half-size side window is what sets the FOV
count; top-only parts ride along almost for free.  That is why `shrink` spends
its budget on pieces that hold side parts.
"""
import math

import numpy as np

TOL = 1e-9
SAFE = 1e-7          # checker's containment test runs at 1e-5; stay inside it
TIE_BREAK = 'first'  # offline-tunable choice among equal-cardinality covers


def _tie_value(box, members, alive, weights=None):
    """Secondary objective used only when two covers use the same FOV count."""
    idx = np.flatnonzero(alive)
    if TIE_BREAK == 'first':
        return 0.0
    width = np.maximum(0.0, box[1][idx] - box[0][idx])
    height = np.maximum(0.0, box[3][idx] - box[2][idx])
    if TIE_BREAK == 'slack':
        return -float(np.sum(width + height))
    if TIE_BREAK == 'area':
        return -float(np.sum(width * height))
    sizes = np.array([len(members[i]) for i in idx], dtype=float)
    if TIE_BREAK == 'balance':
        return float(np.var(sizes))
    if TIE_BREAK == 'packed':
        return -float(np.sum(sizes * sizes))
    if TIE_BREAK in ('load_balance', 'load_packed'):
        if weights is None:
            return float(np.var(sizes)) if TIE_BREAK == 'load_balance' else -float(np.sum(sizes * sizes))
        loads = np.array([sum(weights[m] for m in members[i]) for i in idx], dtype=float)
        return (float(np.var(loads)) if TIE_BREAK == 'load_balance'
                else -float(np.sum(loads * loads)))
    if TIE_BREAK == 'compact':
        if len(idx) < 2:
            return 0.0
        x = (box[0][idx] + box[1][idx]) / 2.0
        y = (box[2][idx] + box[3][idx]) / 2.0
        d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        np.fill_diagonal(d, np.inf)
        return float(np.sum(np.min(d, axis=1)))
    raise ValueError(f'unknown cover tie-break {TIE_BREAK!r}')


# --------------------------------------------------------------------------
# minimum piercing
# --------------------------------------------------------------------------

def strip_cover(lo_x, hi_x, lo_y, hi_y, flip_x, flip_y, swap):
    """Piercing set from a strip sweep: exact 1-D greedy inside each strip.

    Placing the strip line at the smallest upper bound still open is what makes
    the inner sweep optimal in one dimension; `flip_*`/`swap` re-frame the board
    (negating an interval just swaps its bounds) for eight cheap restarts.
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
        sy = hi_y[idx].min()
        strip = idx[lo_y[idx] <= sy + TOL]

        open_ = np.ones(len(strip), dtype=bool)
        while open_.any():
            rest = strip[open_]
            sx = hi_x[rest].min()
            sel = lo_x[rest] <= sx + TOL
            out.append(list(rest[sel]))
            live[rest[sel]] = False
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
    """Fuse FOVs whose feasible boxes still intersect (greedy clique growth)."""
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
            # take the merge that leaves the most slack, so i can keep growing
            k = int(np.argmax((n_hi_x - n_lo_x) + (n_hi_y - n_lo_y)))
            j = int(js[k])
            b_lo_x[i], b_hi_x[i] = n_lo_x[k], n_hi_x[k]
            b_lo_y[i], b_hi_y[i] = n_lo_y[k], n_hi_y[k]
            members[i].extend(members[j])
            members[j] = []
            alive[j] = False


def _dissolve(box, members, alive, lo_x, hi_x, lo_y, hi_y):
    """Delete a FOV by handing its components to FOVs that can still shift onto
    them -- receivers are tightened as the hand-off is planned, so the plan is
    always feasible.  Stronger than `_absorb`, which only accepts receivers that
    already sit on the component.

    The candidate scan is vectorised over receivers: it is the hot loop of the
    whole solver (92% of cover time before, 4.9M scalar min/max calls on a dense
    board), and one numpy pass per component replaces the inner loop entirely.
    Semantics are unchanged -- the receiver taken is still the lowest-indexed
    one that fits, and it is still tightened before the next component looks.
    """
    b_lo_x, b_hi_x, b_lo_y, b_hi_y = box
    dropped = False
    changed = True
    while changed:
        changed = False
        for i in sorted(np.flatnonzero(alive), key=lambda z: len(members[z])):
            if not alive[i] or alive.sum() <= 1:
                continue
            others = np.flatnonzero(alive)
            others = others[others != i]
            a = b_lo_x[others].copy()
            b = b_hi_x[others].copy()
            c = b_lo_y[others].copy()
            d = b_hi_y[others].copy()

            plan = []
            for m in members[i]:
                n_a = np.maximum(a, lo_x[m])
                n_b = np.minimum(b, hi_x[m])
                n_c = np.maximum(c, lo_y[m])
                n_d = np.minimum(d, hi_y[m])
                w = np.flatnonzero((n_a <= n_b + TOL) & (n_c <= n_d + TOL))
                if w.size == 0:
                    plan = None
                    break
                k = int(w[0])
                a[k], b[k], c[k], d[k] = n_a[k], n_b[k], n_c[k], n_d[k]
                plan.append(k)
            if plan is None:
                continue

            for m, k in zip(members[i], plan):
                j = int(others[k])
                members[j].append(m)
                b_lo_x[j], b_hi_x[j] = a[k], b[k]
                b_lo_y[j], b_hi_y[j] = c[k], d[k]
            members[i] = []
            alive[i] = False
            changed = dropped = True
    return dropped


def _absorb(px, py, members, alive, lo_x, hi_x, lo_y, hi_y):
    """Drop a FOV when every component of it also fits a surviving FOV as it
    already stands."""
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


def cover_piece(pos, lo_x, hi_x, lo_y, hi_y, weights=None):
    """Cover one interaction piece with as few FOVs as possible.

    Eight restart frames, each polished, best kept -- the covering counterpart of
    POMO's take-the-argmax-over-augmented-rollouts inference.
    Returns [{box, comp}].
    """
    sub = (lo_x[pos], hi_x[pos], lo_y[pos], hi_y[pos])
    sub_weights = None if weights is None else np.asarray(weights)[pos]

    best_alive = best_members = None
    best_key = None
    for flip_x in (False, True):
        for flip_y in (False, True):
            for swap in (False, True):
                members = strip_cover(*sub, flip_x, flip_y, swap)
                box = _boxes_of(members, *sub)
                alive = np.ones(len(members), dtype=bool)

                # merge frees slack that lets dissolve empty another FOV, whose
                # slack lets merge grow again.  One alternation reaches the fixed
                # point: a second and a fourth round were measured over all 96
                # boards and removed exactly zero further FOVs.
                _merge(box, members, alive)
                if _dissolve(box, members, alive, *sub):
                    _merge(box, members, alive)
                _absorb((box[0] + box[1]) / 2.0, (box[2] + box[3]) / 2.0,
                        members, alive, *sub)

                key = (int(alive.sum()), _tie_value(box, members, alive, sub_weights))
                if best_key is None or key < best_key:
                    best_key = key
                    best_alive, best_members = alive, members

    out = []
    for i in np.flatnonzero(best_alive):
        ms = best_members[i]
        out.append({'lo_x': sub[0][ms].max(), 'hi_x': sub[1][ms].min(),
                    'lo_y': sub[2][ms].max(), 'hi_y': sub[3][ms].min(),
                    'comp': [int(pos[m]) for m in ms]})
    return out


# --------------------------------------------------------------------------
# splitting the board into pieces that provably cannot interact
# --------------------------------------------------------------------------

def components_of(cx, cy, ew, eh):
    """Split the board into pieces that cannot interact -- exactly, not heuristically.

    Two components can share one shot only if their feasible-centre boxes meet,
    and box_A.x = [br_x - ew_A/2, tl_x + ew_A/2] meets box_B.x only when the two
    centres lie within (ew_A + ew_B)/2 <= fov_w of each other (likewise in y).
    So bucket centres into fov-sized cells and 8-connect them: anything that
    could ever share a FOV lands in the same piece, and the piercing problem
    splits with *no* loss of optimality.

    This is not clustering in the yumin_solver_2 sense -- nothing is grouped by
    similarity and nothing is decided here.  It is the connected-component
    decomposition the constraint graph already has, and skipping it would only
    make the same search slower.  It also means no FOV is ever generated over
    bare laminate, because every piece is anchored on real components.
    """
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
        for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):   # forward half only
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
# special components
# --------------------------------------------------------------------------

def tile_big(idx, tl_x, tl_y, br_x, br_y, ew, eh):
    """Rule 2.5: a component wider than its window is covered by a union of FOVs.

    ceil(w/ew) * ceil(h/eh) is the minimum for a rectangle, and the checker
    re-inspects the component once per tile, so the duplicated inspection time is
    unavoidable -- but the tiles are then a sunk cost worth refilling.
    """
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


def fiducial_fovs(idx, geom):
    """Rules 2.3 / 3.2: one dedicated FOV per fiducial, centred exactly on it,
    emitted in registration order.  The box is pinned to a point so no later
    relaxation can drift it off centre."""
    tl_x, tl_y, br_x, br_y = geom
    out = []
    for i in sorted(int(v) for v in idx):
        fx = (tl_x[i] + br_x[i]) / 2.0
        fy = (tl_y[i] + br_y[i]) / 2.0
        out.append({'lo_x': fx, 'hi_x': fx, 'lo_y': fy, 'hi_y': fy,
                    'x': fx, 'y': fy, 'comp': [i]})
    return out


def cover_level(idx, geom, ew, eh, grid, weights=None):
    """Cover one step level -- every non-fiducial component on it, in one problem.

    Barcode and Normal parts are covered *together*, not in separate passes.
    Rule 2.4 lets one FOV hold both and keeps fov.type = max(member types), so
    splitting by type would be an invented constraint: it would force a Barcode
    to claim its own shot and then bolt neighbouring parts on afterwards, which
    is strictly weaker than letting the piercing choose the pairing itself.

    Step is the one split that *is* forced -- rule 2.7 forbids mixing step
    levels inside a FOV -- so it is the only partition here.

    `ew`/`eh` are per-component windows, so a level may legally mix sides
    (rule 1.3 captures Top and Side at one position).  Returned FOVs are
    unordered: the visiting order is the router's job (pomo.py).
    """
    tl_x, tl_y, br_x, br_y = geom

    lo_x, hi_x = br_x - ew / 2.0, tl_x + ew / 2.0
    lo_y, hi_y = br_y - eh / 2.0, tl_y + eh / 2.0

    big = (lo_x[idx] > hi_x[idx] + 1e-9) | (lo_y[idx] > hi_y[idx] + 1e-9)
    big_idx, fit_idx = idx[big], idx[~big]

    fovs = []
    for i in big_idx:
        fovs.extend(tile_big(i, tl_x[i], tl_y[i], br_x[i], br_y[i], ew[i], eh[i]))

    # tiles of an oversized part are already paid for -- refill them
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
        g = (lo_x[fit_idx], hi_x[fit_idx], lo_y[fit_idx], hi_y[fit_idx])
        cx = (tl_x[fit_idx] + br_x[fit_idx]) / 2.0
        cy = (tl_y[fit_idx] + br_y[fit_idx]) / 2.0
        for blob in components_of(cx, cy, *grid):
            local_weights = None if weights is None else np.asarray(weights)[fit_idx]
            for f in cover_piece(np.array(blob), *g, weights=local_weights):
                f['comp'] = [int(fit_idx[p]) for p in f['comp']]
                fovs.append(f)

    for f in fovs:
        f['x'] = (f['lo_x'] + f['hi_x']) / 2.0
        f['y'] = (f['lo_y'] + f['hi_y']) / 2.0
    return fovs


# --------------------------------------------------------------------------
# ruin and recreate
# --------------------------------------------------------------------------

def _grow(seed, pool, taken, lo_x, hi_x, lo_y, hi_y):
    """Grow one shot out of `seed`: keep absorbing the component that leaves the
    running intersection the most room, which is the greedy that keeps the shot
    able to take one more."""
    a, b, c, d = lo_x[seed], hi_x[seed], lo_y[seed], hi_y[seed]
    grp = [seed]
    taken[seed] = True
    while True:
        free = pool[~taken[pool]]
        if free.size == 0:
            break
        n_a = np.maximum(a, lo_x[free])
        n_b = np.minimum(b, hi_x[free])
        n_c = np.maximum(c, lo_y[free])
        n_d = np.minimum(d, hi_y[free])
        room = (n_b - n_a) + (n_d - n_c)
        room[(n_a > n_b + TOL) | (n_c > n_d + TOL)] = -np.inf
        k = int(np.argmax(room))
        if not np.isfinite(room[k]):
            break
        j = int(free[k])
        a, b, c, d = n_a[k], n_b[k], n_c[k], n_d[k]
        grp.append(j)
        taken[j] = True
    return grp, (a, b, c, d)


def _clique_cover(pool, lo_x, hi_x, lo_y, hi_y, rng, tries, limit):
    """Re-cover `pool` from scratch; best of `tries` randomised seed orders.

    Stops early once a run needs `limit` shots, since only a strictly smaller
    cover is ever accepted by the caller.
    """
    best = None
    tight = (hi_x[pool] - lo_x[pool]) + (hi_y[pool] - lo_y[pool])
    base = pool[np.argsort(tight)]            # most constrained part seeds first
    for t in range(tries):
        order = base if t == 0 else rng.permutation(pool)
        taken = np.zeros(len(lo_x), dtype=bool)
        out = []
        for m in order:
            if taken[m]:
                continue
            if len(out) >= limit:
                out = None
                break
            out.append(_grow(int(m), pool, taken, lo_x, hi_x, lo_y, hi_y))
        if out is not None and (best is None or len(out) < len(best)):
            best = out
            if len(best) <= 1:
                break
    return best


def shrink(fovs, lo_x, hi_x, lo_y, hi_y, budget, clock, rng, pinned=()):
    """Ruin a handful of neighbouring shots, recreate, keep the cover if it got
    smaller.  In place; returns the number of FOVs removed.

    Why this and not a longer construction: the eight-frame construction is
    already within a couple of percent of what any amount of re-running it
    finds, because each frame commits to a global sweep direction.  What it
    cannot do is revisit one crowded neighbourhood on its own terms.  Ruining
    3-6 adjacent shots and re-covering just those parts is exactly that, and it
    is cheap -- the ruined set is tens of parts, not thousands.

    `pinned` marks FOVs that must survive untouched (oversized-part tiles,
    whose positions rule 2.5 fixes).
    """
    pin = set(pinned)
    free = [i for i in range(len(fovs)) if i not in pin]
    if len(free) < 3:
        return 0

    dropped = 0
    while clock() < budget:
        seed = fovs[free[rng.integers(len(free))]]
        sx, sy = seed['x'], seed['y']
        step = seed['step']
        near = [i for i in free if fovs[i]['step'] == step]
        if len(near) < 3:
            return dropped
        near.sort(key=lambda i: (fovs[i]['x'] - sx) ** 2 + (fovs[i]['y'] - sy) ** 2)
        take = near[:int(rng.integers(3, 7))]

        pool = np.array([m for i in take for m in fovs[i]['comp']], dtype=np.int64)
        got = _clique_cover(pool, lo_x, hi_x, lo_y, hi_y, rng, 4, len(take))
        if got is None or len(got) >= len(take):
            continue

        keep = set(take)
        rest = [f for i, f in enumerate(fovs) if i not in keep]
        for grp, (a, b, c, d) in got:
            rest.append({'lo_x': a, 'hi_x': b, 'lo_y': c, 'hi_y': d,
                         'x': (a + b) / 2.0, 'y': (c + d) / 2.0,
                         'comp': [int(v) for v in grp], 'step': step})
        dropped += len(take) - len(got)
        fovs[:] = rest
        pin = {i for i, f in enumerate(fovs) if f.get('pinned')}
        free = [i for i in range(len(fovs)) if i not in pin]
        if len(free) < 3:
            break
    return dropped
