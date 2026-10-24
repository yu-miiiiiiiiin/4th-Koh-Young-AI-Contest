"""FOV placement: cover every component with as few shots as possible.

A FOV has a fixed size, so "FOV centred at p covers component c" is just

    p.x in [c.br_x - ew/2, c.tl_x + ew/2]        ew = fov_w   (side=0, rule 2.2)
    p.y in [c.br_y - eh/2, c.tl_y + eh/2]        ew = fov_w/2 (side=1, rule 2.6)

so each component is one axis-aligned *feasible-centre box* and the covering
problem is minimum piercing: stab all boxes with the fewest points.  Axis-
aligned boxes have Helly number 2, so a set of components fits one shot exactly
when their boxes intersect pairwise -- every routine below is therefore a box
intersection test and nothing more.

The pipeline per piece is strip-sweep -> merge -> dissolve -> absorb, run from
all eight axis/direction frames and keeping the best.  Those eight frames are
the same dihedral group POMO uses for instance augmentation (pomo.FRAMES),
applied here to the geometric sub-problem.

A "piece" is not a cluster in any heuristic sense -- see `components_of`.  The
only partition imposed from outside is by step level, which rule 2.7 forces;
type is deliberately *not* partitioned, because rule 2.4 lets one FOV carry a
Barcode and ordinary parts at once.

Output FOVs carry their *box*, not just a point, so the router downstream can
still slide them to shave stage travel (pomo.relax).
"""
import math

import numpy as np

TOL = 1e-9
SAFE = 1e-7          # checker's containment test runs at 1e-5; stay inside it


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
    already sit on the component."""
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


def cover_piece(pos, lo_x, hi_x, lo_y, hi_y):
    """Cover one interaction piece with as few FOVs as possible.

    Eight restart frames, each polished, best kept -- the covering counterpart of
    POMO's take-the-argmax-over-augmented-rollouts inference.
    Returns [{box, comp}].
    """
    sub = (lo_x[pos], hi_x[pos], lo_y[pos], hi_y[pos])

    best_alive = best_members = None
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

                if best_alive is None or alive.sum() < best_alive.sum():
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


def cover_level(idx, geom, ew, eh, grid):
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
            for f in cover_piece(np.array(blob), *g):
                f['comp'] = [int(fit_idx[p]) for p in f['comp']]
                fovs.append(f)

    for f in fovs:
        f['x'] = (f['lo_x'] + f['hi_x']) / 2.0
        f['y'] = (f['lo_y'] + f['hi_y']) / 2.0
    return fovs
