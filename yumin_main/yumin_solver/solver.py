"""
Solver for ICCAS 2026 KohYoung AI Contest.

Strategy (group-wise Riemann tiling):

1. Components are scanned group by group, in this order:

       type=2 (Fiducial)              -- one exclusive FOV each
       type=1 (Barcode)   side=0, 1
       type=0 (Normal)    step=0 side=0, step=0 side=1,
                          step=1 side=0, step=1 side=1

   Groups are pure in (type, step, side), so:
     - FOV type is non-increasing over the whole sequence (order rule),
     - no FOV ever mixes step levels (step rule),
     - the 5s Z-transition penalty is paid only at group boundaries.

2. Each group is covered by tiling the area spanned by its components with a
   regular grid of FOVs (a Riemann-sum style fill). Grid pitch is the group's
   effective FOV: full FOV for side=0, half FOV for side=1. Empty tiles are
   dropped, and tiles are visited in snake order to keep travel short.

3. Components straddling a tile boundary (all 4 corners not inside one tile)
   are left over, and get extra FOVs placed on their own centers, absorbing
   any other leftovers of the same group that fall fully inside. Components
   larger than one FOV are tiled across several FOVs (the checker accepts a
   union of FOVs for those).

Usage:
    python 2_run.py --solver yumin_solver
"""
import os
import json
import math
import numpy as np
import pandas as pd

TOL = 1e-6
SHARE_SIDES = True
SHARE_BARCODE = True
LPT = True
# Check several grid origins; this keeps grid tiling but cuts boundary leftovers.
# Eight phases passed full train/validation/holdout comparisons, including its
# extra runtime in the charged score.  Set to 1 for the earlier fixed origin.
GRID_PHASES = 8


def _snake_sorted(cells):
    """Order (i, j) tiles row by row, alternating direction each row."""
    rows = {}
    for c in cells:
        rows.setdefault(c[1], []).append(c)
    out = []
    for k, j in enumerate(sorted(rows)):
        out.extend(sorted(rows[j], key=lambda c: c[0], reverse=bool(k % 2)))
    return out


def _nearest_neighbor(fovs, start):
    """Greedy nearest-neighbor chain over FOVs, starting from point `start`."""
    remaining = list(fovs)
    out = []
    lx, ly = start
    while remaining:
        b = min(range(len(remaining)),
                key=lambda j: (remaining[j]['x'] - lx) ** 2 + (remaining[j]['y'] - ly) ** 2)
        f = remaining.pop(b)
        out.append(f)
        lx, ly = f['x'], f['y']
    return out


def _tile_big(idx, tl_x, tl_y, br_x, br_y, eff_w, eff_h):
    """Cover one oversized component with a grid of FOVs (union coverage)."""
    n_x = max(1, math.ceil((br_x - tl_x) / eff_w))
    n_y = max(1, math.ceil((br_y - tl_y) / eff_h))
    out = []
    for gx in range(n_x):
        for gy in range(n_y):
            fx = min(tl_x + eff_w / 2 + gx * eff_w, br_x - eff_w / 2)
            fy = min(tl_y + eff_h / 2 + gy * eff_h, br_y - eff_h / 2)
            out.append({'x': fx, 'y': fy, 'comp_idx': [idx]})
    return out


def _cover_group(members, tl_x, tl_y, br_x, br_y, eff_w, eff_h, last_xy):
    """Cover one (type, step, side) group with grid tiles + leftover FOVs."""
    if len(members) == 0:
        return []

    x0 = tl_x[members].min()
    y0 = tl_y[members].min()

    if GRID_PHASES > 1:
        best = (float('inf'), 0, 0)
        phases = min(int(GRID_PHASES), 8)
        for pi in range(phases):
            ox = x0 - pi * eff_w / phases
            it = np.floor((tl_x[members] - ox) / eff_w).astype(int)
            ib = np.floor((br_x[members] - ox) / eff_w - TOL).astype(int)
            for pj in range(phases):
                oy = y0 - pj * eff_h / phases
                jt = np.floor((tl_y[members] - oy) / eff_h).astype(int)
                jb = np.floor((br_y[members] - oy) / eff_h - TOL).astype(int)
                fits_trial = (it == ib) & (jt == jb)
                tiles = len(set(zip(it[fits_trial], jt[fits_trial])))
                cost = tiles + int((~fits_trial).sum())
                choice = (cost, pi, pj)
                if choice < best:
                    best = choice
        x0 -= best[1] * eff_w / phases
        y0 -= best[2] * eff_h / phases

    # Tile index of each corner; a component fits a tile when both corners agree.
    i_tl = np.floor((tl_x[members] - x0) / eff_w).astype(int)
    i_br = np.floor((br_x[members] - x0) / eff_w - TOL).astype(int)
    j_tl = np.floor((tl_y[members] - y0) / eff_h).astype(int)
    j_br = np.floor((br_y[members] - y0) / eff_h - TOL).astype(int)
    fits = (i_tl == i_br) & (j_tl == j_br)

    tiles = {}
    leftover = []
    for k, idx in enumerate(members):
        if fits[k]:
            tiles.setdefault((i_tl[k], j_tl[k]), []).append(int(idx))
        else:
            leftover.append(int(idx))

    fovs = [{'x': x0 + (i + 0.5) * eff_w,
             'y': y0 + (j + 0.5) * eff_h,
             'comp_idx': tiles[(i, j)]}
            for i, j in _snake_sorted(tiles.keys())]

    if not leftover:
        return fovs

    # Leftovers: one FOV per component center, absorbing the rest that fit.
    done = set()
    extra = []
    for idx in leftover:
        if idx in done:
            continue
        w, h = br_x[idx] - tl_x[idx], br_y[idx] - tl_y[idx]
        if w > eff_w + TOL or h > eff_h + TOL:
            extra.extend(_tile_big(idx, tl_x[idx], tl_y[idx], br_x[idx], br_y[idx], eff_w, eff_h))
            done.add(idx)
            continue

        fx = (tl_x[idx] + br_x[idx]) / 2.0
        fy = (tl_y[idx] + br_y[idx]) / 2.0
        a_tl_x, a_tl_y = fx - eff_w / 2, fy - eff_h / 2
        a_br_x, a_br_y = fx + eff_w / 2, fy + eff_h / 2

        taken = []
        for other in leftover:
            if other in done:
                continue
            if (a_tl_x - TOL <= tl_x[other] and a_tl_y - TOL <= tl_y[other] and
                    a_br_x + TOL >= br_x[other] and a_br_y + TOL >= br_y[other]):
                taken.append(other)
                done.add(other)
        extra.append({'x': fx, 'y': fy, 'comp_idx': taken})

    start = (fovs[-1]['x'], fovs[-1]['y']) if fovs else last_xy
    return fovs + _nearest_neighbor(extra, start)


def solve(data_folder: str) -> int:
    comp = pd.read_csv(os.path.join(data_folder, 'input_component.csv'))
    size = pd.read_csv(os.path.join(data_folder, 'input_size.csv'))

    fov_w = float(size['fov_size'].iloc[0])
    fov_h = float(size['fov_size'].iloc[1])

    n = len(comp)
    tl_x = comp['tl_x'].values.astype(float)
    tl_y = comp['tl_y'].values.astype(float)
    br_x = comp['br_x'].values.astype(float)
    br_y = comp['br_y'].values.astype(float)
    c_type = comp['type'].astype(int).values
    c_time = comp['time'].values.astype(float)
    c_side = comp['side'].astype(int).values if 'side' in comp.columns else np.zeros(n, dtype=int)
    c_step = comp['step'].astype(int).values if 'step' in comp.columns else np.zeros(n, dtype=int)

    fovs = []
    assigned = np.zeros(n, dtype=bool)

    def absorb(group, candidates):
        """Assign already-covered compatible parts to an earlier grid shot."""
        for idx in candidates:
            if assigned[idx]:
                continue
            ew = fov_w / 2.0 if c_side[idx] else fov_w
            eh = fov_h / 2.0 if c_side[idx] else fov_h
            for shot in group:
                if (tl_x[idx] >= shot['x'] - ew / 2.0 - TOL and
                    br_x[idx] <= shot['x'] + ew / 2.0 + TOL and
                    tl_y[idx] >= shot['y'] - eh / 2.0 - TOL and
                    br_y[idx] <= shot['y'] + eh / 2.0 + TOL):
                    shot['comp_idx'].append(int(idx))
                    assigned[idx] = True
                    break

    # --- Fiducials (type=2): one exclusive FOV each, no other components ---
    for idx in np.flatnonzero(c_type == 2):
        eff_w = fov_w / 2.0 if c_side[idx] == 1 else fov_w
        eff_h = fov_h / 2.0 if c_side[idx] == 1 else fov_h
        if br_x[idx] - tl_x[idx] > eff_w + TOL or br_y[idx] - tl_y[idx] > eff_h + TOL:
            fovs.extend(_tile_big(int(idx), tl_x[idx], tl_y[idx], br_x[idx], br_y[idx], eff_w, eff_h))
        else:
            fovs.append({'x': (tl_x[idx] + br_x[idx]) / 2.0,
                         'y': (tl_y[idx] + br_y[idx]) / 2.0,
                         'comp_idx': [int(idx)]})

    # --- Barcode (type=1) then Normal (type=0), each split by step and side ---
    last_xy = (fovs[-1]['x'], fovs[-1]['y']) if fovs else (0.0, 0.0)
    for t in (1, 0):
        for step in (0, 1):
            group_all = []
            for side in ((1, 0) if SHARE_SIDES else (0, 1)):
                members = np.flatnonzero((c_type == t) & (c_step == step) &
                                          (c_side == side) & (~assigned))
                if len(members) == 0:
                    continue
                eff_w = fov_w / 2.0 if side == 1 else fov_w
                eff_h = fov_h / 2.0 if side == 1 else fov_h
                group = _cover_group(members, tl_x, tl_y, br_x, br_y, eff_w, eff_h, last_xy)
                if group:
                    assigned[members] = True
                    if SHARE_SIDES and side == 1:
                        top = np.flatnonzero((c_type == t) & (c_step == step) &
                                              (c_side == 0) & (~assigned))
                        absorb(group, top)
                    fovs.extend(group)
                    group_all.extend(group)
                    last_xy = (group[-1]['x'], group[-1]['y'])
            if SHARE_BARCODE and t == 1 and group_all:
                normal = np.flatnonzero((c_type == 0) & (c_step == step) & (~assigned))
                absorb(group_all, normal)

    rows = [{'x': f['x'], 'y': f['y'],
             'comp_idx': json.dumps(sorted((int(i) for i in f['comp_idx']),
                                          key=lambda i: -c_time[i]) if LPT
                                    else [int(i) for i in f['comp_idx']])}
            for f in fovs if f['comp_idx']]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_folder, 'output_fov.csv'), index=True)

    return len(df)
