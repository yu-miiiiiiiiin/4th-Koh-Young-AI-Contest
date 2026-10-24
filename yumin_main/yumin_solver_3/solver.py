"""ICCAS 2026 Koh Young AI Contest -- POMO-routed AOI scan planner.

    python 2_run.py --solver yumin_solver_3
    python 4_score.py yumin_solver_2 yumin_solver_3

What this solver optimises
--------------------------
`4_score.py` charges CT(data, k*) + 10*k* + runtime, and `Checker.compute_ct`
picks k* = argmin_k [CT(data,k) + 10k].  The two halves are the same expression,
so a board's score is exactly

    J = min_{k=4..max_core} [ CT(plan, k) + 10k ]

and that is what this solver measures every candidate plan by -- `sim.py` is a
verified line-by-line port of the grader's pipeline model, so the number the
search minimises is the number the grader pays.  Optimising CT alone is a trap:
CT falls when the plan leans on more inspection cores, and each extra core costs
10 points, roughly fourteen FOVs' worth of capture time.

Stages
------
1. COVER (`cover.py`).  Minimum piercing of the feasible-centre boxes: eight
   restart frames, each polished by merge / dissolve / absorb, best kept.  The
   only partition imposed is by step level, which rule 2.7 forces.  Type is not
   partitioned -- rule 2.4 lets a Barcode share its FOV with ordinary parts, so
   the piercing chooses those pairings rather than being handed them.  Fiducials
   are the exception rule 2.3 demands: one dedicated centred FOV each, emitted
   in registration order (rule 3.2).

2. ROUTE (`pomo.py`).  POMO: multi-start rollouts, shared baseline, x8 instance
   augmentation, argmax over everything seen.  Reward is J, so the router trades
   stage travel against the 5 s Z transitions, the inspection tail and k* at
   once instead of just minimising distance.  Rule 3.1 is the one ordering
   constraint it must respect, and it does so by only ever offering candidates
   from the highest ordering class still unvisited -- no sequence it can emit
   violates it.  The step order is left entirely to the reward.

3. EMIT.  Components inside each FOV are written longest-inspection-first.  The
   checker list-schedules them onto cores 3.. in exactly the order `comp_idx`
   gives, so this is LPT, and LPT is the reason the last FOV's inspection tail
   stops being the thing that sets CT.

Runtime is billed at 1 point/second, so the POMO budget is sized per board from
its FOV count and capped (`_budget`); the search keeps the best plan it has at
any moment, so cutting it short costs quality, never correctness.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from . import cover, pomo, sim

# Rule 1.3 captures Top and Side together at one position, so a group is not
# split by side: scanning side=1 separately revisits nearly the same spots.
# Rule 4.6 charges such a FOV max(capture, side_capture) -- one dearer shot, not
# two shots -- so folding them in wins on FOV count.
SPLIT_SIDE = False

# Seconds of POMO search per FOV, and the ceiling per board.  At alpha=1 a
# second of runtime costs the same as a second of cycle time, so the budget only
# pays for itself while the search is still finding real improvements.
SEC_PER_FOV = 0.005
MAX_BUDGET = 0.80
MIN_BUDGET = 0.025

# Ablation switches, both on in normal operation.  LPT orders each FOV's
# comp_idx longest-inspection-first; POLISH runs the 2-opt / Or-opt pass.
LPT = True
POLISH = True


def _budget(n_fov):
    return min(MAX_BUDGET, max(MIN_BUDGET, SEC_PER_FOV * n_fov))


def build(data_folder: str):
    """Read the board and cover it; returns a `pomo.Router` ready to route.

    Kept separate from `solve` so the cover can be built and inspected without
    running the search -- useful when measuring the router in isolation.
    """
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

    capture = float(param['capture_time'].iloc[0])
    side_capture = (float(param['side_capture_time'].iloc[0])
                    if 'side_capture_time' in param.columns else capture / 2.0)
    travel = sim.Travel(float(param['v_x'].iloc[0]), float(param['v_y'].iloc[0]),
                        float(param['a_x'].iloc[0]), float(param['a_y'].iloc[0]))
    inst = sim.Instance(travel, float(param['recon_time'].iloc[0]),
                        int(param['max_core'].iloc[0]))

    # per-component window: side=1 is inspected through the half FOV (rule 2.6)
    ew = np.where(c_side == 1, fov_w / 2.0, fov_w)
    eh = np.where(c_side == 1, fov_h / 2.0, fov_h)
    # ---- 1. fiducials: fixed prefix, one dedicated centred FOV each --------
    prefix = cover.fiducial_fovs(np.flatnonzero(c_type == 2), geom)
    for f in prefix:
        i = f['comp'][0]
        f['block'], f['step'] = 2, int(c_step[i])
        f['t_comp'] = [c_time[i]]
        f['load'] = float(c_time[i])
        f['t_img'] = max(capture, side_capture) if c_side[i] else capture

    # ---- 2. cover, one problem per step level ------------------------------
    # Step is the only forced partition (rule 2.7).  Type is *not* partitioned:
    # rule 2.4 lets one FOV carry a Barcode and ordinary parts together and
    # still be type=1, so the piercing gets to choose those pairings itself.
    # Side is not partitioned either -- rule 1.3 captures Top and Side at one
    # position, so each part just keeps its own window (half FOV for side=1).
    levels = {}
    for i in range(n):
        if c_type[i] != 2:
            levels.setdefault((c_step[i], c_side[i] if SPLIT_SIDE else 0), []).append(i)

    fovs = []
    for (step, side), members in sorted(levels.items()):
        grid = (fov_w / 2.0, fov_h / 2.0) if (SPLIT_SIDE and side == 1) else (fov_w, fov_h)
        for f in cover.cover_level(np.array(members), geom, ew, eh, grid):
            if not f['comp']:
                continue
            f['x'] = (f['lo_x'] + f['hi_x']) / 2.0
            f['y'] = (f['lo_y'] + f['hi_y']) / 2.0
            f['block'] = max(c_type[i] for i in f['comp'])   # rule 3.1 class
            f['step'] = step
            fovs.append(f)

    # LPT: the checker inspects a FOV's components in the order comp_idx lists
    # them, list-scheduled onto cores 3.., so longest-first is a free makespan
    # cut on exactly the tail that sets CT.
    for f in fovs:
        f['comp'] = sorted((int(i) for i in f['comp']),
                           key=(lambda i: -c_time[i]) if LPT else (lambda i: i))
        f['t_comp'] = [c_time[i] for i in f['comp']]
        f['load'] = float(sum(f['t_comp']))
        f['t_img'] = max(capture, side_capture) if any(c_side[i] for i in f['comp']) else capture

    return pomo.Router(inst, fovs, prefix), fovs, prefix


def solve(data_folder: str) -> int:
    t0 = time.perf_counter()
    clock = lambda: time.perf_counter() - t0

    router, fovs, prefix = build(data_folder)

    # ---- 3. POMO routing ---------------------------------------------------
    rng = np.random.default_rng(20260919)
    budget = _budget(len(fovs))
    seq, _ = router.search(budget, clock, rng)
    if POLISH:
        seq = pomo.polish(router, seq, budget * 1.5, clock)

    # ---- 4. emit -----------------------------------------------------------
    _, _, x, y = router.evaluate(seq)
    rows = [{'x': f['x'], 'y': f['y'], 'comp_idx': json.dumps(f['comp'])}
            for f in prefix]
    rows += [{'x': float(x[i]), 'y': float(y[i]),
              'comp_idx': json.dumps(fovs[f]['comp'])}
             for i, f in enumerate(seq)]

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_folder, 'output_fov.csv'), index=True)
    return len(df)
