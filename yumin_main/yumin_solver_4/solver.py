"""ICCAS 2026 Koh Young AI Contest -- AOI scan planner.

    python 2_run.py --solver yumin_solver_4
    python 4_score.py yumin_solver_3 yumin_solver_4

What is being minimised
-----------------------
`Checker.compute_ct` reports CT(plan, k*) with k* = argmin_k [CT(plan,k) + 10k],
and `4_score.py` adds 10k* back on.  The two halves are the same expression, so
a board's score is exactly

    J(plan) = min_{k=4..max_core} [ CT(plan, k) + 10k ]   (+ solver runtime)

`sim.py` is that expression, and every stage below is measured by it.  CT alone
is a trap: CT falls when a plan leans on more inspection cores, and each core
costs 10 points -- about twenty captures' worth.

Where the score actually sits, measured on the 96 practice boards
-----------------------------------------------------------------
Writing W for the inspection work a board cannot avoid (a part's time, once per
FOV that must list it -- rule 2.5 forces an oversized part to be re-inspected
per tile) and G for core 0's chain of captures, Z transitions and stage moves,

    J  >=  min_k [ max(G, W/(k-3)) + 10k ]

The right-hand side over all 96 boards is 12471, of which 11700 survives even
at G = 0.  Both halves were pushed on before this solver was written:

  * Cover.  A ruin-and-recreate pass over the finished cover, given two seconds
    a board, removes 57 FOVs out of 5274 -- 1.1%, worth about 50 points of J for
    192 points of runtime.  The cover is at its floor, so `shrink` is run only
    inside the short window where it still pays for itself.
  * Route.  Re-routing the same covers with 40x the search budget moves the
    total by under a point.  The policy converges in well under a tenth of a
    second; the rest was runtime bought for nothing.  Hence `route.STALL`.

What is left is the seam between them, the 362 points by which the plans exceed
max(G, W/(k-3)): work stranded in shots taken late in the scan, and the drain
after the final capture.  `refine.py` is the stage that attacks it, and it is
the one thing here solver 3 has no equivalent of.

Stages
------
1. COVER (`cover.py`).  Minimum piercing of the feasible-centre boxes, eight
   restart frames, each polished by merge / dissolve / absorb, best kept, then
   ruin-and-recreate while the budget allows.  Step level is the only partition
   imposed (rule 2.7); type is not, because rule 2.4 lets a Barcode share its
   shot with ordinary parts and the piercing should choose those pairings
   itself.  Fiducials are the exception rule 2.3 demands: one dedicated centred
   shot each, emitted in registration order (rule 3.2).

2. ROUTE (`route.py`).  POMO -- multi-start rollouts, shared baseline, x8
   instance augmentation, argmax over everything seen -- with J as the reward,
   so the router trades stage travel against the 5 s Z transitions, the
   inspection tail and k* at once.  Rule 3.1 holds by construction: candidates
   are only ever offered from the highest ordering class still unvisited.

3. REFINE (`refine.py`).  Parts move to the earliest shot that can hold them,
   emptied shots are deleted, and the closing shots are lightened.  Then the
   order is re-optimised over the new loads and the better plan kept.

4. EMIT.  Parts are written longest-inspection-first inside each shot, which is
   LPT on the cores the grader feeds in that exact order.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from . import cover, refine, route, sim

# Rule 1.3 captures Top and Side at one position, so a shot is not split by
# side: covering side=1 separately revisits nearly the same spots.  Measured,
# the side=1 parts alone need within a few percent of the FOVs the whole board
# needs, so they set the count and the top-only parts ride along.
SPLIT_SIDE = False

# Runtime is charged at 1 point/second (alpha=1), so every budget here is a
# ceiling that the stall detectors normally beat.  Sized per board off its FOV
# count, which is what the cost of a search iteration scales with.
SEC_PER_FOV = 0.0054
MAX_BUDGET = 0.69
MIN_BUDGET = 0.036

# Allowance per stage as a multiple of the board budget.  Each is spent from
# the moment its stage starts, not as a checkpoint on the wall clock: a shared
# deadline silently starves whatever runs last, which on a small board was
# every second of shaping.
# `shrink` is off by default: ruin-and-recreate over the finished cover removes
# 1.1% of the FOVs for 2 s a board, which is about 50 points of CT bought with
# 192 points of runtime.  The switch stays so the measurement can be repeated.
SHRINK_FRAC = 0.35

POLISH_ALLOW = 0.6
SHAPE_ALLOW = 1.2
REFINE_ALLOW = 0.8
ILS_ALLOW = 4.0
ILS_GATE = 1.0              # seconds of profile excess below which ILS is skipped
ILS_PER_POINT = 0.8
LOOP_ALLOW = 6.0            # wall-clock ceiling for the refine/route alternation
SHAPE_GATE = 0.25           # seconds of profile excess below which shaping is skipped
N_START_DEFAULT = 8         # POMO starting nodes per iteration (S4.1)

# (w_pos, w_load) mixtures tried by refine.assign; see its docstring.
MIXES = ((1.0, 0.0), (1.0, 0.5))

# The flexible searches are more expensive than the original fixed-centre
# passes.  Measurements on both practice outputs and 32 fresh generator seeds
# show that shot dissolution pays only above a modest profile gap, while exact
# tail moves pay at larger gaps.  Each gets a tiny absolute cap so runtime stays
# score-aware on the CPU-only evaluator.
FLEX_DISSOLVE_GATE = 1.0
FLEX_TAIL_GATE = 3.0
FLEX_PASS_CAP = 0.03


def _budget(n_fov):
    return min(MAX_BUDGET, max(MIN_BUDGET, SEC_PER_FOV * n_fov))


# --------------------------------------------------------------------------
# reading a board
# --------------------------------------------------------------------------

class Board:
    """One dataset, parsed once."""

    def __init__(self, data_folder):
        comp = pd.read_csv(os.path.join(data_folder, 'input_component.csv'))
        size = pd.read_csv(os.path.join(data_folder, 'input_size.csv'))
        param = pd.read_csv(os.path.join(data_folder, 'input_parameter.csv'))

        n = len(comp)
        self.n = n
        self.tl_x = comp['tl_x'].values.astype(float)
        self.tl_y = comp['tl_y'].values.astype(float)
        self.br_x = comp['br_x'].values.astype(float)
        self.br_y = comp['br_y'].values.astype(float)
        self.type = comp['type'].astype(int).values
        self.time = comp['time'].values.astype(float)
        self.side = comp['side'].astype(int).values if 'side' in comp.columns else np.zeros(n, int)
        self.step = comp['step'].astype(int).values if 'step' in comp.columns else np.zeros(n, int)

        self.fov_w = float(size['fov_size'].iloc[0])
        self.fov_h = float(size['fov_size'].iloc[1])
        self.capture = float(param['capture_time'].iloc[0])
        self.side_capture = (float(param['side_capture_time'].iloc[0])
                             if 'side_capture_time' in param.columns else self.capture / 2.0)
        self.t_img_side = max(self.capture, self.side_capture)

        travel = sim.Travel(float(param['v_x'].iloc[0]), float(param['v_y'].iloc[0]),
                            float(param['a_x'].iloc[0]), float(param['a_y'].iloc[0]))
        self.inst = sim.Instance(travel, float(param['recon_time'].iloc[0]),
                                 int(param['max_core'].iloc[0]))

        # per-part window: side=1 is inspected through the half FOV (rule 2.6)
        self.ew = np.where(self.side == 1, self.fov_w / 2.0, self.fov_w)
        self.eh = np.where(self.side == 1, self.fov_h / 2.0, self.fov_h)
        self.geom = (self.tl_x, self.tl_y, self.br_x, self.br_y)

        # feasible-centre box of each part; empty in an axis means oversized
        self.lo_x = self.br_x - self.ew / 2.0
        self.hi_x = self.tl_x + self.ew / 2.0
        self.lo_y = self.br_y - self.eh / 2.0
        self.hi_y = self.tl_y + self.eh / 2.0
        self.big = (self.lo_x > self.hi_x + 1e-9) | (self.lo_y > self.hi_y + 1e-9)

    def t_img_of(self, members):
        return self.t_img_side if any(self.side[i] for i in members) else self.capture

    def decorate(self, f):
        f['comp'] = sorted((int(i) for i in f['comp']), key=lambda i: -self.time[i])
        f['t_comp'] = [self.time[i] for i in f['comp']]
        f['load'] = float(sum(f['t_comp']))
        f['t_img'] = self.t_img_of(f['comp'])
        f['block'] = max((int(self.type[i]) for i in f['comp']), default=0)
        return f


# --------------------------------------------------------------------------
# stage 1: cover
# --------------------------------------------------------------------------

def build_cover(b, budget, clock, rng):
    """Fiducial prefix plus a minimum piercing of everything else."""
    prefix = cover.fiducial_fovs(np.flatnonzero(b.type == 2), b.geom)
    for f in prefix:
        i = f['comp'][0]
        f['step'] = int(b.step[i])
        b.decorate(f)
        f['block'] = 2

    levels = {}
    for i in range(b.n):
        if b.type[i] != 2:
            levels.setdefault((b.step[i], b.side[i] if SPLIT_SIDE else 0), []).append(i)

    fovs = []
    for (step, side), members in sorted(levels.items()):
        grid = ((b.fov_w / 2.0, b.fov_h / 2.0) if (SPLIT_SIDE and side == 1)
                else (b.fov_w, b.fov_h))
        for f in cover.cover_level(np.array(members), b.geom, b.ew, b.eh, grid,
                                   weights=b.time):
            if not f['comp']:
                continue
            f['step'] = int(step)
            if any(b.big[i] for i in f['comp']):
                f['pinned'] = True          # rule 2.5 fixes an oversized tile
            fovs.append(f)

    if budget > 0 and clock() < budget:
        pinned = [i for i, f in enumerate(fovs) if f.get('pinned')]
        cover.shrink(fovs, b.lo_x, b.hi_x, b.lo_y, b.hi_y, budget, clock, rng,
                     pinned=pinned)

    for f in fovs:
        f['x'] = (f['lo_x'] + f['hi_x']) / 2.0
        f['y'] = (f['lo_y'] + f['hi_y']) / 2.0
        b.decorate(f)
    return prefix, fovs


# --------------------------------------------------------------------------
# stage 3: refine
# --------------------------------------------------------------------------

def refine_plan(b, prefix, fovs, seq, x, y, budget, clock,
                relax_assign=False, relax_dissolve=False):
    """Re-assign parts over the routed plan.

    Returns (winner, its J, the untouched input plan).  The caller compares the
    winner against the third value to tell whether anything actually moved.
    """
    n_pre = len(prefix)
    xs = np.concatenate((np.array([f['x'] for f in prefix]), x)) if n_pre else x
    ys = np.concatenate((np.array([f['y'] for f in prefix]), y)) if n_pre else y
    steps = [f['step'] for f in prefix] + [fovs[i]['step'] for i in seq]
    blocks = [f['block'] for f in prefix] + [fovs[i]['block'] for i in seq]
    members = [list(f['comp']) for f in prefix] + [list(fovs[i]['comp']) for i in seq]

    base = refine.Plan(b.inst, xs, ys, steps, blocks, members,
                       b.time, b.side, b.capture, b.side_capture)
    best, best_j = base, base.score()[0]
    if clock() > budget:
        return best, best_j, base

    # fiducial shots are exclusive and centred (rule 2.3) and oversized parts
    # are pinned to their tiles (rule 2.5): neither may be re-assigned
    movable = (~b.big) & (b.type != 2)
    cols, ok, feas = refine.feasible_matrix(base.x, base.y, np.array(base.step),
                                            b.tl_x, b.tl_y, b.br_x, b.br_y,
                                            b.ew, b.eh, b.step, movable)
    pinned_comp = b.big | (b.type == 2)
    # Release time and core occupancy pull in opposite directions and which
    # one a board wants is not knowable from the board; try the mixtures and
    # let the exact score choose.  MIXES[0] is pure release time, which is
    # `pull_forward`.
    for w_pos, w_load in MIXES:
        if clock() > budget and best is not base:
            break
        trial = base.copy()
        if not refine.assign(trial, feas, cols, b.type, n_pre, w_pos, w_load):
            continue
        trial.compact(n_pre)
        refine.resort(trial, n_pre)
        if relax_assign:
            refine.relax_positions(trial, b.lo_x, b.hi_x, b.lo_y, b.hi_y,
                                   pinned_comp, n_pre)
        j = trial.score()[0]
        if j < best_j - 1e-9:
            best, best_j = trial, j

    if clock() < budget:
        trial = best.copy()
        cols2, ok2, _ = refine.feasible_matrix(trial.x, trial.y, np.array(trial.step),
                                               b.tl_x, b.tl_y, b.br_x, b.br_y,
                                               b.ew, b.eh, b.step, movable)
        j, _ = refine.dissolve_shots(trial, ok2, cols2, b.type, n_pre, budget, clock)
        if relax_dissolve:
            refine.relax_positions(trial, b.lo_x, b.hi_x, b.lo_y, b.hi_y,
                                   pinned_comp, n_pre)
            j = trial.score()[0]
        if j < best_j - 1e-9:
            best, best_j = trial, j

    if clock() < budget:
        trial = best.copy()
        cols3, _, feas3 = refine.feasible_matrix(trial.x, trial.y, np.array(trial.step),
                                                 b.tl_x, b.tl_y, b.br_x, b.br_y,
                                                 b.ew, b.eh, b.step, movable)
        j = refine.lighten_tail(trial, feas3, cols3, n_pre, budget, clock)
        if j < best_j - 1e-9:
            best, best_j = trial, j

    return best, best_j, base


def plan_to_fovs(b, best, n_pre):
    """Turn a refined plan back into cover-style FOVs so it can be re-routed.

    A shot's feasible box is the intersection of its members' boxes, which is
    exactly how `cover.py` defines it, so the router may slide the shot again
    within whatever freedom the new membership leaves.  Oversized-part tiles
    (rule 2.5) and fiducials (rule 2.3) keep their box pinned to a point.
    """
    prefix, fovs = [], []
    for f in range(len(best.members)):
        ms = best.members[f]
        if not ms:
            continue
        if any(b.big[i] or b.type[i] == 2 for i in ms):
            lo_x = hi_x = float(best.x[f])
            lo_y = hi_y = float(best.y[f])
        else:
            lo_x = max(b.lo_x[i] for i in ms)
            hi_x = min(b.hi_x[i] for i in ms)
            lo_y = max(b.lo_y[i] for i in ms)
            hi_y = min(b.hi_y[i] for i in ms)
            if lo_x > hi_x or lo_y > hi_y:       # should not happen; stay safe
                lo_x = hi_x = float(best.x[f])
                lo_y = hi_y = float(best.y[f])
        g = {'lo_x': lo_x, 'hi_x': hi_x, 'lo_y': lo_y, 'hi_y': hi_y,
             'x': min(max(float(best.x[f]), lo_x), hi_x),
             'y': min(max(float(best.y[f]), lo_y), hi_y),
             'comp': list(ms), 'step': best.step[f]}
        b.decorate(g)
        (prefix if f < n_pre else fovs).append(g)
    return prefix, fovs


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

# Candidate strategies, in the order they were built and measured.  `bench.py`
# runs all of them over the 96 practice boards; STRATEGY names the one adopted.
# Each is a strict superset of the one before it, so the table reads as an
# ablation of the final solver as much as a comparison of alternatives.
LADDER = ('greedy', 'hier', 'pomo', 'shape', 'refine', 'unified', 'ils')
STRATEGY = 'unified'


def plan(data_folder, budget=None, seed=20260920, shrink=False, do_refine=True,
         stall=route.STALL, bscale=1.0, do_shape=True, do_ils=False, n_start=None, n_frame=None,
         ils_gate=None, ils_per=None, ils_allow=None, rounds=1, strategy=None,
         n_start_default=N_START_DEFAULT, do_flexible=True,
         do_flexible_assign=False, relax_assign=False, relax_dissolve=False,
         flex_pass_cap=None, flex_dissolve_gate=None, flex_tail_gate=None,
         _board=None):
    """Produce (rows, J) for one board without writing anything.

    Normal operation passes nothing but the folder.  Every other argument is
    there so a stage can be switched off, widened or starved and measured on its
    own -- `strategy` walks the candidate ladder, `bscale` scales every time
    budget at once, and the rest are the knobs each sweep in the README moved.
    """
    t0 = time.perf_counter()
    clock = lambda: time.perf_counter() - t0

    # `_board` lets a wrapper reuse a Board already parsed for feature
    # extraction.  Normal solver_4 calls leave it unset and behave identically.
    b = Board(data_folder) if _board is None else _board
    rng = np.random.default_rng(seed)

    prefix, fovs = build_cover(b, 0.0, clock, rng)
    total = (_budget(len(fovs)) * bscale) if budget is None else budget

    if shrink and len(fovs) > 2:
        prefix, fovs = build_cover(b, total * SHRINK_FRAC, clock, rng)

    strategy = STRATEGY if strategy is None else strategy
    if strategy not in LADDER:
        raise ValueError(f'unknown strategy {strategy!r}; pick one of {LADDER}')
    rank = LADDER.index(strategy)
    do_shape = do_shape and rank >= LADDER.index('shape')
    do_refine = do_refine and rank >= LADDER.index('refine')
    rounds = rounds if rank >= LADDER.index('unified') else 0
    do_ils = do_ils or rank >= LADDER.index('ils')

    router = route.Router(b.inst, fovs, prefix)
    if rank >= LADDER.index('pomo'):
        seq, _ = router.search(total, clock, rng, stall=stall,
                               n_start=n_start or n_start_default,
                               n_frame=n_frame)
    else:
        seq = route.nearest(router)
    if rank >= LADDER.index('hier'):
        seq = route.polish(router, seq, clock() + total * POLISH_ALLOW, clock)

    j0, k0 = router.evaluate(seq)[:2]
    j0 = -j0
    ks = [k for k in (k0 - 1, k0, k0 + 1) if 4 <= k <= b.inst.max_core]
    shaper = route.Shaper(router, ks)
    # How far this plan sits above the best any ordering of this cover could
    # reach.  Shaping moves work along the scan, so on a board already sitting
    # on its floor there is nothing for it to move and the time is spent for
    # certain while the score is not.  Runtime is charged at 1 point/second, so
    # a stage that cannot pay has to be skipped, not merely time-boxed.
    room = j0 - shaper.floor(seq)

    if do_shape and room > SHAPE_GATE:
        # polish fixes the drive; shape fixes where the work sits along it
        alt = route.shape(router, seq, clock() + total * SHAPE_ALLOW, clock, rng, ks=ks)
        alt = route.polish(router, alt, clock() + total * POLISH_ALLOW, clock)
        alt_j = -router.evaluate(alt)[0]
        if alt_j < j0 - 1e-9:
            seq, j0 = alt, alt_j

    if do_ils and len(seq) >= 8:
        gate = ILS_GATE if ils_gate is None else ils_gate
        per = ILS_PER_POINT if ils_per is None else ils_per
        allow = ILS_ALLOW if ils_allow is None else ils_allow
        if room > gate:
            give = min(allow, per * room) * total
            alt = route.ils(router, seq, clock() + give, clock, rng, ks=ks)
            if -router.evaluate(alt)[0] < j0 - 1e-9:
                seq = alt
    _, _, x, y = router.evaluate(seq)

    if do_refine:
        best, best_j, base_plan = refine_plan(b, prefix, fovs, seq, x, y,
                                              clock() + total * REFINE_ALLOW, clock,
                                              relax_assign, relax_dissolve)
    else:
        n_pre = len(prefix)
        xs = np.concatenate((np.array([f['x'] for f in prefix]), x)) if n_pre else x
        ys = np.concatenate((np.array([f['y'] for f in prefix]), y)) if n_pre else y
        best = refine.Plan(b.inst, xs, ys,
                           [f['step'] for f in prefix] + [fovs[i]['step'] for i in seq],
                           [f['block'] for f in prefix] + [fovs[i]['block'] for i in seq],
                           [list(f['comp']) for f in prefix] + [list(fovs[i]['comp']) for i in seq],
                           b.time, b.side, b.capture, b.side_capture)
        best_j = best.score()[0]
        base_plan = best

    # ---- 4. re-route over the new loads, then refine again -----------------
    # Cover then route is a hierarchy, and a hierarchy is only as good as the
    # guess the first stage made about the second.  Once `refine` has moved the
    # work, the order that was optimal for the old loads usually is not, so the
    # two stages are run once more against each other -- the unified-method idea
    # from Park et al. (IJCAS 2006), applied as an alternation rather than as a
    # single joint encoding.
    for _ in range(rounds if do_refine else 0):
        # Nothing moved, so the cover the router already optimised is the cover
        # it would be handed again: re-routing it is guaranteed waste.
        if best is base_plan or clock() > total * LOOP_ALLOW:
            break
        pre2, fov2 = plan_to_fovs(b, best, len(prefix))
        if not fov2:
            break
        r2 = route.Router(b.inst, fov2, pre2)
        s2 = list(range(len(fov2)))
        s2 = route.polish(r2, s2, clock() + total * POLISH_ALLOW, clock)
        j2, k2 = r2.evaluate(s2)[:2]
        ks2 = [k for k in (k2 - 1, k2, k2 + 1) if 4 <= k <= b.inst.max_core]
        if -j2 - route.Shaper(r2, ks2).floor(s2) > SHAPE_GATE:
            s3 = route.shape(r2, s2, clock() + total * SHAPE_ALLOW, clock, rng, ks=ks2)
            s3 = route.polish(r2, s3, clock() + total * POLISH_ALLOW, clock)
            s3_j = -r2.evaluate(s3)[0]
            if s3_j < -j2 - 1e-9:
                s2 = s3
        _, _, x2, y2 = r2.evaluate(s2)
        cand, cand_j, base2 = refine_plan(b, pre2, fov2, s2, x2, y2,
                                          clock() + total * REFINE_ALLOW, clock,
                                          relax_assign, relax_dissolve)
        if cand_j < best_j - 1e-9:
            best, best_j, prefix, base_plan = cand, cand_j, pre2, base2
        else:
            break

    # One final movable-centre neighbourhood after cover/route/refine have
    # settled.  Running it inside every alternation bought the same deletions
    # repeatedly and spent more runtime than they were worth; once here keeps
    # the useful search and pays its cost only once.
    if do_refine and do_flexible:
        n_pre = len(prefix)
        movable = (~b.big) & (b.type != 2)
        pinned_comp = b.big | (b.type == 2)
        pass_cap = FLEX_PASS_CAP if flex_pass_cap is None else flex_pass_cap
        dissolve_gate = (FLEX_DISSOLVE_GATE if flex_dissolve_gate is None
                          else flex_dissolve_gate)
        tail_gate = FLEX_TAIL_GATE if flex_tail_gate is None else flex_tail_gate
        if do_flexible_assign:
            for w_pos, w_load in MIXES:
                trial = best.copy()
                if not refine.assign_flexible(
                        trial, b.lo_x, b.hi_x, b.lo_y, b.hi_y,
                        b.step, b.type, movable, pinned_comp, n_pre,
                        w_pos=w_pos, w_load=w_load):
                    continue
                trial.compact(n_pre)
                refine.resort(trial, n_pre)
                refine.relax_positions(
                    trial, b.lo_x, b.hi_x, b.lo_y, b.hi_y,
                    pinned_comp, n_pre)
                j = trial.score()[0]
                if j < best_j - 1e-9:
                    best, best_j = trial, j
        if best_j - refine.profile_floor(best) > dissolve_gate:
            trial = best.copy()
            j, _ = refine.dissolve_flexible(
                trial, b.lo_x, b.hi_x, b.lo_y, b.hi_y, b.step, b.type,
                movable, pinned_comp, n_pre, clock() + pass_cap, clock,
                initial_j=best_j)
            if j < best_j - 1e-9:
                best, best_j = trial, j
        if best_j - refine.profile_floor(best) > tail_gate:
            trial = best.copy()
            j = refine.lighten_critical(
                trial, b.lo_x, b.hi_x, b.lo_y, b.hi_y, b.step, b.type,
                movable, pinned_comp, n_pre, clock() + pass_cap, clock,
                max_moves=3, initial_j=best_j)
            if j < best_j - 1e-9:
                best, best_j = trial, j

    rows = [{'x': float(best.x[i]), 'y': float(best.y[i]),
             'comp_idx': json.dumps(sorted((int(c) for c in ms),
                                           key=lambda c: -b.time[c]))}
            for i, ms in enumerate(best.members)]
    return rows, best_j


def solve(data_folder: str) -> int:
    rows, _ = plan(data_folder)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_folder, 'output_fov.csv'), index=True)
    return len(df)
