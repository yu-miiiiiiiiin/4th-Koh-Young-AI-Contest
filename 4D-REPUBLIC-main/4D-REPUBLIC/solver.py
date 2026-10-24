"""Final solver: every component that measurably earned its runtime.

Assembled from the ablation study in exp/results.csv:

  construction portfolio  integral-image weighted set cover (side-first two
                          pass, box growing) + dual-lattice Hochbaum-Maass
                          offsets; whichever costs less capture+reconstruction
  cleanup                 box-intersection merge -> redistribute -> merge ->
                          drop-redundant   (5394 score-points per second, by
                          far the best value in the whole pipeline)
  LNS                     localised destroy/repair with extreme-point candidate
                          repair (IMP_C) and record-to-record acceptance with
                          four destroy operators (IMP_E)
  tile sliding            big-component tiles slid inside their slack (IMP_G)
  sequencing              five block-structured orderings, best by exact CT
  polish                  prefix-cached hill climb: relocate + 2-opt

Dropped after measurement: adaptive budget (A), component re-assignment (B),
joint cost-aware cover (D), k-exchange (F), grid cropping (H2/H3), order ILS
(I), multi-restart cleanup (J), polish-top-2 (K), thirteen sequencing modes.
"""
import os
import time

import numpy as np

from . import solvercore as sc
from . import core
from . import lattice as lat


def _env(n, d, c=float):
    v = os.environ.get('BEST_' + n)
    return c(v) if v is not None else d


BUDGET = _env('BUDGET', 0.35)
CONSTRUCT = _env('CONSTRUCT', 0.30)      # fraction of budget for the portfolio
LNS_END = _env('LNS_END', 0.40)
RESTARTS = _env('RESTARTS', 1, int)     # independent LNS restarts from the construction
LAT_SHIFTS = _env('LAT_SHIFTS', 0, int)
LNS_ADAPT = _env('LNS_ADAPT', 0, int)
LNS_LOW = _env('LNS_LOW', 0.12)
LNS_HIGH = _env('LNS_HIGH', 0.60)
SCALE = _env('SCALE', 0.4)      # 1.0 = post-LNS budget proportional to FOV count
SCALE_REF = _env('SCALE_REF', 54.4)   # mean FOV count of the practice set -- see SCALE_POW
SCALE_POW = _env('SCALE_POW', 1.0)    # exponent on the size ratio (1 = proportional)
MULT_LO = _env('MULT_LO', 0.0)        # clamp on the post-LNS budget multiplier
MULT_HI = _env('MULT_HI', 1e9)


def solve(data_folder: str) -> int:
    t0 = time.perf_counter()
    P = core.Problem(data_folder)
    deadline = t0 + BUDGET
    rng = np.random.default_rng(20260919)

    # ---- 1. construction portfolio -----------------------------------------
    plans = [core._finalise(P, core._build(P, core._groups_global, deadline))]
    cut = t0 + CONSTRUCT * BUDGET
    for i in range(LAT_SHIFTS):
        if time.perf_counter() > cut:
            break
        ofx = i / max(1, LAT_SHIFTS)
        ofy = ((i * 2) % max(1, LAT_SHIFTS)) / max(1, LAT_SHIFTS)
        try:
            plans.append(core._finalise(P, lat._build_lattice(P, ofx, ofy)))
        except Exception:
            pass
    plan = min((p for p in plans if p), key=lambda p: core._cap_of(P, p))

    # ---- 2. localised LNS ---------------------------------------------------
    # LNS only ever accepts improvements, so one long trajectory can stall.
    # Splitting the same budget into independent restarts from the construction
    # explores more of the neighbourhood for free.
    # Cutting FOVs only pays while core 0 is the bottleneck: when inspection
    # dominates, CT = sum(t)/(k-3) however few FOVs there are, so the budget is
    # better spent on the order/centre/relabel loop downstream.
    frac = LNS_END
    if LNS_ADAPT:
        c0, insp = core._bottleneck(P, plan)
        if insp > c0 * 1.05:
            frac = LNS_LOW
        elif c0 > insp * 1.05:
            frac = LNS_HIGH
    end = t0 + frac * BUDGET
    pre = core._clone(plan) if sc.PICK_PLAN else None
    if RESTARTS <= 1:
        plan = core._lns(P, plan, end, rng)
    else:
        span = (end - time.perf_counter()) / RESTARTS
        best, bestc = plan, core._cap_of(P, plan)
        for r in range(RESTARTS):
            if time.perf_counter() >= end:
                break
            got = core._lns(P, core._clone(plan),
                            min(end, time.perf_counter() + span),
                            np.random.default_rng(1009 + 97 * r))
            c = core._cap_of(P, got)
            if c < bestc - 1e-9:
                best, bestc = got, c
        plan = best

    # The LNS optimises a capture proxy, not the score, so the plan it returns
    # is not always the better one -- decide on the real objective.
    if pre is not None:
        plan = sc.pick_plan(P, [plan, pre])

    # ---- 3. cleanup, sequencing, polish, output ----------------------------
    if SCALE:
        # a 130-FOV order has ~5x the search space of a 25-FOV one, so a flat
        # per-dataset budget over-serves the small ones
        # SCALE_REF is the practice set's mean FOV count, so a plain
        # proportional rule makes the budget -- and RT, which costs 1 point per
        # second -- grow without bound on a corpus of larger boards.  SCALE_POW
        # and the clamp bound that; the defaults reproduce the measured rule.
        r = (len(plan) / SCALE_REF) if SCALE_REF > 0 else 1.0
        mult = 1.0 - SCALE + SCALE * (r ** SCALE_POW)
        deadline = t0 + BUDGET * min(MULT_HI, max(MULT_LO, mult))
    return sc.finish(P, plan, deadline, rng, data_folder)
