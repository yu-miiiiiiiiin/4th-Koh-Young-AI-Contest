"""Cycle-time model of the grader, and the score it actually charges.

`utils/checker.py` never reports CT on its own.  It reports

    CT(plan, k*)      k* = argmin_{k=4..max_core} [ CT(plan, k) + 10k ]

and `4_score.py` then adds `10 * k*` back, so the number a board contributes is

    J(plan) = min_{k=4..max_core} [ CT(plan, k) + 10k ]      (+ solver runtime)

That is the only objective in this package.  CT alone is a trap: buying CT with
inspection cores costs 10 points each, about twenty captures' worth.

`ct_for_k` is the grader's `Checker.get_real_cost` with the per-component loop
replaced by a heap.  The rewrite is exact, not an approximation:

  * The grader syncs every core up to core 0 after each capture
    (`t_last[:] = max(t_last, t_last[0])`).  A job of FOV i can only start at
    max(core_free, recon_end_i), and recon_end_i >= capture_end_i >= every
    earlier floor, so the floor is already implied by the recon term and never
    has to be materialised.
  * max(a, floor) is monotone in a, so argmin_c max(a_c, done) = argmin_c a_c:
    popping the heap minimum picks exactly the core the grader picks.

One `heapreplace` per component, in C, instead of a Python scan over cores.
"""
import heapq

import numpy as np

BETA = 10.0           # 4_score.py: score = sum(CT) + sum(RT) + 10 * sum(core)
Z_DELAY = 5.0         # rule 4.7, Z-axis settling charged on a step change


# --------------------------------------------------------------------------
# stage travel -- rule 5.1/5.2: trapezoidal profile per axis, slower axis wins
# --------------------------------------------------------------------------

def axis_time(dist, vel, acc):
    """Checker._compute_axis_time: triangular below v^2/a, trapezoidal above."""
    th = vel * vel / acc
    dist = np.asarray(dist, dtype=float)
    over = 2.0 * vel / acc + (dist - th) / vel
    under = 2.0 * np.sqrt(np.maximum(dist, 0.0) / acc)
    return np.where(dist >= th, over, under)


class Travel:
    """max(T_x, T_y) between stage positions."""

    def __init__(self, v_x, v_y, a_x, a_y):
        self.v_x, self.v_y, self.a_x, self.a_y = v_x, v_y, a_x, a_y

    def __call__(self, px, py, qx, qy):
        return np.maximum(axis_time(np.abs(np.asarray(px, float) - qx), self.v_x, self.a_x),
                          axis_time(np.abs(np.asarray(py, float) - qy), self.v_y, self.a_y))

    def pairwise(self, x, y):
        dx = np.abs(x[:, None] - x[None, :])
        dy = np.abs(y[:, None] - y[None, :])
        return np.maximum(axis_time(dx, self.v_x, self.a_x),
                          axis_time(dy, self.v_y, self.a_y))


# --------------------------------------------------------------------------
# exact cycle time
# --------------------------------------------------------------------------

def ct_for_k(t_move, t_img, t_recon, t_comp, steps, k):
    """Cycle time of one FOV sequence on `k` cores (core 0 stage, 1-2 recon,
    3..k-1 inspection).

    t_move[i]  stage time into FOV i (t_move[0] unused)
    t_comp[i]  inspection times of FOV i in the order `comp_idx` lists them --
               the grader feeds the cores in exactly that order
    """
    n = len(t_img)
    if n == 0:
        return 0.0

    m = k - 3                            # inspection cores
    core0 = t_img[0]
    r0 = r1 = core0
    ins = [core0] * m
    zd = Z_DELAY

    for i in range(n):
        if i:
            core0 += t_move[i] + t_img[i]
            if steps[i] != steps[i - 1]:
                core0 += zd
            if r0 < core0:
                r0 = core0
            if r1 < core0:
                r1 = core0

        if r0 <= r1:
            r0 += t_recon
            done = r0
        else:
            r1 += t_recon
            done = r1

        jobs = t_comp[i]
        if not jobs:
            continue
        if m == 1:
            a = ins[0]
            if a < done:
                a = done
            for t in jobs:
                a += t
            ins[0] = a
        else:
            for t in jobs:
                a = ins[0]
                if a < done:
                    a = done
                heapq.heapreplace(ins, a + t)

    return max(core0, r0, r1, max(ins))


def score(t_move, t_img, t_recon, t_comp, steps, max_core):
    """(J, k*) -- the grader's own minimisation over the core count.

    CT falls monotonically in k while 10k grows, so once CT(max_core) + 10k
    passes the incumbent no larger k can win and the sweep stops.
    """
    max_core = max(int(max_core), 4)
    floor = ct_for_k(t_move, t_img, t_recon, t_comp, steps, max_core)

    best_j, best_k = None, 4
    for k in range(4, max_core + 1):
        if best_j is not None and floor + BETA * k >= best_j:
            break
        ct = floor if k == max_core else ct_for_k(t_move, t_img, t_recon, t_comp, steps, k)
        j = ct + BETA * k
        if best_j is None or j < best_j:
            best_j, best_k = j, k
    return best_j, best_k


# --------------------------------------------------------------------------
# O(n) surrogate used inside the search loops
# --------------------------------------------------------------------------

def relax_ct(chain, load, t_recon, m):
    """Continuous relaxation of the inspection makespan on `m` cores.

    `chain[i]` is core 0's capture-end clock at FOV i.  Everything from FOV i
    onward is released no earlier than chain[i] + t_recon, so

        CT >= max_i ( chain[i] + t_recon + (sum_{j>=i} load[j]) / m )

    and the bound is tight whenever the cores stay saturated -- which is the
    regime that decides the order.  O(n) with a suffix sum, against O(#parts)
    for the exact model, so the search runs it and pays for `ct_for_k` only
    when ranking survivors.
    """
    tail = np.cumsum(load[::-1])[::-1]
    return float(np.max(chain + t_recon + tail / m))


class Instance:
    """Everything the objective needs once the cover is fixed."""

    def __init__(self, travel, t_recon, max_core):
        self.travel = travel
        self.t_recon = float(t_recon)
        self.max_core = max(int(max_core), 4)

    def moves(self, x, y):
        n = len(x)
        t_move = np.empty(n)
        t_move[0] = 0.0
        if n > 1:
            t_move[1:] = self.travel(x[:-1], y[:-1], x[1:], y[1:])
        return t_move

    def evaluate(self, x, y, t_img, t_comp, steps):
        """J and k* of a FOV sequence given as parallel arrays."""
        if len(x) == 0:
            return 0.0, 4
        return score(self.moves(x, y), t_img, self.t_recon, t_comp, steps, self.max_core)
