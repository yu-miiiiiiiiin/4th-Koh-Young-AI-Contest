"""Exact mirror of the grader's cycle-time model, plus the score it really pays.

Why this file exists
--------------------
`utils/checker.py` does not report CT.  It reports

    CT(data, k*)     where   k* = argmin_{k=4..max_core} [ CT(data, k) + 10k ]

and `4_score.py` then charges `CT(data, k*) + 10 * k*`.  The two halves cancel:
the score a dataset contributes is exactly the value the checker minimised,

    J(plan) = min_{k=4..max_core} [ CT(plan, k) + BETA * k ]        (+ runtime)

so J -- not CT -- is the quantity a solver should be steering.  A plan that
lowers CT by adding FOVs the extra cores have to chew through can raise k* and
lose points overall; a plan that finishes with a light FOV lets the checker
settle on k*=4 and bank 120 points against max_core=16.

`ct_for_k` below is a line-by-line port of `Checker.get_real_cost`, including
the two behaviours that drive most of the tail:

  * after every capture, `t_last[:] = max(t_last, t_last[0])` drags every core
    up to core 0's clock, so inspection work that fits inside the capture
    window is free and only the overhang carries forward; and
  * components inside one FOV are list-scheduled onto cores 3.. in the order
    they appear in `comp_idx`, which the solver controls (see LPT in cover.py).

Reference: rules 4.1-4.7 and 5.1-5.2 of ICCAS_2026_CONTEST_RULES.html.
"""
import numpy as np

BETA = 10.0           # 4_score.py: score = sum(CT) + sum(RT)*1 + sum(core)*10
Z_DELAY = 5.0         # rule 4.7, Z-axis settling on a step change


# --------------------------------------------------------------------------
# stage travel -- rule 5.1/5.2, trapezoidal profile per axis, slower axis wins
# --------------------------------------------------------------------------

def axis_time(dist, vel, acc):
    """Time to traverse `dist` on one axis (Checker._compute_axis_time)."""
    th = vel * vel / acc
    dist = np.asarray(dist, dtype=float)
    over = 2.0 * vel / acc + (dist - th) / vel
    under = 2.0 * np.sqrt(np.maximum(dist, 0.0) / acc)
    return np.where(dist >= th, over, under)


class Travel:
    """max(T_x, T_y) between points, vectorised the way the checker does it."""

    def __init__(self, v_x, v_y, a_x, a_y):
        self.v_x, self.v_y, self.a_x, self.a_y = v_x, v_y, a_x, a_y

    def __call__(self, px, py, qx, qy):
        return np.maximum(axis_time(np.abs(np.asarray(px, float) - qx), self.v_x, self.a_x),
                          axis_time(np.abs(np.asarray(py, float) - qy), self.v_y, self.a_y))

    def pairwise(self, x, y):
        """Full n x n travel-time matrix over FOV centres."""
        dx = np.abs(x[:, None] - x[None, :])
        dy = np.abs(y[:, None] - y[None, :])
        return np.maximum(axis_time(dx, self.v_x, self.a_x),
                          axis_time(dy, self.v_y, self.a_y))

    def swapped(self):
        """Travel model seen after transposing the board (x <-> y)."""
        return Travel(self.v_y, self.v_x, self.a_y, self.a_x)


# --------------------------------------------------------------------------
# cycle time
# --------------------------------------------------------------------------

def ct_for_k(t_move, t_img, t_recon, t_comp, steps, k):
    """Cycle time of one FOV sequence on `k` cores.

    t_move[i]  stage time from FOV i-1 to FOV i  (t_move[0] unused)
    t_img[i]   capture time of FOV i, already max(capture, side_capture)
    t_recon    scalar 3D-reconstruction time per FOV
    t_comp[i]  list of inspection times of FOV i, in the order comp_idx lists
               them -- the checker feeds cores 3.. in exactly this order
    steps[i]   step level of FOV i (0/1); a change costs Z_DELAY on core 0
    """
    n = len(t_img)
    if n == 0:
        return 0.0

    n_ins = k - 3                       # cores 3..k-1 do inspection
    core0 = t_img[0]
    recon = [core0, core0]              # cores 1 and 2
    ins = [core0] * n_ins

    for i in range(n):
        if i:
            core0 += t_move[i]
            if steps[i] != steps[i - 1]:
                core0 += Z_DELAY
            core0 += t_img[i]
            # the grader's sync: every core is pulled up to core 0's clock
            if recon[0] < core0:
                recon[0] = core0
            if recon[1] < core0:
                recon[1] = core0
            for c in range(n_ins):
                if ins[c] < core0:
                    ins[c] = core0

        # 3D reconstruction on the earlier-free of cores 1 and 2
        r = 0 if recon[0] <= recon[1] else 1
        done = recon[r] + t_recon
        recon[r] = done

        # inspection: list-schedule this FOV's components onto cores 3..
        for t in t_comp[i]:
            best, best_at = 0, ins[0] if ins[0] > done else done
            for c in range(1, n_ins):
                at = ins[c] if ins[c] > done else done
                if at < best_at:
                    best, best_at = c, at
            ins[best] = best_at + t

    return max(core0, recon[0], recon[1], max(ins))


def score(t_move, t_img, t_recon, t_comp, steps, max_core):
    """(J, k*) -- the checker's own minimisation, which is what the score pays.

    CT is non-increasing in k while the BETA*k term grows, so once
    CT(max_core) + BETA*k exceeds the incumbent no larger k can win and the
    sweep stops.  On these boards that usually settles after two or three k.
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
# packaging a plan for the simulator
# --------------------------------------------------------------------------

class Instance:
    """Everything the reward needs, fixed once the cover is chosen."""

    def __init__(self, travel, t_recon, max_core):
        self.travel = travel
        self.t_recon = float(t_recon)
        self.max_core = max(int(max_core), 4)

    def evaluate(self, x, y, t_img, t_comp, steps):
        """J and k* of a FOV sequence given as parallel arrays."""
        n = len(x)
        if n == 0:
            return 0.0, 4
        t_move = np.empty(n)
        t_move[0] = 0.0
        if n > 1:
            t_move[1:] = self.travel(x[:-1], y[:-1], x[1:], y[1:])
        return score(t_move, t_img, self.t_recon, t_comp, steps, self.max_core)
