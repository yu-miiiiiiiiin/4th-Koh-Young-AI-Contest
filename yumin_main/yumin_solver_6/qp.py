"""Small convex QPs with a feasible incumbent and a genuine dual certificate.

min 0.5*z.T@H@z + q.T@z, A@z <= b. H must be positive definite;
zero must be feasible. No external optimisation package is required.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class QPResult:
    x: np.ndarray
    primal: float
    dual: float
    gap: float
    violation: float
    sweeps: int


def solve_qp(H, q, A, b, max_sweeps=24, tolerance=1e-7):
    H, q, A, b = (np.asarray(v, dtype=float) for v in (H, q, A, b))
    if not all(np.all(np.isfinite(v)) for v in (H, q, A, b)):
        raise ValueError('QP data must be finite')
    if np.any(b < 0):
        raise ValueError('zero must be feasible')
    # Hildreth: coordinate ascent of g(lambda), lambda >= 0.
    inv_q = np.linalg.solve(H, q)
    inv_a = np.linalg.solve(H, A.T)
    M = A @ inv_a
    diag = np.diag(M)
    c = b + A @ inv_q
    lam = np.zeros(len(b))
    residual = c.copy()
    best_x = np.zeros(len(q))
    best_primal, best_dual = 0.0, -0.5 * float(q @ inv_q)
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        for i in range(len(b)):
            if diag[i] <= 1e-18:
                continue
            delta = max(0.0, lam[i] - residual[i] / diag[i]) - lam[i]
            if delta:
                lam[i] += delta
                residual += delta * M[:, i]
        z = -inv_q - inv_a @ lam
        az = A @ z
        positive = az > 0
        fraction = min(1.0, float(np.min(b[positive] / az[positive]))) if positive.any() else 1.0
        # A radial feasible recovery, NOT clipping coordinates out of the basis.
        feasible = z * max(0.0, fraction * (1.0 - 1e-12))
        primal = float(0.5 * feasible @ H @ feasible + q @ feasible)
        v = q + A.T @ lam
        dual = float(-0.5 * v @ np.linalg.solve(H, v) - b @ lam)
        if primal < best_primal:
            best_x, best_primal = feasible, primal
        best_dual = max(best_dual, dual)
        if best_primal - best_dual <= tolerance * (1 + abs(best_primal)):
            break
    violation = max(0.0, float(np.max(A @ best_x - b, initial=0.0)))
    return QPResult(best_x, best_primal, best_dual,
                    max(0.0, best_primal - best_dual), violation, sweeps)


def laguerre_basis(horizon, order=4, pole=0.5):
    """Discrete Laguerre recurrence: Jeong & Choi (2022), equation (17)."""
    if horizon < 1 or not 1 <= order <= horizon or not 0 <= pole < 1:
        raise ValueError('require horizon >= order >= 1 and 0 <= pole < 1')
    beta = 1.0 - pole * pole
    state = np.sqrt(beta) * (-pole) ** np.arange(order)
    transition = np.eye(order) * pole
    for i in range(1, order):
        for j in range(i):
            transition[i, j] = beta * (-pole) ** (i - j - 1)
    basis = np.empty((horizon, order))
    for i in range(horizon):
        basis[i] = state
        state = transition @ state
    return basis


def tighten_box(lo, hi, margin):
    """Numerical interior margin, capped to avoid empty or pinned intervals.

    The achieved margin may be smaller than requested: this is NOT a physical
    disturbance guarantee. A genuine tube requires a specified disturbance set.
    """
    if not np.isfinite(margin) or margin < 0:
        raise ValueError('margin must be finite and nonnegative')
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if np.any(lo > hi + 1e-8):
        raise ValueError('infeasible coverage box')
    mid = (lo + hi) / 2
    lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)
    eps = np.minimum(margin, (hi - lo) * 0.25)
    return lo + eps, hi - eps
