"""Rolling AOI schedule improvement and reduced-coordinate QP refinement.

These are offline, MPC-inspired neighbourhoods, not a feedback controller.
Every accepted update is checked against the full deterministic contest model.
"""
import time
import numpy as np
from yumin_solver_4 import sim, refine
from .qp import laguerre_basis, solve_qp, tighten_box


def membership_bound(plan):
    """Order/position-independent bound for the current fixed shot memberships.

    Relax all travel and Z transitions, and allow the earliest possible first
    capture. Work and reconstruction count remain unavoidable. Unlike the
    fixed-route bound, this can gate BOTH routing and position refinement.
    """
    if not len(plan.x):
        return 0.0
    img, jobs = plan.t_img(), plan.t_comp()
    first = float(img.min())
    work = sum(sum(js) for js in jobs)
    largest = max((max(js, default=0.0) for js in jobs), default=0.0)
    recon = plan.inst.t_recon
    floor = max(float(img.sum()), first + ((len(img) + 1) // 2) * recon,
                first + recon + largest)
    return min(max(floor, first + recon + work / (k - 3)) + sim.BETA * k
               for k in range(4, plan.inst.max_core + 1))


def schedule_bound(plan):
    """Lower bound for THIS fixed route, geometry and assignment only.

    Exact recon release times plus divisible suffix inspection work relax the
    list scheduling problem. This is not a global bound across other covers.
    """
    n = len(plan.x)
    if not n:
        return 0.0
    img = plan.t_img()
    moves = plan.inst.moves(plan.x, plan.y)
    jumps = np.r_[0.0, np.diff(plan.step) != 0] * sim.Z_DELAY
    chain = np.cumsum(moves + img + jumps)
    ready = np.empty(n)
    recon = [0.0, 0.0]
    for i, release in enumerate(chain):
        c = 0 if recon[0] <= recon[1] else 1
        recon[c] = max(recon[c], release) + plan.inst.t_recon
        ready[i] = recon[c]
    jobs = plan.t_comp()
    loads = np.array([sum(js) for js in jobs])
    tail = np.cumsum(loads[::-1])[::-1]
    longest = np.array([max(js, default=0.0) for js in jobs])
    fixed = max(float(chain[-1]), max(recon), float(np.max(ready + longest)))
    return min(max(fixed, float(np.max(ready + tail / (k - 3)))) + sim.BETA * k
               for k in range(4, plan.inst.max_core + 1))


def _reorder(plan, order):
    result = plan.copy()
    result.x, result.y = plan.x[order], plan.y[order]
    result.step = [plan.step[i] for i in order]
    result.block = [plan.block[i] for i in order]
    result.members = [plan.members[i][:] for i in order]
    return result


def rolling_order(plan, prefix, deadline, stats, horizon=6):
    """Roll a horizon forward, retaining and evaluating the full suffix.

    Cache geometry and per-shot jobs: candidate generation only permutes
    indices, rather than rebuilding every component list for every proposal.
    """
    n = len(plan.x)
    order = np.arange(n)
    images, jobs = plan.t_img(), plan.t_comp()
    steps, blocks = np.asarray(plan.step), np.asarray(plan.block)
    loads = np.array([sum(js) for js in jobs])
    longest = np.array([max(js, default=0.0) for js in jobs])
    distances = plan.inst.travel.pairwise(plan.x, plan.y)
    recon_time = plan.inst.t_recon
    ks = np.arange(4, plan.inst.max_core + 1)

    def timing(seq):
        moves = np.r_[0.0, distances[seq[:-1], seq[1:]]]
        chain = np.cumsum(moves + images[seq] +
                          np.r_[0.0, steps[seq[1:]] != steps[seq[:-1]]] * sim.Z_DELAY)
        # Equal reconstruction durations imply FIFO alternating cores.
        ready = chain.copy()
        for i in range(n):
            ready[i] = max(chain[i], ready[i-2] if i >= 2 else 0.0) + recon_time
        tail = np.cumsum(loads[seq][::-1])[::-1]
        fixed = max(chain[-1], ready.max(), np.max(ready + longest[seq]))
        bound = float(np.min(np.maximum(fixed, np.max(
            ready[:, None] + tail[:, None] / (ks[None, :] - 3), axis=0)) + 10 * ks))
        return moves, bound

    best_j = plan.score()[0]
    for start in range(prefix, n - 1):
        if time.perf_counter() >= deadline:
            break
        candidates = []
        for end in range(start + 1, min(n, start + horizon)):
            if blocks[order[end]] != blocks[order[start]]:
                break
            for reverse in (False, True):
                if reverse and end == start + 1:  # same move as relocation
                    continue
                seq = order.copy()
                if reverse:
                    seq[start:end+1] = seq[start:end+1][::-1]
                else:
                    seq[start+1:end+1] = order[start:end]
                    seq[start] = order[end]
                moves, bound = timing(seq)
                stats['route_candidates'] += 1
                if bound < best_j - 1e-8:
                    candidates.append((bound, seq, moves))
                else:
                    stats['bound_pruned'] += 1
        for _, seq, moves in sorted(candidates, key=lambda item: item[0]):
            if time.perf_counter() >= deadline:
                break
            j, _ = sim.score(moves, images[seq], recon_time,
                             [jobs[i] for i in seq], steps[seq], plan.inst.max_core)
            if j < best_j - 1e-8:
                order, best_j = seq, j
                stats['route_accepted'] += 1
    return _reorder(plan, order)


def _axis_proposal(values, lower, upper, start, end, basis, edge_weight, margin):
    """Convex weighted squared-distance QP on a window, boundaries fixed."""
    n, h = len(values), end - start
    lo, hi = tighten_box(lower[start:end], upper[start:end], margin)
    anchor = (lo + hi) * 0.5
    B = basis.copy()
    B[(hi - lo) < 1e-10] = 0.0
    # Neighbour differences include both edges of the window. No initial
    # origin edge: the grader does not charge a move before the first FOV.
    first, last = max(1, start), min(n - 1, end)
    D = np.zeros((last - first + 1, h))
    offset = np.zeros(len(D))
    for row, i in enumerate(range(first, last + 1)):
        for j, sign in ((i, 1.0), (i - 1, -1.0)):
            if start <= j < end:
                D[row, j - start] = sign
            else:
                offset[row] += sign * values[j]
    weights = edge_weight[first - 1:last]
    DB = D @ B
    error = D @ anchor + offset
    # Small ridge keeps H positive definite, including pinned/short windows.
    H = 2 * (DB.T @ (weights[:, None] * DB) + 1e-5 * np.eye(B.shape[1]))
    q = 2 * DB.T @ (weights * error)
    A = np.vstack((B, -B))
    b = np.r_[hi - anchor, anchor - lo]
    result = solve_qp(H, q, A, b)
    return anchor + B @ result.x, result


def qp_positions(plan, board, deadline, stats, horizon=12, order=4,
                 poles=(0.0, 0.5, 0.85), margin=1e-7, basis_mode='laguerre'):
    best, best_j = plan, plan.score()[0]
    n = len(plan.x)
    if n < 2:
        return best
    lx, hx, ly, hy, _ = refine.shot_boxes(
        plan, board.lo_x, board.hi_x, board.lo_y, board.hi_y,
        board.big | (board.type == 2))
    for start in range(0, n, max(1, horizon // 2)):
        if time.perf_counter() >= deadline:
            break
        end = min(n, start + horizon)
        h = end - start
        if h < 2:
            break
        travel = best.inst.travel
        # IRLS-like local quadratic surrogate for triangular/trapezoidal move
        # time. This surrogate is NOT a certified bound on contest CT.
        dx, dy = np.abs(np.diff(best.x)), np.abs(np.diff(best.y))
        tx = sim.axis_time(dx, travel.v_x, travel.a_x)
        ty = sim.axis_time(dy, travel.v_y, travel.a_y)
        wx = (0.1 + 0.9 * (tx >= ty)) / np.maximum(dx, 1.0) ** 1.5
        wy = (0.1 + 0.9 * (ty >= tx)) / np.maximum(dy, 1.0) ** 1.5
        for pole in (poles if basis_mode == 'laguerre' else (0.0,)):
            if time.perf_counter() >= deadline:
                break
            if basis_mode == 'laguerre':
                B = np.cumsum(laguerre_basis(h, min(order, h), pole), axis=0)
                B /= np.maximum(np.linalg.norm(B, axis=0), 1e-12)
            elif basis_mode == 'full':
                B = np.eye(h)
            else:
                raise ValueError('basis_mode must be laguerre or full')
            x, rx = _axis_proposal(best.x, lx, hx, start, end, B, wx, margin)
            y, ry = _axis_proposal(best.y, ly, hy, start, end, B, wy, margin)
            stats['qp_solved'] += 2
            stats['qp_max_gap'] = max(stats['qp_max_gap'], rx.gap, ry.gap)
            stats['qp_max_violation'] = max(stats['qp_max_violation'], rx.violation, ry.violation)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                continue
            if (np.any(x < lx[start:end] - 1e-8) or np.any(x > hx[start:end] + 1e-8)
                    or np.any(y < ly[start:end] - 1e-8) or np.any(y > hy[start:end] + 1e-8)):
                continue
            trial = best.copy()
            trial.x[start:end], trial.y[start:end] = x, y
            # The exact objective, including reconstruction and every core
            # count, decides pole selection and acceptance.
            j = trial.score()[0]
            if j < best_j - 1e-8:
                best, best_j = trial, j
                stats['qp_accepted'] += 1
                stats['accepted_poles'].append(float(pole))
    return best
