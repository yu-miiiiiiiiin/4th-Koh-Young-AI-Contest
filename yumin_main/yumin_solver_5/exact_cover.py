"""Exact and learning-guided improvement of the geometric FOV cover.

For an ordinary component i, every legal FOV centre lies in an axis-aligned
box Bi.  There is an optimum whose centres are pairs of upper box boundaries,
so a finite set-cover instance is exact.  Small independent pieces are solved
completely.  On large pieces the same solver is used as an exact repair oracle
inside a learning-guided LNS.

The rankers only order work.  They never remove a column or prune a B&B node,
so a completed search retains its exactness regardless of model quality.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from yumin_solver_4 import cover as cover4


TOL = 1e-9
MAX_EXACT_FOVS = 11
MAX_EXACT_COMPONENTS = 96
MAX_REPAIR_COMPONENTS = 128
LNS_WIDTH = 8


class OnlineLogisticRanker:
    """Tiny online model suitable for the scored CPU-only evaluator."""

    def __init__(self, weights, lr=0.035):
        self.w = np.asarray(weights, dtype=float)
        self.lr = float(lr)
        self.n_update = 0

    def score(self, feature):
        return float(np.dot(self.w, feature))

    def observe(self, feature, label):
        z = float(np.clip(self.score(feature), -20.0, 20.0))
        p = 1.0 / (1.0 + math.exp(-z))
        rate = self.lr / math.sqrt(1.0 + self.n_update / 64.0)
        self.w += rate * (float(label) - p) * np.asarray(feature)
        self.n_update += 1


# Module lifetime spans all boards in 2_run.py, so these models adapt across
# boards without any file writes or test-time gradient-heavy training.
_COLUMN_RANKER = OnlineLogisticRanker(
    [0.0, 3.2, 2.1, 0.7, 0.3, 0.2, -0.25, 0.4], lr=0.025)
_NEIGHBOUR_RANKER = OnlineLogisticRanker(
    [-0.2, 1.1, 1.8, 0.6, 0.5, -0.3, 0.4], lr=0.04)


@dataclass
class RepairResult:
    groups: list | None
    complete: bool
    nodes: int
    features: list


def _iter_bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


def _column_features(mask, px, py, pool, board, frequency):
    loc = list(_iter_bits(mask))
    members = pool[loc]
    n = max(1, len(pool))
    gain = len(loc) / n
    rarity = sum(1.0 / max(1, frequency[k]) for k in loc) / n
    a = float(board.lo_x[members].max())
    b = float(board.hi_x[members].min())
    c = float(board.lo_y[members].max())
    d = float(board.hi_y[members].min())
    slack_x = max(0.0, b - a) / max(board.fov_w, 1e-9)
    slack_y = max(0.0, d - c) / max(board.fov_h, 1e-9)
    side = float(board.side[members].mean()) if len(members) else 0.0
    load = float(board.time[members].sum())
    load /= max(float(board.time[pool].sum()), 1e-9)
    cx = (float(board.tl_x[pool].min()) + float(board.br_x[pool].max())) / 2.0
    cy = (float(board.tl_y[pool].min()) + float(board.br_y[pool].max())) / 2.0
    dist = math.hypot((px - cx) / max(board.fov_w, 1e-9),
                      (py - cy) / max(board.fov_h, 1e-9))
    return np.array([1.0, gain, rarity, slack_x, slack_y, side, dist, load])


def _candidate_columns(pool, board, deadline):
    """Return all distinct, non-dominated coverage masks.

    If p hits a collection of boxes, moving p right/up until it meets an upper
    boundary preserves that collection.  Therefore hi_x x hi_y is a complete
    event-point set for ordinary components.
    """
    pool = np.asarray(pool, dtype=np.int64)
    xs = np.unique(board.hi_x[pool])
    ys = np.unique(board.hi_y[pool])
    raw = {}
    for x in xs:
        if time.perf_counter() >= deadline:
            return None, None
        active_x = ((board.lo_x[pool] <= x + TOL)
                    & (board.hi_x[pool] >= x - TOL))
        if not active_x.any():
            continue
        # All y event points for this x in one vectorised pass.  packbits turns
        # each coverage row directly into the Python integer used by B&B.
        active_y = ys[np.isin(ys, board.hi_y[pool][active_x])]
        hit = (active_x[None, :]
               & (board.lo_y[pool][None, :] <= active_y[:, None] + TOL)
               & (board.hi_y[pool][None, :] >= active_y[:, None] - TOL))
        packed = np.packbits(hit, axis=1, bitorder='little')
        for row, y in zip(packed, active_y):
            mask = int.from_bytes(row.tobytes(), 'little')
            if mask:
                raw.setdefault(mask, (float(x), float(y)))

    # A unit-cost column contained in another column can never be necessary.
    masks = sorted(raw, key=int.bit_count, reverse=True)
    if len(masks) <= 2500:
        kept = []
        for at, mask in enumerate(masks):
            if (at & 63) == 0 and time.perf_counter() >= deadline:
                return None, None
            if any(mask | sup == sup for sup in kept):
                continue
            kept.append(mask)
        masks = kept

    frequency = np.zeros(len(pool), dtype=np.int32)
    for mask in masks:
        for i in _iter_bits(mask):
            frequency[i] += 1
    features = [_column_features(mask, *raw[mask], pool, board, frequency)
                for mask in masks]
    return masks, features


def _exact_repair(pool, incumbent, board, deadline):
    """Find a cover strictly smaller than incumbent; prove absence if complete."""
    pool = np.asarray(sorted(set(int(v) for v in pool)), dtype=np.int64)
    if len(pool) == 0 or incumbent <= 1 or time.perf_counter() >= deadline:
        return RepairResult(None, False, 0, [])
    # A single dense/grid FOV can contain hundreds of almost interchangeable
    # top components.  Materialising their O(n^2) event grid cannot respect a
    # sub-second scored budget, so such a neighbourhood is left untouched.
    if len(pool) > MAX_REPAIR_COMPONENTS:
        return RepairResult(None, False, 0, [])

    masks, features = _candidate_columns(pool, board, deadline)
    if masks is None or time.perf_counter() >= deadline or not masks:
        return RepairResult(None, False, 0, features or [])

    by_element = [[] for _ in range(len(pool))]
    for j, mask in enumerate(masks):
        for i in _iter_bits(mask):
            by_element[i].append(j)

    full = (1 << len(pool)) - 1
    best_count = int(incumbent)
    best_choice = None
    memo = {}
    nodes = 0
    timed_out = False

    class _Timeout(Exception):
        pass

    def dfs(uncovered, chosen):
        nonlocal best_count, best_choice, nodes
        nodes += 1
        if (nodes & 127) == 0 and time.perf_counter() >= deadline:
            raise _Timeout
        if uncovered == 0:
            best_count = len(chosen)
            best_choice = tuple(chosen)
            return
        depth = len(chosen)
        if depth + 1 >= best_count:
            return
        old = memo.get(uncovered)
        if old is not None and old <= depth:
            return
        memo[uncovered] = depth

        max_gain = max((mask & uncovered).bit_count() for mask in masks)
        if max_gain == 0:
            return
        lower = (uncovered.bit_count() + max_gain - 1) // max_gain
        if depth + lower >= best_count:
            return

        # Most constrained uncovered component first.
        elems = list(_iter_bits(uncovered))
        pivot = min(elems, key=lambda e: len(by_element[e]))
        options = by_element[pivot]
        options = sorted(
            options,
            key=lambda j: (_COLUMN_RANKER.score(features[j])
                           + 2.5 * (masks[j] & uncovered).bit_count() / len(pool)),
            reverse=True,
        )
        for j in options:
            new_uncovered = uncovered & ~masks[j]
            if new_uncovered == uncovered:
                continue
            dfs(new_uncovered, chosen + [j])

    try:
        dfs(full, [])
    except _Timeout:
        timed_out = True

    if best_choice is None:
        return RepairResult(None, not timed_out, nodes, features)

    # Assign to the first chosen column.  Because every branch covers a new
    # pivot, each chosen column retains at least one member under this order.
    assigned = [[] for _ in best_choice]
    for local in range(len(pool)):
        for k, j in enumerate(best_choice):
            if masks[j] & (1 << local):
                assigned[k].append(int(pool[local]))
                break
    groups = [g for g in assigned if g]

    if not timed_out:
        selected = set(best_choice)
        for j, feat in enumerate(features):
            _COLUMN_RANKER.observe(feat, 1.0 if j in selected else 0.0)
    return RepairResult(groups, not timed_out, nodes, features)


def _make_fov(members, step, board):
    ms = np.asarray(members, dtype=np.int64)
    f = {
        'lo_x': float(board.lo_x[ms].max()),
        'hi_x': float(board.hi_x[ms].min()),
        'lo_y': float(board.lo_y[ms].max()),
        'hi_y': float(board.hi_y[ms].min()),
        'comp': [int(v) for v in members],
        'step': int(step),
    }
    f['x'] = (f['lo_x'] + f['hi_x']) / 2.0
    f['y'] = (f['lo_y'] + f['hi_y']) / 2.0
    board.decorate(f)
    return f


def _replace(fovs, indices, groups, board):
    indices = sorted(set(int(i) for i in indices))
    step = int(fovs[indices[0]]['step'])
    drop = set(indices)
    kept = [f for i, f in enumerate(fovs) if i not in drop]
    kept.extend(_make_fov(group, step, board) for group in groups)
    fovs[:] = kept


def _neighbour_feature(fovs, indices, board):
    pool = np.asarray([m for i in indices for m in fovs[i]['comp']], dtype=np.int64)
    xs = np.asarray([fovs[i]['x'] for i in indices])
    ys = np.asarray([fovs[i]['y'] for i in indices])
    span = ((xs.max() - xs.min()) / max(board.fov_w, 1e-9)
            + (ys.max() - ys.min()) / max(board.fov_h, 1e-9))
    slack = np.mean((board.hi_x[pool] - board.lo_x[pool]) / max(board.fov_w, 1e-9)
                    + (board.hi_y[pool] - board.lo_y[pool]) / max(board.fov_h, 1e-9))
    side = float(board.side[pool].mean()) if len(pool) else 0.0
    load = float(board.time[pool].sum()) / max(len(pool), 1)
    return np.array([1.0, len(indices) / LNS_WIDTH,
                     len(pool) / max(1.0, 12.0 * len(indices)),
                     float(slack), side, float(span), math.tanh(load)])


def improve_cover(fovs, board, seconds=0.16):
    """Improve ``fovs`` in place and return audit statistics."""
    started = time.perf_counter()
    deadline = started + max(0.0, float(seconds))
    stats = {'removed': 0, 'certified_blobs': 0, 'timed_out': 0,
             'repairs': 0, 'nodes': 0}

    # First solve every small independent interaction blob completely.
    for step in sorted(set(int(f['step']) for f in fovs)):
        ids = [i for i, f in enumerate(fovs)
               if int(f['step']) == step and not f.get('pinned')]
        if not ids or time.perf_counter() >= deadline:
            break
        comps = sorted(set(m for i in ids for m in fovs[i]['comp']
                           if not board.big[m] and board.type[m] != 2))
        if not comps:
            continue
        pos = np.asarray(comps, dtype=np.int64)
        cx = (board.tl_x[pos] + board.br_x[pos]) / 2.0
        cy = (board.tl_y[pos] + board.br_y[pos]) / 2.0
        blobs = cover4.components_of(cx, cy, board.fov_w, board.fov_h)
        owner = {m: i for i in ids for m in fovs[i]['comp']}
        for blob in blobs:
            pool = [int(pos[k]) for k in blob]
            take = sorted(set(owner[m] for m in pool if m in owner))
            if (not take or len(take) > MAX_EXACT_FOVS
                    or len(pool) > MAX_EXACT_COMPONENTS):
                continue
            result = _exact_repair(pool, len(take), board, deadline)
            stats['repairs'] += 1
            stats['nodes'] += result.nodes
            stats['certified_blobs'] += int(result.complete)
            stats['timed_out'] += int(not result.complete)
            if result.groups is not None and len(result.groups) < len(take):
                before = len(take)
                _replace(fovs, take, result.groups, board)
                stats['removed'] += before - len(result.groups)
                # Indices changed; finish exact-blob phase safely and let LNS
                # discover any further neighbourhoods on the new cover.
                break

    # Learning-guided LNS for pieces too large for a global exact solve.
    attempted = set()
    while time.perf_counter() < deadline:
        free = [i for i, f in enumerate(fovs) if not f.get('pinned')]
        if len(free) < 3:
            break
        proposals = []
        for seed in free:
            step = int(fovs[seed]['step'])
            same = [i for i in free if int(fovs[i]['step']) == step]
            same.sort(key=lambda i: ((fovs[i]['x'] - fovs[seed]['x']) ** 2
                                     + (fovs[i]['y'] - fovs[seed]['y']) ** 2))
            take = tuple(sorted(same[:min(LNS_WIDTH, len(same))]))
            if len(take) < 3 or take in attempted:
                continue
            feat = _neighbour_feature(fovs, take, board)
            proposals.append((_NEIGHBOUR_RANKER.score(feat), take, feat))
        if not proposals:
            break
        _, take, feat = max(proposals, key=lambda z: z[0])
        attempted.add(take)
        pool = [m for i in take for m in fovs[i]['comp']]
        result = _exact_repair(pool, len(take), board, deadline)
        stats['repairs'] += 1
        stats['nodes'] += result.nodes
        stats['timed_out'] += int(not result.complete)
        success = result.groups is not None and len(result.groups) < len(take)
        _NEIGHBOUR_RANKER.observe(feat, float(success))
        if success:
            before = len(take)
            _replace(fovs, take, result.groups, board)
            stats['removed'] += before - len(result.groups)
            attempted.clear()

    stats['seconds'] = time.perf_counter() - started
    return stats
