"""Scan order: which FOV to shoot next.

POMO (Kwon et al., NeurIPS 2020) supplies the search, and three of its
mechanisms are used in structure exactly as published:

  S4.1  Multiple starting nodes.  A scan plan is a sequence, and the first FOV
        of a permutable block is a free choice, so one plan has many sequence
        representations.  N rollouts are forced from N *different* first FOVs.

  S4.2  Shared baseline.  b(s) = mean_j R(tau_j) (their Eq. 4), with the
        REINFORCE step of their Eq. 3.  The advantage has zero mean by
        construction, so no trajectory is judged against a critic that fails
        the same way it does.

  S4.3  Multiple greedy rollouts + instance augmentation.  Inference takes the
        argmax over greedy rollouts across the 8 unit-square transforms of
        their Table 1 (their Algorithm 2).

NOTE -- what is not from the paper.  POMO trains an Attention Model offline for
days on a GPU.  Runtime is billed at 1 point/second here (4_score.py, alpha=1),
so the policy is instead a softmax over w . phi(j | state) with seven hand-built
features, trained *per board* by the POMO update.  The algorithm carries over;
the learned representation does not.

The reward is the exact number the grader pays, R(tau) = -J(tau) from sim.py,
so the search trades stage travel, the 5 s Z transitions, the inspection tail
and the core count k* against each other instead of minimising distance.

Two things here are not in solver 3, and both come from measuring it:

  * `search` stops on a stall, not on a clock.  Re-routing solver 3's covers
    with 40x its budget moved the 96-board total by under a point: the policy
    converges in a fraction of a second and then spends real score on runtime.
    The budget is now a ceiling, and the stall counter is what normally ends
    the search.
  * `seeds` hands the search a structured incumbent per step arrangement, so
    the Z-transition decision is made by construction and confirmed by the
    exact reward, rather than having to be discovered by the policy.
"""
import numpy as np

from . import sim

# POMO Table 1: the eight unit-square transforms on normalised coordinates.
FRAMES = (
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
)   # (swap, flip_u, flip_v)

N_FEAT = 7
N_START = 8
N_FRAME = 8
STALL = 4            # validated on fresh generator seeds: later probes cost more than they gain
STALL_EXTENDED = 6
STALL_EXTEND_GATE = 7.0
RELAX_PASSES = 2

# Starting policy and step size for the per-board REINFORCE loop.
#
# POMO trains its policy offline before it is ever asked to solve anything.  The
# per-board loop here cannot -- runtime is scored -- but the *starting point*
# can be trained offline and costs nothing at solve time, so it was: a (1+1)
# evolution strategy over these eight numbers, scored by the summed J of all 96
# practice boards after search and polish, took the route-only total from
# 12867 to 12791.  What it learned, against the hand-set
# [2.5, 4.0, 0.6, 1.2, 0.3, 0.2, 0.1], is that travel should dominate far more
# than a human would set it (2.5 -> 6.5, and the search pushed against an
# upper clamp of 8 repeatedly), that load shaping is worth five times what was
# guessed, and that the last two features are worth about nothing.
#
# Fitted on the practice set, so it is a starting point and not a conclusion:
# the per-board REINFORCE loop still moves it, and the exact reward still
# decides.  `tune_policy.py` regenerates these.
LR = 0.212
# travel, step-change, sweep, row-lock, load-shaping, isolation, mass-pull
W0 = np.array([6.475, 2.996, 0.725, 0.265, 1.556, 0.276, -0.067])


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _unit(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)


class Router:
    """Sequencing problem for one board.

    FOVs arrive unordered from cover.py.  `block` is the ordering class rule 3.1
    pins (type 2 before 1 before 0); fiducials are a fixed prefix (rules 2.3 and
    3.2) and never enter the search, the rest permutes freely inside its block.
    """

    def __init__(self, inst, fovs, prefix):
        self.inst = inst
        self.fovs = fovs
        self.prefix = prefix

        # The prefix is not permutable but it is part of the same pipeline: it
        # seeds core 0's clock and leaves the stage on its last FOV.  Scoring
        # without it would hide the entry travel.
        self.pre_x = np.array([f['x'] for f in prefix])
        self.pre_y = np.array([f['y'] for f in prefix])
        self.pre_step = [f['step'] for f in prefix]
        self.pre_img = [f['t_img'] for f in prefix]
        self.pre_comp = [f['t_comp'] for f in prefix]

        self.entry = (prefix[-1]['x'], prefix[-1]['y']) if prefix else None
        # -1 never equals a real step level, so with no fiducial "did Z change?"
        # is simply always false before the first capture
        self.entry_step = prefix[-1]['step'] if prefix else -1

        n = len(fovs)
        self.n = n
        self.px = np.array([f['x'] for f in fovs])
        self.py = np.array([f['y'] for f in fovs])
        self.block = np.array([f['block'] for f in fovs])
        self.step = np.array([f['step'] for f in fovs])
        self.load = np.array([f['load'] for f in fovs])
        self.t_img = np.array([f['t_img'] for f in fovs])
        self.t_comp = [f['t_comp'] for f in fovs]

        self.D = inst.travel.pairwise(self.px, self.py) if n else np.zeros((0, 0))
        if n and self.entry is not None:
            self.d_entry = inst.travel(self.entry[0], self.entry[1], self.px, self.py)
        else:
            # no fiducial: the first capture happens wherever the scan opens and
            # the grader charges no move before it
            self.d_entry = np.zeros(n)

        # feature scales, so one weight vector means the same thing on a
        # 1000 mm/s board and a 200 mm/s one
        self.d_scale = float(np.median(self.D[self.D > 0])) if np.any(self.D > 0) else 1.0
        self.l_scale = float(self.load.mean()) if n and self.load.mean() > 0 else 1.0

        if n > 1:
            d = self.D + np.eye(n) * 1e18
            self.nn = d.min(axis=1) / self.d_scale
        else:
            self.nn = np.zeros(n)

        self.frames = []
        if n:
            u0, v0 = _unit(self.px), _unit(self.py)
            for swap, fu, fv in FRAMES:
                u, v = (v0, u0) if swap else (u0, v0)
                self.frames.append((1.0 - u if fu else u, 1.0 - v if fv else v))
        self.max_block = int(self.block.max()) if n else 0
        self._census = {}
        for v_ in self.block:
            self._census[int(v_)] = self._census.get(int(v_), 0) + 1

    # ---------------------------------------------------------------- reward

    def evaluate(self, seq, relax=True):
        """R(tau) = -J(tau): exactly what 4_score.py charges for this board."""
        x = self.px[seq].copy()
        y = self.py[seq].copy()
        if relax:
            self.relax(seq, x, y)

        j, k = self.inst.evaluate(
            np.concatenate((self.pre_x, x)),
            np.concatenate((self.pre_y, y)),
            np.concatenate((self.pre_img, self.t_img[seq])),
            self.pre_comp + [self.t_comp[i] for i in seq],
            self.pre_step + [int(s) for s in self.step[seq]])
        return -j, k, x, y

    def relax(self, seq, x, y):
        """Slide each FOV inside its feasible box toward its two neighbours.

        The box is where the shot may legally sit (cover.py), so this is free
        travel time -- and a fiducial's box is a point, so rule 2.3's centring
        survives it untouched.
        """
        for _ in range(RELAX_PASSES):
            for i, f in enumerate(seq):
                fov = self.fovs[f]
                if i:
                    ax, ay = x[i - 1], y[i - 1]
                elif self.entry is not None:
                    ax, ay = self.entry
                else:
                    ax, ay = x[0], y[0]
                bx, by = (x[i + 1], y[i + 1]) if i + 1 < len(seq) else (ax, ay)
                x[i] = min(max((ax + bx) / 2.0, fov['lo_x']), fov['hi_x'])
                y[i] = min(max((ay + by) / 2.0, fov['lo_y']), fov['hi_y'])

    # --------------------------------------------------------------- rollout

    def rollout(self, w, frame, start, rng=None, lock=False):
        """One trajectory.  `start` fixes the first action (POMO S4.1);
        `rng=None` is the greedy rollout of Algorithm 2, otherwise the sampled
        rollout of Algorithm 1 with the log-prob gradient accumulated as it goes.

        `lock` holds the candidate set to the current Z height until that height
        is exhausted inside the ordering block.  A transition costs 5 s against
        a typical 0.2 s move (rules 4.7, 5.2), so one greedy slip costs twenty
        moves' worth; locking makes that structurally impossible but can strand
        a far corner.  Both variants are rolled out and the exact reward picks.
        """
        n = self.n
        if n == 0:
            return [], None
        u, v = self.frames[frame]
        left = np.ones(n, dtype=bool)
        seq = []
        grad = None if rng is None else np.zeros(N_FEAT)

        at = -1                              # -1 means "still at the entry point"
        cur_step = self.entry_step
        # running centroid and block census: recomputing either over `left` at
        # every step is what makes a rollout quadratic, and a rollout is the
        # inner loop of the whole search
        su, sv = float(u.sum()), float(v.sum())
        n_left = n
        census = self._census.copy()
        blk = self.max_block
        while census.get(blk, 0) == 0 and blk > 0:
            blk -= 1
        for t in range(n):
            while census.get(blk, 0) == 0 and blk > 0:
                blk -= 1
            ok = left & (self.block == blk)
            if lock:
                same = ok & (self.step == cur_step)
                if same.any():
                    ok = same
            cand = np.flatnonzero(ok)

            if t == 0 and start in cand:
                pick = start
            else:
                phi = self._features(cand, at, cur_step, u, v,
                                     su / n_left, sv / n_left, t / n)
                z = phi @ w
                if rng is None:
                    k = int(np.argmax(z))
                else:
                    p = _softmax(z)
                    k = int(rng.choice(len(cand), p=p))
                    grad += phi[k] - p @ phi
                pick = int(cand[k])

            seq.append(pick)
            left[pick] = False
            at = pick
            cur_step = self.step[pick]
            su -= u[pick]
            sv -= v[pick]
            n_left -= 1
            census[int(self.block[pick])] -= 1
        return seq, grad

    def _features(self, cand, at, cur_step, u, v, cu, cv, progress):
        """phi(j | state), columns in the order W0 documents."""
        d = (self.D[at, cand] if at >= 0 else self.d_entry[cand]) / self.d_scale
        v_at = v[at] if at >= 0 else v[cand].mean()

        phi = np.empty((len(cand), N_FEAT))
        phi[:, 0] = -d
        phi[:, 1] = -(self.step[cand] != cur_step).astype(float)
        phi[:, 2] = -u[cand]
        phi[:, 3] = -np.abs(v[cand] - v_at)
        # heavy FOVs early, light ones last: whatever the last FOV carries is
        # paid after the final capture, on top of the whole scan
        phi[:, 4] = (self.load[cand] / self.l_scale) * (1.0 - 2.0 * progress)
        phi[:, 5] = -self.nn[cand]
        phi[:, 6] = -(np.abs(u[cand] - cu) + np.abs(v[cand] - cv))
        return phi

    def _starts(self, n_start):
        """Distinct first FOVs, spread by farthest-point sampling: POMO's N
        starting nodes want to be as heterogeneous as the shared baseline can
        make use of."""
        eligible = np.flatnonzero(self.block == self.max_block)
        if len(eligible) <= n_start:
            return [int(i) for i in eligible]
        first = int(np.argmin(self.d_entry[eligible]))
        out = [int(eligible[first])]
        far = self.D[out[0], eligible].copy()
        far[first] = -1.0
        while len(out) < n_start:
            k = int(np.argmax(far))
            out.append(int(eligible[k]))
            far = np.minimum(far, self.D[out[-1], eligible])
            far[k] = -1.0
        return out

    # ---------------------------------------------------------------- search

    def search(self, budget, clock, rng, n_start=None, lr=None, stall=STALL,
               n_frame=None):
        """POMO Algorithm 1 (training) interleaved with Algorithm 2 (inference).

        `budget` is a ceiling.  What normally ends the search is `stall`
        consecutive iterations with no new best -- measured, the policy converges
        well inside a tenth of a second on these boards and every second after
        that is a point of runtime paid for nothing.
        """
        n_start = N_START if n_start is None else n_start
        lr = LR if lr is None else lr
        n_frame = max(1, min(N_FRAME if n_frame is None else n_frame, len(FRAMES)))
        if self.n == 0:
            return [], 0.0
        if self.n == 1:
            return [0], -self.evaluate([0])[0]

        starts = self._starts(n_start)
        w = W0.copy()

        # Everything here is ranked by the exact pipeline model.  Ranking by
        # the O(#shots) relaxation instead was measured twice -- as the ranker
        # and as a filter in front of the exact model -- and both lose more
        # score than the saved time buys back.  The relaxation's error is a
        # median 0.00 s but has a long tail, and two good rollouts differ by
        # less than that tail, so it inverts exactly the comparisons that
        # matter.  It earns its place in `shape`, where the move set is large
        # and the comparisons are coarse, and not here.
        best_seq, best_r, best_lock = None, -np.inf, False

        def consider(seq, lock=None):
            nonlocal best_seq, best_r, best_lock
            r = self.evaluate(seq)[0]
            if r > best_r:
                best_seq, best_r = seq, r
                if lock is not None:
                    best_lock = lock
                return True
            return False

        # a lattice cover's shortest path is a serpentine; start from one
        for seed in serpentine(self):
            consider(seed)

        # Algorithm 2 on the untrained policy first: greedy multi-start rollouts
        # over the frames are already a strong incumbent, and taking the argmax
        # means the search can only improve on it.  Locked and free variants
        # alternate across frames -- same rollout count, and the two disagree
        # often enough to be worth one each.
        for frame in range(n_frame):
            lock = bool(frame & 1)
            for s in starts:
                seq, _ = self.rollout(w, frame, s, lock=lock)
                consider(seq, lock)
            if clock() > budget * 0.45 and frame >= 1:
                break

        frame = 0
        quiet = 0
        stall_limit = stall
        while clock() < budget:
            if quiet >= stall_limit:
                # Most boards have converged after four misses.  A small subset
                # still sits far above its release-profile floor and contains
                # the rare multi-point late improvement; give only those the
                # former six-miss allowance.
                excess = -best_r - Shaper(self).floor(best_seq)
                if stall_limit < STALL_EXTENDED and excess > STALL_EXTEND_GATE:
                    stall_limit = STALL_EXTENDED
                else:
                    break
            rewards = np.empty(len(starts))
            grads, seqs = [], []
            for s in starts:
                seq, g = self.rollout(w, frame, s, rng, lock=best_lock)
                rewards[len(seqs)] = self.evaluate(seq)[0]
                grads.append(g)
                seqs.append(seq)

            b = rewards.mean()                              # POMO Eq. 4
            adv = rewards - b
            scale = adv.std()
            if scale > 1e-9:
                # advantages are seconds and vary by orders of magnitude across
                # boards; normalising keeps one learning rate usable
                step = sum(a * g for a, g in zip(adv / scale, grads)) / len(starts)
                w = w + lr * step                           # POMO Eq. 3

            gained = consider(seqs[int(np.argmax(rewards))])

            for s in starts[:2]:                            # cheap greedy probe
                seq, _ = self.rollout(w, frame, s, lock=best_lock)
                gained = consider(seq) or gained

            quiet = 0 if gained else quiet + 1
            frame = (frame + 1) % n_frame

        return best_seq, -best_r


def nearest(router):
    """Nearest-neighbour order inside each ordering class -- the baseline the
    contest guide describes, and the starting point the polish improves on."""
    n = router.n
    if n == 0:
        return []
    left = np.ones(n, dtype=bool)
    seq = []
    at = -1
    for _ in range(n):
        blk = router.block[left].max()
        cand = np.flatnonzero(left & (router.block == blk))
        d = router.D[at, cand] if at >= 0 else router.d_entry[cand]
        d = d + np.where(router.step[cand] != (router.step[at] if at >= 0
                                               else router.entry_step), sim.Z_DELAY, 0.0)
        pick = int(cand[int(np.argmin(d))])
        seq.append(pick)
        left[pick] = False
        at = pick
    return seq


# --------------------------------------------------------------------------
# post-processing (not POMO -- see module docstring)
# --------------------------------------------------------------------------

def polish(router, seq, budget, clock):
    """2-opt and Or-opt on core 0's timeline, each pass confirmed by the true J.

    The move cost is core 0's own clock -- stage travel plus the 5 s Z
    transitions (rules 5.2, 4.7) -- because that is local to the edges a move
    touches.  Reversal keeps a segment's interior adjacencies, so only the two
    seams change.  Core-0 cost is a proxy for J, so a pass that improves it is
    re-scored exactly and rolled back when J did not actually fall.

    Both neighbourhoods are swept with numpy over the whole candidate row at
    once rather than one candidate at a time: the sweep is O(n) array
    operations per anchor instead of O(n) interpreted iterations, which is what
    makes a full local optimum affordable inside a runtime that is itself
    scored.
    """
    n0 = len(seq)
    if n0 < 4:
        return seq
    D, step, block = router.D, router.step, router.block
    d_entry, Z = router.d_entry, sim.Z_DELAY
    has_entry = router.entry is not None
    entry_step = router.entry_step

    def head(v):
        """cost of shooting v first, straight out of the fiducial prefix"""
        if not has_entry:
            return 0.0
        return d_entry[v] + (Z if step[v] != entry_step else 0.0)

    def row(u, vs):
        """edge cost from u to every node in `vs` (u < 0 means the entry)"""
        if u < 0:
            if not has_entry:
                return np.zeros(len(vs))
            return d_entry[vs] + np.where(step[vs] != entry_step, Z, 0.0)
        return D[u, vs] + np.where(step[vs] != step[u], Z, 0.0)

    def chain_cost(node):
        """cost of every edge of the tour; the last entry is the open end"""
        e = np.zeros(len(node))
        e[0] = head(node[0])
        if len(node) > 1:
            e[1:] = D[node[:-1], node[1:]] + np.where(step[node[1:]] != step[node[:-1]], Z, 0.0)
        return e

    best = list(seq)
    best_j = -router.evaluate(best)[0]
    node = np.array(seq)

    improved = True
    while improved and clock() < budget:
        improved = False

        # ---- 2-opt: reverse node[i+1..k] -------------------------------
        n = len(node)
        e = chain_cost(node)                 # e[t] = cost of entering node[t]
        for i in range(-1, n - 2):
            if clock() > budget:
                break
            a = int(node[i]) if i >= 0 else -1
            ks = np.arange(i + 2, n)
            b = int(node[i + 1])
            c = node[ks]
            # rule 3.1: blocks are non-increasing, so a segment with matching
            # ends is uniform and reversing it cannot break the order
            good = block[c] == block[b]
            if not good.any():
                continue
            new_head = row(a, c)
            nxt = ks + 1
            inner = nxt < n
            new_tail = np.zeros(len(ks))
            if inner.any():
                new_tail[inner] = D[b, node[nxt[inner]]] + np.where(
                    step[node[nxt[inner]]] != step[b], Z, 0.0)
            old_tail = np.zeros(len(ks))
            old_tail[inner] = e[nxt[inner]]
            delta = new_head + new_tail - e[i + 1] - old_tail
            delta[~good] = 0.0
            k = int(np.argmin(delta))
            if delta[k] < -1e-9:
                kk = int(ks[k])
                node[i + 1:kk + 1] = node[i + 1:kk + 1][::-1]
                e = chain_cost(node)
                improved = True

        # ---- Or-opt: relocate runs of 1..3 -----------------------------
        # Every slot is scored at once with numpy.  Restricting the slots to a
        # neighbour list was measured and is slower, not faster: the per-slot
        # work is a handful of gathers either way, and a short candidate list
        # accepts fewer moves per sweep, so the sweep count rises by more than
        # the per-sweep cost falls.
        moved = True
        while moved and clock() < budget:
            moved = False
            n = len(node)
            e = chain_cost(node)
            for i in range(n):
                for ln in (1, 2, 3):
                    if i + ln > n:
                        break
                    s0, s1 = int(node[i]), int(node[i + ln - 1])
                    if block[s0] != block[s1]:
                        continue
                    prev = int(node[i - 1]) if i else -1
                    tail_in = e[i + ln] if i + ln < n else 0.0
                    close = (row(prev, node[i + ln:i + ln + 1])[0]
                             if i + ln < n else 0.0)
                    gain = e[i] + tail_in - close
                    if gain <= 1e-9:
                        continue

                    rest = np.concatenate((node[:i], node[i + ln:]))
                    m = len(rest)
                    fit = np.ones(m + 1, dtype=bool)
                    fit[1:] &= block[rest] >= block[s0]
                    fit[:m] &= block[rest] <= block[s0]
                    fit[i:i + ln] = False                    # same place
                    if not fit.any():
                        continue
                    # cutting the run out changes exactly one edge -- the one
                    # that closes over the gap -- so the rest of the vector is
                    # the old one with a hole, not a fresh O(n) rebuild
                    er = np.empty(m)
                    if m:
                        er[:i] = e[:i]
                        if i < m:
                            er[i] = close
                            er[i + 1:] = e[i + ln + 1:]
                    ins = np.empty(m + 1)
                    ins[0] = head(s0)
                    if m:
                        ins[1:] = D[rest, s0] + np.where(step[s0] != step[rest], Z, 0.0)
                    out = np.zeros(m + 1)
                    if m:
                        out[:m] = D[s1, rest] + np.where(step[rest] != step[s1], Z, 0.0)
                    drop = np.zeros(m + 1)
                    if m:
                        drop[:m] = er
                    cost = ins + out - drop
                    cost[~fit] = np.inf
                    j = int(np.argmin(cost))
                    if cost[j] < gain - 1e-9:
                        node = np.concatenate((rest[:j], node[i:i + ln], rest[j:]))
                        e = chain_cost(node)
                        moved = improved = True
                        break
                if moved:
                    break

        j = -router.evaluate(list(node))[0]
        if j < best_j:
            best, best_j = [int(v) for v in node], j

    return best


# --------------------------------------------------------------------------
# profile shaping
# --------------------------------------------------------------------------

class Shaper:
    """The scan's cost as a function of where the work sits along it.

    A part is released at its own FOV's reconstruction, so on m inspection
    cores everything from capture i onward is bounded below by

        chain_i + t_recon + (sum_{j>=i} load_j) / m

    and the cycle time is the largest such bound (plus core 0's own chain, and
    the final shot's longest part, which no core can split).  Measured against
    the exact pipeline model over all 96 boards and every feasible k, this
    relaxation has median error 0.00 s and 75th percentile 0.045 s, with
    correlation 0.9993 -- it is the exact model almost everywhere, at O(n)
    per evaluation instead of O(#parts).

    That is what makes the shaping search affordable: `route.polish` can only
    afford a move cost local to the edges it touches, which is core 0's clock
    and therefore blind to *when* the work is released.  `shape` sees the whole
    profile and can tell that dragging a heavy shot forward pays for the travel
    it costs.
    """

    def __init__(self, router, ks=None):
        self.r = router
        self.recon = router.inst.t_recon
        self.max_core = router.inst.max_core
        # Only the core counts that could plausibly win are swept.  CT(k) is
        # non-increasing and 10k grows, so k* moves by at most a step or two
        # under a local move, and sweeping all 13 possible k on a max_core=16
        # board would cost 13 full passes over the profile for nothing.
        self.ks = (tuple(sorted(ks)) if ks else
                   tuple(range(4, self.max_core + 1)))

        # the fiducial prefix is a fixed head on both the chain and the load
        pre = router.prefix
        if pre:
            c = 0.0
            chain, load = [], []
            for i, f in enumerate(pre):
                if i:
                    c += router.inst.travel(pre[i - 1]['x'], pre[i - 1]['y'], f['x'], f['y'])
                    if f['step'] != pre[i - 1]['step']:
                        c += sim.Z_DELAY
                c += f['t_img']
                chain.append(c)
                load.append(f['load'])
            self.pre_chain = np.array(chain)
            self.pre_load = np.array(load)
        else:
            self.pre_chain = np.zeros(0)
            self.pre_load = np.zeros(0)

    def floor(self, seq):
        """The best J this cover could reach at this scan length.

        Nothing can beat core 0's own chain, and nothing can beat the moment the
        first capture and its reconstruction free the cores plus all the work
        divided over them.  The gap between this and what a plan actually scores
        is the profile excess -- work stranded late in the scan -- and it is the
        only thing further search can still win, so it is what decides whether
        `ils` is worth starting on a board at all.
        """
        r = self.r
        s = np.asarray(seq)
        t = r.t_img[s].copy()
        if len(s) > 1:
            t[1:] += r.D[s[:-1], s[1:]]
            t[1:] += np.where(r.step[s[1:]] != r.step[s[:-1]], sim.Z_DELAY, 0.0)
        if r.entry is not None:
            t[0] += r.d_entry[s[0]]
            if r.step[s[0]] != r.entry_step:
                t[0] += sim.Z_DELAY
        end = float(t.sum()) + (float(self.pre_chain[-1]) if len(self.pre_chain) else 0.0)
        work = float(r.load[s].sum() + self.pre_load.sum())
        head = (float(self.pre_chain[0]) if len(self.pre_chain) else float(t[0])) + self.recon
        best = None
        for k in range(4, self.max_core + 1):
            j = max(end, head + work / (k - 3)) + sim.BETA * k
            if best is None or j < best:
                best = j
        return best

    def cost(self, seq):
        """Surrogate J of a sequence: min over k of [relaxed CT + 10k]."""
        r = self.r
        s = np.asarray(seq)
        n = len(s)
        if n == 0:
            return 0.0

        t = r.t_img[s].copy()
        if n > 1:
            t[1:] += r.D[s[:-1], s[1:]]
            t[1:] += np.where(r.step[s[1:]] != r.step[s[:-1]], sim.Z_DELAY, 0.0)
        if r.entry is not None:
            t[0] += r.d_entry[s[0]]
            if r.step[s[0]] != r.entry_step:
                t[0] += sim.Z_DELAY
        chain = np.cumsum(t)
        if len(self.pre_chain):
            chain += self.pre_chain[-1]
            chain = np.concatenate((self.pre_chain, chain))
        load = r.load[s]
        if len(self.pre_load):
            load = np.concatenate((self.pre_load, load))

        tail = np.cumsum(load[::-1])[::-1]
        end = chain[-1]
        last = r.t_comp[s[-1]]
        floor = max(end, end + self.recon + (max(last) if last else 0.0))

        chain += self.recon
        best = None
        for k in self.ks:
            ct = max(floor, float(np.max(chain + tail / (k - 3))))
            j = ct + sim.BETA * k
            if best is None or j < best:
                best = j
            elif ct <= floor:
                break            # CT has bottomed out; larger k only costs 10
        return best


def shape(router, seq, budget, clock, rng, width=6, ks=None):
    """Or-opt on the release profile rather than on core 0's clock.

    `polish` asks "does this move shorten the drive?".  This asks "does this
    move lower the score?", which also counts the work the move drags earlier or
    later along the scan.  The two disagree exactly where the slack is: a heavy
    shot taken late is work no core can start until the stage has already
    crossed the board, and buying it forward with a little extra travel is a
    trade `polish` cannot even express.

    Single-shot relocation only.  Adding 2-opt spans and 2-3 shot runs was
    measured and loses: the extra families cost more `Shaper` evaluations than
    they find, and at 1 point of runtime per second the later stages are a
    better use of the same seconds.  Candidate destinations are the `width`
    nearest shots plus the shot's own neighbours and the head of the scan, so a
    sweep is O(n * width) rather than O(n^2), rule 3.1 is checked on every
    insertion, and the sweep runs heaviest shot first -- those are the ones
    whose placement moves the profile.
    """
    n = len(seq)
    if n < 4:
        return seq
    S = Shaper(router, ks)
    block = router.block
    cur = list(seq)
    best_c = S.cost(cur)

    w = min(width, max(1, n - 1))
    nbr = np.argpartition(router.D, w, axis=1)[:, :w + 1]

    for _ in range(6):
        if clock() > budget:
            break
        gained = False
        order = sorted(range(n), key=lambda p: -router.load[cur[p]])
        at = {v: i for i, v in enumerate(cur)}
        for f in [cur[p] for p in order]:
            if clock() > budget:
                break
            p = at[f]                       # earlier moves may have shifted it
            spots = {at[int(v)] for v in nbr[f] if int(v) in at}
            spots.update((max(0, p - 2), min(n - 1, p + 1), 0))
            rest = cur[:p] + cur[p + 1:]
            hit = None
            for q in sorted(spots):
                for j in (q, q + 1):
                    if j == p or j > len(rest):
                        continue
                    if j and block[rest[j - 1]] < block[f]:
                        continue
                    if j < len(rest) and block[rest[j]] > block[f]:
                        continue
                    trial = rest[:j] + [f] + rest[j:]
                    c = S.cost(trial)
                    if c < best_c - 1e-9:
                        hit, best_c = trial, c
                        break
                if hit is not None:
                    break
            if hit is not None:
                cur = hit
                at = {v: i for i, v in enumerate(cur)}
                gained = True
        if not gained:
            break
    return cur


def _double_bridge(cur, block, rng):
    """Perturbation: a 4-opt double bridge, rejected unless it keeps rule 3.1.

    Chosen over a random restart because it moves four edges at once without
    reversing anything, so it escapes a 2-opt/Or-opt optimum while leaving the
    structure the local search built mostly intact -- the standard kick for
    iterated local search on tours.
    """
    n = len(cur)
    if n < 8:
        return None
    i, j, k = sorted(int(v) for v in rng.choice(np.arange(1, n), 3, replace=False))
    if i == j or j == k:
        return None
    out = cur[:i] + cur[j:k] + cur[i:j] + cur[k:]
    b = block[out]
    if np.any(b[:-1] < b[1:]):
        return None
    return out


def ils(router, seq, budget, clock, rng, ks=None, width=6):
    """Iterated local search over the scan order: kick, re-optimise, keep if better.

    `search` converges and then stops improving; `polish` and `shape` each run
    to their own local optimum.  Past that point the only thing left is to leave
    the basin, and measured on the boards with the largest profile excess a
    double-bridge kick followed by the same two local searches finds 3-8 points
    each that no amount of extra POMO budget reaches.

    The inner loop is judged by `Shaper`, not by the exact model: the relaxation
    agrees with the grader's pipeline to a median 0.00 s, and being 20x cheaper
    is what makes a kick-and-reoptimise cycle fit inside a runtime that is
    itself scored.  The incumbent is confirmed exactly before it is returned.
    """
    n = len(seq)
    if n < 8:
        return seq
    S = Shaper(router, ks)
    block = router.block

    cur = list(seq)
    cur_c = S.cost(cur)
    best, best_c = list(cur), cur_c

    while clock() < budget:
        kick = _double_bridge(cur, block, rng)
        if kick is None:
            continue
        kick = polish(router, kick, min(budget, clock() + 0.25 * (budget - clock())), clock)
        kick = shape(router, kick, budget, clock, rng, width=width, ks=S.ks)
        c = S.cost(kick)
        if c < cur_c - 1e-9:
            cur, cur_c = kick, c
            if c < best_c - 1e-9:
                best, best_c = list(kick), c
        else:
            cur, cur_c = list(best), best_c      # restart from the incumbent
    return best


def serpentine(router):
    """Boustrophedon seeds: sweep bands, alternating direction each band.

    A cover is close to a lattice -- shots sit roughly a FOV apart -- and on a
    lattice the shortest Hamiltonian path is a serpentine.  Handing the search
    that order up front means the POMO rollouts start from something already
    near-optimal on the regular boards and can spend their budget on the
    irregular ones, and the stall detector ends the search sooner.

    Four seeds are produced: bands across y or across x, each swept from either
    end.  Within an ordering class (rule 3.1) the step levels are kept
    contiguous and tried in both orders, so each seed also costs the minimum
    number of Z transitions that class allows (rule 4.7).
    """
    n = router.n
    if n == 0:
        return []
    out = []
    blocks = sorted(set(int(v) for v in router.block), reverse=True)
    for axis in (0, 1):
        band = router.py if axis == 0 else router.px
        run = router.px if axis == 0 else router.py
        width = max(1e-9, (band.max() - band.min()) / max(1, round(np.sqrt(n))))
        for rev in (False, True):
            for step_first in (0, 1):
                seq = []
                for b in blocks:
                    idx = np.flatnonzero(router.block == b)
                    steps = sorted(set(int(router.step[i]) for i in idx),
                                   reverse=bool(step_first))
                    for s in steps:
                        part = [int(i) for i in idx if router.step[i] == s]
                        rows = {}
                        for i in part:
                            rows.setdefault(int(band[i] // width), []).append(i)
                        for r, key in enumerate(sorted(rows, reverse=rev)):
                            rows[key].sort(key=lambda i: run[i], reverse=bool(r & 1))
                            seq.extend(rows[key])
                out.append(seq)
    return out


# Measured and dropped: forcing each ordering class's step levels contiguous.
# The scans this solver emits change Z height 161 times over the 96 boards
# against a composition minimum of 117, and a repair pass that rewrites them to
# the minimum does remove the transitions -- 12 of them over the first 39
# boards, 60 seconds of stage time -- and wins on one board by nothing at all.
# The reason is that the boards carrying the excess are inspection-bound: their
# CT is set by W/(k-3), not by core 0's chain, so stage time taken out of G
# below that level buys no score.  A 5 s penalty being large is not the same as
# it being on the critical path.
