"""POMO applied to the AOI scan order.

Kwon et al., "POMO: Policy Optimization with Multiple Optima for Reinforcement
Learning" (NeurIPS 2020).  Three mechanisms are taken from the paper and used
here verbatim in structure; only the policy net is different (see NOTE).

  S4.1  Explorations from multiple starting nodes.  A scan plan is a sequence,
        and the first FOV of the permutable block is a free choice, so one plan
        has many sequence representations.  Instead of one rollout we force N
        rollouts from N *different* first FOVs, which is exactly the symmetry
        POMO exploits (their Figure 1).

  S4.2  Shared baseline.  b(s) = (1/N) * sum_j R(tau_j), the same scalar for
        every trajectory (their Eq. 4), and the REINFORCE gradient is
        (1/N) * sum_i (R(tau_i) - b) * grad log p(tau_i) (their Eq. 3).  The
        advantage has zero mean by construction, so no trajectory is judged
        against a critic that fails in the same way it does -- this is what
        makes the search resistant to the local minimum that a greedy-rollout
        baseline falls into.

  S4.3  Multiple greedy trajectories + instance augmentation.  Inference draws
        N greedy rollouts per frame over the 8 unit-square transforms of their
        Table 1, {(x,y),(y,x),(x,1-y),...}, and returns the argmax over all of
        them (their Algorithm 2).  Transposing frames also swaps (v_x,a_x) with
        (v_y,a_y) so the transformed board is a true isomorph; the reward is
        always evaluated on the original instance.

NOTE -- what is *not* from the paper.  POMO trains an Attention Model offline
over ~200 epochs on a GPU.  That is the wrong shape for this contest: solver
runtime is billed at 1 point per second (4_score.py, alpha=1), so inference
must fit in well under a second per board, and a trained checkpoint could not
be validated against the hidden test set anyway.  The policy here is instead a
softmax over a linear score w . phi(j | state) with seven hand-built features,
trained *per instance* by the POMO update above -- REINFORCE with the shared
baseline, run for as many iterations as the time budget allows.  This is POMO's
algorithm with a 7-parameter policy in place of a Transformer: the multi-start
rollouts, the shared baseline and the augmented argmax all carry over, the
learned representation does not.

The reward is the exact score the grader pays, R(tau) = -J(tau) from sim.py,
not travel distance -- so the search optimises stage travel, the 5 s Z-axis
transitions, the inspection tail and the core count k* all at once.

2-opt / Or-opt polish afterwards is not POMO; the paper (S2, "Inference
techniques") notes classical post-processing composes with it, and here it
cleans up crossings the construction policy leaves behind.
"""
import numpy as np

from . import sim

# POMO Table 1: the eight unit-square transforms, on coordinates already
# normalised to [0,1].  (u, v) is what the policy sees; the reward never does.
FRAMES = (
    (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
    (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
)   # (swap, flip_u, flip_v)

N_FEAT = 7

# Search shape.  N_START is POMO's number of starting nodes (S4.1), N_FRAME how
# many of the 8 instance-augmentation transforms are used (S4.3), LR the step
# size of the REINFORCE update (S4.2).  All three are ablatable; see README.
N_START = 8
N_FRAME = 8
LR = 0.35

# travel, step-change, sweep, row-lock, load-shaping, isolation, mass-pull.
# Starting point of the per-instance search: a nearest-neighbour policy that
# already prefers to finish a row before changing Z height.
W0 = np.array([6.475, 2.996, 0.725, 0.265, 1.556, 0.276, -0.067])


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Router:
    """Sequencing problem for one board: which FOV to shoot next, and when.

    FOVs arrive unordered from cover.py.  `block` is the ordering class the
    inspection rule pins (rule 3.1: type 2 before 1 before 0); fiducials are a
    fixed prefix and never enter the search, the rest is free to permute inside
    its block.
    """

    def __init__(self, inst, fovs, prefix):
        self.inst = inst
        self.fovs = fovs
        self.prefix = prefix               # fiducial FOVs, order fixed by rule 3.2

        # The prefix is not permutable, but it is part of the same pipeline:
        # it seeds core 0's clock and the stage sits on its last FOV when the
        # permutable block starts.  Scoring without it would hide the entry
        # travel and let the search start the scan anywhere on the board.
        self.pre_x = np.array([f['x'] for f in prefix])
        self.pre_y = np.array([f['y'] for f in prefix])
        self.pre_step = [f['step'] for f in prefix]
        self.pre_img = [f['t_img'] for f in prefix]
        self.pre_comp = [f['t_comp'] for f in prefix]

        self.entry = (prefix[-1]['x'], prefix[-1]['y']) if prefix else None
        # -1 never equals a real step level, so with no fiducial the "did the Z
        # height change?" test is simply always false before the first capture
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
            # no fiducial: the first capture happens wherever the scan opens,
            # and the grader charges no move before it
            self.d_entry = np.zeros(n)

        # feature scales, so one weight vector means the same thing on a fast
        # 1000 mm/s board and a slow 200 mm/s one
        self.d_scale = float(np.median(self.D[self.D > 0])) if np.any(self.D > 0) else 1.0
        self.l_scale = float(self.load.mean()) if n and self.load.mean() > 0 else 1.0

        # per-FOV nearest-neighbour distance: a FOV far from everything is
        # cheap to reach from its own neighbourhood and expensive later
        if n > 1:
            d = self.D + np.eye(n) * 1e18
            self.nn = d.min(axis=1) / self.d_scale
        else:
            self.nn = np.zeros(n)

        # the 8 POMO frames, precomputed (coordinates only -- cost is invariant)
        self.frames = []
        if n:
            u0 = _unit(self.px)
            v0 = _unit(self.py)
            for swap, fu, fv in FRAMES:
                u, v = (v0, u0) if swap else (u0, v0)
                self.frames.append((1.0 - u if fu else u, 1.0 - v if fv else v))
        self.max_block = int(self.block.max()) if n else 0

    # ---------------------------------------------------------------- rollout

    def rollout(self, w, frame, start, rng=None, lock=False):
        """One trajectory.  `start` fixes the first action (POMO S4.1);
        `rng=None` makes it the greedy rollout of Algorithm 2, otherwise it is
        the sampled rollout of Algorithm 1 and the log-prob gradient is
        accumulated as it goes.

        `lock` runs the step-blocked variant: the candidate set is held to the
        current Z height until that height is exhausted inside the ordering
        block.  A Z change costs 5 s against a typical 0.2 s move (rules 4.7 and
        5.2), so the unconstrained policy can be tempted into a transition that
        no amount of saved travel repays, and one greedy slip costs 25 moves'
        worth.  Locking makes that structurally impossible but can strand a far
        corner; both variants are rolled out and the exact reward picks.
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
        for t in range(n):
            blk = self.block[left].max()
            ok = left & (self.block == blk)
            if lock:
                same = ok & (self.step == cur_step)
                if same.any():
                    ok = same
            cand = np.flatnonzero(ok)

            if t == 0 and start in cand:
                pick = start
            else:
                phi = self._features(cand, at, cur_step, u, v, left, t / n)
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
        return seq, grad

    def _features(self, cand, at, cur_step, u, v, left, progress):
        """phi(j | state) for every candidate, columns as documented in W0."""
        d = (self.D[at, cand] if at >= 0 else self.d_entry[cand]) / self.d_scale
        v_at = v[at] if at >= 0 else v[cand].mean()

        cu, cv = u[left].mean(), v[left].mean()

        phi = np.empty((len(cand), N_FEAT))
        phi[:, 0] = -d
        phi[:, 1] = -(self.step[cand] != cur_step).astype(float)
        phi[:, 2] = -u[cand]
        phi[:, 3] = -np.abs(v[cand] - v_at)
        # heavy FOVs early, light ones last: whatever the final FOV carries is
        # paid after the last capture, on top of the whole scan (sim.ct_for_k)
        phi[:, 4] = (self.load[cand] / self.l_scale) * (1.0 - 2.0 * progress)
        phi[:, 5] = -self.nn[cand]
        phi[:, 6] = -(np.abs(u[cand] - cu) + np.abs(v[cand] - cv))
        return phi

    # ----------------------------------------------------------------- reward

    def evaluate(self, seq, relax=True):
        """R(tau) = -J(tau): the exact number 4_score.py charges for this board.

        Scored over the whole scan, fiducial prefix included, so the move onto
        the first permutable FOV is paid for like any other.
        """
        x = self.px[seq].copy()
        y = self.py[seq].copy()
        if relax:
            self._relax(seq, x, y)

        j, k = self.inst.evaluate(
            np.concatenate((self.pre_x, x)),
            np.concatenate((self.pre_y, y)),
            np.concatenate((self.pre_img, self.t_img[seq])),
            self.pre_comp + [self.t_comp[i] for i in seq],
            self.pre_step + [int(s) for s in self.step[seq]])
        return -j, k, x, y

    def _relax(self, seq, x, y):
        """Slide each FOV inside its feasible box toward its two neighbours.

        The box is where the FOV may legally sit (cover.py), so this is free
        travel time -- and for a fiducial the box is a point, so rule 2.3's
        centring survives it.
        """
        for _ in range(2):
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

    # ------------------------------------------------------------ POMO search

    def search(self, budget, clock, rng, n_start=None, lr=None):
        """POMO Algorithm 1 (training) interleaved with Algorithm 2 (inference).

        Each iteration draws `n_start` sampled rollouts from `n_start` distinct
        first FOVs, scores them with the true reward, shares one baseline across
        them and takes a REINFORCE step; then it draws the greedy rollouts of
        the same starts and keeps the best solution ever seen.  Frames rotate
        through the 8 transforms so the augmentation is spread over the budget
        rather than spent all at once.
        """
        n_start = N_START if n_start is None else n_start
        lr = LR if lr is None else lr
        n_frame = max(1, min(N_FRAME, len(FRAMES)))
        if self.n == 0:
            return [], 0.0
        if self.n == 1:
            r, _, _, _ = self.evaluate([0])
            return [0], -r

        starts = self._starts(n_start)
        w = W0.copy()
        best_seq, best_r, best_lock = None, -np.inf, False

        # Algorithm 2 on the untrained policy first: the greedy multi-start
        # rollouts across all 8 frames are already a strong incumbent, and
        # taking the argmax means the search can only improve on it.
        # Alternate the step-blocked variant across frames rather than running
        # both everywhere: same number of rollouts, and the two halves of the
        # augmentation sweep disagree often enough to be worth one each.
        for frame in range(n_frame):
            lock = bool(frame & 1)
            for s in starts:
                seq, _ = self.rollout(w, frame, s, lock=lock)
                r = self.evaluate(seq)[0]
                if r > best_r:
                    best_seq, best_r, best_lock = seq, r, lock
            if clock() > budget * 0.45 and frame >= 1:
                break

        frame = 0
        while clock() < budget:
            rewards = np.empty(len(starts))
            grads = []
            seqs = []
            for i, s in enumerate(starts):
                seq, g = self.rollout(w, frame, s, rng, lock=best_lock)
                rewards[i] = self.evaluate(seq)[0]
                grads.append(g)
                seqs.append(seq)

            b = rewards.mean()                              # POMO Eq. 4
            adv = rewards - b
            scale = adv.std()
            if scale > 1e-9:
                # advantages are seconds here and vary by orders of magnitude
                # between boards; normalising keeps one learning rate usable
                step = sum(a * g for a, g in zip(adv / scale, grads)) / len(starts)
                w = w + lr * step                           # POMO Eq. 3

            k = int(np.argmax(rewards))
            if rewards[k] > best_r:
                best_seq, best_r = seqs[k], rewards[k]

            for s in starts[:2]:                            # cheap greedy probe
                seq, _ = self.rollout(w, frame, s, lock=best_lock)
                r = self.evaluate(seq)[0]
                if r > best_r:
                    best_seq, best_r = seq, r
            frame = (frame + 1) % n_frame

        return best_seq, -best_r

    def _starts(self, n_start):
        """Distinct first FOVs, spread out by farthest-point sampling.

        POMO takes N different starting nodes; picking them far apart makes the
        N trajectories as heterogeneous as possible, which is what the shared
        baseline needs to stay informative.
        """
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


def _unit(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(a)


# --------------------------------------------------------------------------
# post-processing (not POMO -- see module docstring)
# --------------------------------------------------------------------------

def polish(router, seq, budget, clock):
    """2-opt and Or-opt on the core-0 timeline, then confirm with the true J.

    The move cost here is core 0's own clock -- stage travel plus the 5 s Z
    transitions (rules 5.2 and 4.7) -- because that is local to the edges a move
    touches and so evaluable in O(1).  Reversal keeps the segment's interior
    adjacencies, so only the two seam edges change.  Every accepted pass is then
    re-scored with the exact reward and rolled back if J did not actually fall.
    """
    if len(seq) < 4:
        return seq
    D, step, block = router.D, router.step, router.block
    d_entry = router.d_entry

    best = list(seq)
    best_j = -router.evaluate(best)[0]
    cur = list(seq)

    def link(a, b):
        """core-0 cost of shooting b straight after a.

        a = -1 is the entry point (the last fiducial), b = None is the end of
        the scan, where there is no edge to pay at all.
        """
        if b is None:
            return 0.0
        if a < 0:
            if router.entry is None:
                return 0.0
            return d_entry[b] + (sim.Z_DELAY if step[b] != router.entry_step else 0.0)
        return D[a, b] + (sim.Z_DELAY if step[a] != step[b] else 0.0)

    improved = True
    while improved and clock() < budget:
        improved = False
        n = len(cur)

        for i in range(-1, n - 2):              # 2-opt: reverse cur[i+1..k]
            if clock() > budget:
                break
            a = cur[i] if i >= 0 else -1
            for k in range(i + 2, n):
                b, c = cur[i + 1], cur[k]
                # block is non-increasing, so equal ends means a uniform
                # segment, and reversing one cannot break rule 3.1
                if block[b] != block[c]:
                    continue
                nxt = cur[k + 1] if k + 1 < n else None
                if link(a, c) + link(b, nxt) < link(a, b) + link(c, nxt) - 1e-9:
                    cur[i + 1:k + 1] = cur[i + 1:k + 1][::-1]
                    improved = True

        moved = True
        while moved and clock() < budget:       # Or-opt: relocate runs of 1..3
            moved = False
            n = len(cur)
            for i in range(n):
                for ln in (1, 2, 3):
                    if i + ln > n:
                        break
                    seg = cur[i:i + ln]
                    if block[seg[0]] != block[seg[-1]]:
                        continue
                    rest = cur[:i] + cur[i + ln:]
                    prev = cur[i - 1] if i else -1
                    nxt = cur[i + ln] if i + ln < n else None
                    gain = link(prev, seg[0]) + link(seg[-1], nxt) - link(prev, nxt)
                    if gain <= 1e-9:
                        continue
                    for j in range(len(rest) + 1):
                        if j >= i and j <= i + ln - 1:
                            continue                     # same place
                        if j and block[rest[j - 1]] < block[seg[0]]:
                            continue
                        if j < len(rest) and block[rest[j]] > block[seg[0]]:
                            continue
                        p = rest[j - 1] if j else -1
                        q = rest[j] if j < len(rest) else None
                        cost = link(p, seg[0]) + link(seg[-1], q) - link(p, q)
                        if cost < gain - 1e-9:
                            cur = rest[:j] + seg + rest[j:]
                            moved = improved = True
                            break
                    if moved:
                        break
                if moved:
                    break

        # core-0 cost is a proxy; only the exact score decides what is kept
        j = -router.evaluate(cur)[0]
        if j < best_j:
            best, best_j = list(cur), j

    return best
