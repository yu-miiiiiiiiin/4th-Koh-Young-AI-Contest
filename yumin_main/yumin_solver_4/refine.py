"""Move parts between shots once the scan order is known.

Cover and route are separate stages, and the seam between them is where score
is left behind.  `cover.py` decides *which* shots exist by minimising their
number; it has no idea when each one will be taken.  `route.py` then fixes the
order but may only permute whole shots -- it cannot move a single part out of
the shot the cover put it in.

That matters because of how the grader drains the inspection queue.  A part is
released at its own FOV's reconstruction, so with `m = k-3` inspection cores

    CT  >=  max_i [ chain_i + t_recon + (sum_{j>=i} load_j) / m ]

where chain_i is core 0's clock at capture i.  Work sitting in a *late* shot is
work no core can start until the stage has already driven the whole board.  The
same part in an *early* shot is free: it is chewed through while the stage is
still moving.  Nothing about the cover is wrong -- the part simply fits both
shots, and only the order says which one is cheaper.

So: with positions and order fixed, walk the parts and hand each one to the
earliest shot that can legally hold it.  Two further things fall out of it.

  * A shot can empty completely, and an empty shot is deleted -- one capture,
    one reconstruction and one stage move gone.
  * Rule 4.6 charges a shot max(capture, side_capture) if it holds even one
    side=1 part.  Moving the last side part out of a shot makes it cheap again;
    moving one *in* makes a cheap shot dear.  The walk accounts for both, and
    the exact score decides.

Every candidate plan is scored with `sim` and kept only if J actually fell.
"""
import numpy as np

from . import sim

TOL = 1e-5            # the grader's own containment tolerance


def feasible_matrix(fx, fy, f_step, tl_x, tl_y, br_x, br_y, ew, eh, c_step, movable):
    """`ok[c, f]` -- may part c legally sit in shot f as the shot now stands?

    Rule 2.2 for side=0 and rule 2.6 for side=1 differ only in the window, which
    `ew`/`eh` already carry per part.  Rule 2.7 forbids mixing step levels.
    Oversized parts (rule 2.5) and fiducials (rule 2.3) are not movable and are
    excluded by the caller through `movable`.
    """
    c = np.flatnonzero(movable)
    hw = ew[c] / 2.0
    hh = eh[c] / 2.0
    ok = ((fx[None, :] - hw[:, None] <= tl_x[c][:, None] + TOL) &
          (fx[None, :] + hw[:, None] >= br_x[c][:, None] - TOL) &
          (fy[None, :] - hh[:, None] <= tl_y[c][:, None] + TOL) &
          (fy[None, :] + hh[:, None] >= br_y[c][:, None] - TOL) &
          (c_step[c][:, None] == f_step[None, :]))
    # The row of shots a part may sit in is asked for once per part per pass and
    # never changes, so it is materialised here rather than re-scanned: doing it
    # lazily was half a million `flatnonzero` calls over the practice set.
    feas = [np.flatnonzero(r) for r in ok]
    return c, ok, feas


class Plan:
    """An ordered scan: shot positions, their parts, and the score they cost."""

    def __init__(self, inst, xs, ys, steps, blocks, members, c_time, c_side,
                 capture, side_capture):
        self.inst = inst
        self.x = np.asarray(xs, float)
        self.y = np.asarray(ys, float)
        self.step = list(steps)
        self.block = list(blocks)
        self.members = [list(m) for m in members]
        self.c_time = c_time
        self.c_side = c_side
        self.capture = capture
        self.side_capture = max(capture, side_capture)

    def t_img(self):
        return np.array([self.side_capture if any(self.c_side[i] for i in ms)
                         else self.capture for ms in self.members])

    def t_comp(self):
        # LPT inside a shot: the grader list-schedules a shot's parts onto cores
        # 3.. in exactly the order comp_idx gives, so longest-first is a free
        # cut on the tail that sets CT.
        return [sorted((self.c_time[i] for i in ms), reverse=True)
                for ms in self.members]

    def score(self):
        if len(self.x) == 0:
            return 0.0, 4
        return self.inst.evaluate(self.x, self.y, self.t_img(), self.t_comp(), self.step)

    def copy(self):
        p = Plan.__new__(Plan)
        p.__dict__.update(self.__dict__)
        p.x, p.y = self.x.copy(), self.y.copy()
        p.step, p.block = list(self.step), list(self.block)
        p.members = [list(m) for m in self.members]
        return p

    def compact(self, keep_first):
        """Delete empty shots, preserving the fiducial prefix."""
        keep = [i for i in range(len(self.members))
                if i < keep_first or self.members[i]]
        if len(keep) == len(self.members):
            return False
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.step = [self.step[i] for i in keep]
        self.block = [self.block[i] for i in keep]
        self.members = [self.members[i] for i in keep]
        return True


def _shot_box(plan, f, lo_x, hi_x, lo_y, hi_y, pinned_comp):
    """Feasible centre box of shot ``f`` under its current membership.

    Fiducials and oversized-component tiles are position-pinned by the rules.
    Every other shot may move anywhere in the intersection of its members'
    feasible-centre boxes.  This is deliberately membership based rather than
    based on the shot's current point: the latter misses legal reassignments
    that only require sliding the camera a little.
    """
    ms = plan.members[f]
    if not ms or any(pinned_comp[i] for i in ms):
        x, y = float(plan.x[f]), float(plan.y[f])
        return x, x, y, y, True
    return (max(float(lo_x[i]) for i in ms),
            min(float(hi_x[i]) for i in ms),
            max(float(lo_y[i]) for i in ms),
            min(float(hi_y[i]) for i in ms), False)


def shot_boxes(plan, lo_x, hi_x, lo_y, hi_y, pinned_comp):
    """Return the current feasible box and pin flag of every shot."""
    out = [_shot_box(plan, f, lo_x, hi_x, lo_y, hi_y, pinned_comp)
           for f in range(len(plan.members))]
    return tuple(np.array([v[i] for v in out], dtype=float if i < 4 else bool)
                 for i in range(5))


def _can_add(c, f, boxes, lo_x, hi_x, lo_y, hi_y, c_step, plan):
    """Whether component ``c`` can join shot ``f`` after moving that shot."""
    if plan.step[f] != int(c_step[c]):
        return False
    lx, hx, ly, hy, _ = (a[f] for a in boxes)
    return (max(lx, lo_x[c]) <= min(hx, hi_x[c]) + TOL and
            max(ly, lo_y[c]) <= min(hy, hi_y[c]) + TOL)


def _targets(c, source, boxes, lo_x, hi_x, lo_y, hi_y,
             c_step, plan, keep_first):
    """Vectorised list of shots whose feasible box can absorb component c."""
    bx0, bx1, by0, by1, _ = boxes
    step = np.asarray(plan.step)
    good = ((step == int(c_step[c])) &
            (np.maximum(bx0, lo_x[c]) <= np.minimum(bx1, hi_x[c]) + TOL) &
            (np.maximum(by0, lo_y[c]) <= np.minimum(by1, hi_y[c]) + TOL))
    good[:keep_first] = False
    if 0 <= source < len(good):
        good[source] = False
    return np.flatnonzero(good)


def relax_positions(plan, lo_x, hi_x, lo_y, hi_y, pinned_comp,
                    keep_first, passes=2):
    """Slide movable shots inside their feasible boxes toward their neighbours.

    This is the refine-side counterpart of ``route.relax``.  Membership changes
    alter a shot's box, so reusing the old centre leaves travel savings on the
    table.  The exact score still decides whether the enclosing candidate plan
    is retained.
    """
    if len(plan.members) < 2:
        return
    bx0, bx1, by0, by1, pin = shot_boxes(
        plan, lo_x, hi_x, lo_y, hi_y, pinned_comp)
    n = len(plan.members)
    for _ in range(passes):
        for f in range(keep_first, n):
            if pin[f] or not plan.members[f]:
                continue
            if f:
                ax, ay = plan.x[f - 1], plan.y[f - 1]
            else:
                ax, ay = plan.x[f], plan.y[f]
            if f + 1 < n:
                cx, cy = plan.x[f + 1], plan.y[f + 1]
            else:
                cx, cy = ax, ay
            plan.x[f] = min(max((ax + cx) / 2.0, bx0[f]), bx1[f])
            plan.y[f] = min(max((ay + cy) / 2.0, by0[f]), by1[f])


def refresh_blocks(plan, c_type):
    for f, ms in enumerate(plan.members):
        plan.block[f] = max((int(c_type[i]) for i in ms), default=0)


def profile_floor(plan):
    """Cheap lower bound for this fixed membership and scan length."""
    if not plan.members:
        return 40.0
    t_img = plan.t_img()
    move = plan.inst.moves(plan.x, plan.y)
    chain = float(np.sum(t_img) + np.sum(move[1:]))
    chain += sim.Z_DELAY * sum(plan.step[i] != plan.step[i - 1]
                               for i in range(1, len(plan.step)))
    head = float(t_img[0]) + plan.inst.t_recon
    work = float(sum(plan.c_time[c] for ms in plan.members for c in ms))
    return min(max(chain, head + work / (k - 3)) + sim.BETA * k
               for k in range(4, plan.inst.max_core + 1))


def profile_excess(plan):
    return plan.score()[0] - profile_floor(plan)


def resort(plan, keep_first):
    """Stable sort the permutable tail by block, descending (rule 3.1).

    Stability matters: within a block the route's order is the optimised one and
    must survive.
    """
    tail = list(range(keep_first, len(plan.members)))
    tail.sort(key=lambda i: -plan.block[i])
    order = list(range(keep_first)) + tail
    if order == list(range(len(plan.members))):
        return
    plan.x = plan.x[order]
    plan.y = plan.y[order]
    plan.step = [plan.step[i] for i in order]
    plan.block = [plan.block[i] for i in order]
    plan.members = [plan.members[i] for i in order]


def lighten_tail(plan, feas, cols, keep_first, budget, clock):
    """Drag the heaviest parts out of the closing shots.

    Whatever the final shot holds is paid after the last capture, on top of the
    entire scan, and at best it is spread over m cores -- so the makespan never
    beats chain_last + t_recon + max_part_time_of_that_shot.  Emptying the
    biggest part out of the closing shots attacks exactly that term.  Tried
    heaviest-first and confirmed against the exact score.
    """
    best_j = plan.score()[0]
    n_f = len(plan.members)
    at_row = {int(c): r for r, c in enumerate(cols)}

    for f in range(n_f - 1, max(keep_first, n_f - 12) - 1, -1):
        if clock() > budget:
            break
        for i in sorted(plan.members[f], key=lambda i: -plan.c_time[i])[:3]:
            r = at_row.get(int(i))
            if r is None:
                continue
            w = feas[r]
            w = w[(w < f) & (w >= keep_first)]      # rule 2.3
            if w.size == 0:
                continue
            tgt = int(w[np.argmin([len(plan.members[t]) for t in w])])
            plan.members[f].remove(i)
            plan.members[tgt].append(i)
            j = plan.score()[0]
            if j < best_j - 1e-9:
                best_j = j
            else:
                plan.members[tgt].remove(i)
                plan.members[f].append(i)
    return best_j


def dissolve_shots(plan, ok, cols, c_type, keep_first, budget, clock, order=None):
    """Try to empty a shot completely and delete it.

    `cover._absorb` asks the same question before the scan is ordered, against
    each FOV's *feasible box*.  By the time the route has run, every shot has
    been slid to a definite point inside that box (`route.relax`), so the
    question has a different answer: a part that no box-centred shot could take
    may well sit inside a neighbour as it now stands, and vice versa.  Asking
    again here is what turns the leftovers into deletions.

    Deleting a shot removes a capture, a reconstruction and a stage move from
    core 0's chain at once -- the one move in this file that shortens G rather
    than just moving work along it.  Cheapest shots first, since those are the
    ones most likely to find homes for everything they hold.
    """
    row = {int(c): r for r, c in enumerate(cols)}
    best_j = plan.score()[0]
    killed = 0

    # Deleting a shot renumbers every shot after it, and `ok`'s columns are
    # renumbered with it, so the queue has to be renumbered too -- otherwise a
    # later entry names a different shot than the one it was queued for.
    cand = list(order) if order is not None else sorted(
        range(keep_first, len(plan.members)), key=lambda f: len(plan.members[f]))
    while cand:
        if clock() > budget:
            break
        f = cand.pop(0)
        if f >= len(plan.members) or not plan.members[f]:
            continue
        home = []
        for i in plan.members[f]:
            r = row.get(int(i))
            if r is None:
                home = None
                break
            w = np.flatnonzero(ok[r])
            # rule 2.3: a fiducial's shot takes no other part, ever
            w = w[(w != f) & (w >= keep_first)]
            if w.size == 0:
                home = None
                break
            # the emptiest shot that can take it, so loads stay spread
            home.append(int(w[np.argmin([len(plan.members[t]) for t in w])]))
        if home is None:
            continue

        moved = list(zip(plan.members[f], home))
        for i, t in moved:
            plan.members[t].append(i)
        kept = plan.members[f]
        plan.members[f] = []
        trial = plan.copy()
        trial.compact(keep_first)
        j = trial.score()[0]
        if j < best_j - 1e-9:
            best_j = j
            plan.__dict__.update(trial.__dict__)
            # positions shifted by one: rebuild the row map against the new plan
            ok = np.delete(ok, f, axis=1)
            cand = [c - 1 if c > f else c for c in cand]
            killed += 1
        else:
            for i, t in moved:
                plan.members[t].remove(i)
            plan.members[f] = kept
    return best_j, killed


def assign(plan, feas, cols, c_type, keep_first, w_pos=1.0, w_load=0.0, passes=2):
    """Re-home every movable part by a weighted cost over the shots that fit it.

        cost(shot f for part c) = w_pos * (position of f along the scan)
                                + w_load * (load f already carries)

    The two terms pull against each other and both are real:

      * `w_pos` buys *release time*.  Work in shot f cannot start before the
        stage has reached f, so the profile bound chain_i + W_{>=i}/m falls when
        work moves toward the head of the scan.
      * `w_load` buys *occupancy*.  The grader drags every core up to core 0's
        clock after each capture, so a core idle when the next capture lands has
        that idle time deleted, not banked.  A shot holding fewer parts than
        there are cores wastes capacity that no later shot can reclaim, and
        spreading work over shots is what keeps the cores fed.

    w_load = 0 recovers `pull_forward`.  Which mixture wins is a property of the
    board -- a scan whose cores are starved wants spreading, one whose cores are
    backlogged wants everything as early as it will go -- so the caller tries
    several and keeps whichever the exact score prefers.
    """
    n_f = len(plan.members)
    load = np.array([sum(plan.c_time[i] for i in ms) for ms in plan.members])
    pos = np.arange(n_f, dtype=float)
    span = max(1.0, float(n_f - 1))
    scale = max(1e-9, float(load.mean()) if n_f else 1.0)

    rows = {int(c): r for r, c in enumerate(cols)}
    changed = False
    for _ in range(passes):
        moved = False
        for f in range(n_f - 1, keep_first - 1, -1):
            for i in list(plan.members[f]):
                r = rows.get(int(i))
                if r is None:
                    continue
                w = feas[r]
                w = w[w >= keep_first]
                if w.size < 2:
                    continue
                cost = w_pos * (pos[w] / span) + w_load * (load[w] / scale)
                tgt = int(w[int(np.argmin(cost))])
                if tgt == f:
                    continue
                # a lone side part carries rule 4.6's surcharge with it: do not
                # make a cheap shot dear unless its own shot loses its last one
                if plan.c_side[i] and not any(plan.c_side[j] for j in plan.members[tgt]):
                    if any(plan.c_side[j] for j in plan.members[f] if j != i):
                        continue
                plan.members[f].remove(i)
                plan.members[tgt].append(i)
                load[f] -= plan.c_time[i]
                load[tgt] += plan.c_time[i]
                moved = changed = True
        if not moved:
            break

    if changed:
        for f in range(n_f):
            plan.block[f] = max((int(c_type[i]) for i in plan.members[f]), default=0)
    return changed


def assign_flexible(plan, lo_x, hi_x, lo_y, hi_y, c_step, c_type,
                    movable, pinned_comp, keep_first, w_pos=1.0,
                    w_load=0.0, w_shift=0.15, passes=2):
    """Re-home parts while allowing every non-pinned FOV to slide.

    ``assign`` only considers shots that contain a part at their current centre.
    Here a target is legal whenever the intersection of its members' feasible
    boxes remains non-empty after adding the part.  A small shift term breaks
    ties in favour of assignments that preserve the routed geometry.
    """
    n_f = len(plan.members)
    if n_f == 0:
        return False
    load = np.array([sum(plan.c_time[i] for i in ms) for ms in plan.members], float)
    pos = np.arange(n_f, dtype=float)
    span = max(1.0, float(n_f - 1))
    scale = max(1e-9, float(load.mean()))
    changed = False

    for _ in range(passes):
        moved = False
        boxes = shot_boxes(plan, lo_x, hi_x, lo_y, hi_y, pinned_comp)
        bx0, bx1, by0, by1, _ = boxes
        for f in range(n_f - 1, keep_first - 1, -1):
            for c in list(plan.members[f]):
                if not movable[c]:
                    continue
                w = _targets(c, f, boxes, lo_x, hi_x, lo_y, hi_y,
                             c_step, plan, keep_first)
                if w.size == 0:
                    continue
                # Minimum displacement needed to enter the new intersection.
                nx0 = np.maximum(bx0[w], lo_x[c])
                nx1 = np.minimum(bx1[w], hi_x[c])
                ny0 = np.maximum(by0[w], lo_y[c])
                ny1 = np.minimum(by1[w], hi_y[c])
                dx = np.maximum(nx0 - plan.x[w], 0.0) + np.maximum(plan.x[w] - nx1, 0.0)
                dy = np.maximum(ny0 - plan.y[w], 0.0) + np.maximum(plan.y[w] - ny1, 0.0)
                shift_scale = max(1.0, float(np.ptp(plan.x) + np.ptp(plan.y)))
                cost = (w_pos * pos[w] / span + w_load * load[w] / scale +
                        w_shift * (dx + dy) / shift_scale)
                t = int(w[int(np.argmin(cost))])

                # Moving a side part into an otherwise top-only FOV costs the
                # capture surcharge.  Allow it only when the source loses its
                # final side part; exact scoring later handles the close calls.
                if plan.c_side[c] and not any(plan.c_side[i] for i in plan.members[t]):
                    if any(plan.c_side[i] for i in plan.members[f] if i != c):
                        continue
                plan.members[f].remove(c)
                plan.members[t].append(c)
                load[f] -= plan.c_time[c]
                load[t] += plan.c_time[c]
                # Membership changed both intersections; refresh before the
                # next component so every subsequent move remains legal.
                for q in (f, t):
                    a, b, d, e, p = _shot_box(
                        plan, q, lo_x, hi_x, lo_y, hi_y, pinned_comp)
                    bx0[q], bx1[q], by0[q], by1[q] = a, b, d, e
                moved = changed = True
        if not moved:
            break

    if changed:
        refresh_blocks(plan, c_type)
        relax_positions(plan, lo_x, hi_x, lo_y, hi_y, pinned_comp, keep_first)
    return changed


def dissolve_flexible(plan, lo_x, hi_x, lo_y, hi_y, c_step, c_type,
                      movable, pinned_comp, keep_first, budget, clock,
                      initial_j=None):
    """Delete shots using constraint-first, multi-strategy reassignment.

    Several complete assignments are tried because independently sending every
    component to the emptiest shot can consume the only home of a constrained
    component.  Each accepted deletion is verified by the exact objective.
    """
    best_j = plan.score()[0] if initial_j is None else initial_j
    killed = 0
    while clock() < budget:
        accepted = False
        candidates = sorted(
            (f for f in range(keep_first, len(plan.members))
             if plan.members[f] and all(movable[c] for c in plan.members[f])),
            key=lambda f: (len(plan.members[f]),
                           sum(plan.c_time[c] for c in plan.members[f])))
        for f in candidates:
            if clock() >= budget:
                break
            base_boxes = shot_boxes(plan, lo_x, hi_x, lo_y, hi_y, pinned_comp)
            # Most constrained and then heaviest first.
            home_cache = {
                c: _targets(c, f, base_boxes, lo_x, hi_x, lo_y, hi_y,
                            c_step, plan, keep_first)
                for c in plan.members[f]
            }
            order = sorted(plan.members[f],
                           key=lambda c: (len(home_cache[c]), -plan.c_time[c]))
            if any(len(home_cache[c]) == 0 for c in order):
                continue

            for mode in range(3):
                if clock() >= budget:
                    break
                trial = plan.copy()
                trial.members[f] = []
                ok = True
                for c in order:
                    boxes = shot_boxes(trial, lo_x, hi_x, lo_y, hi_y, pinned_comp)
                    opts = _targets(c, f, boxes, lo_x, hi_x, lo_y, hi_y,
                                    c_step, trial, keep_first)
                    if opts.size == 0:
                        ok = False
                        break
                    loads = [sum(trial.c_time[i] for i in trial.members[int(t)]) for t in opts]
                    if mode == 0:       # release work as early as possible
                        t = int(opts.min())
                    elif mode == 1:     # preserve inspection-core occupancy
                        t = int(opts[int(np.argmin(loads))])
                    else:               # preserve local scan geometry
                        t = int(opts[int(np.argmin(np.abs(opts - f)))])
                    trial.members[t].append(c)
                if not ok:
                    continue
                refresh_blocks(trial, c_type)
                trial.compact(keep_first)
                resort(trial, keep_first)
                relax_positions(trial, lo_x, hi_x, lo_y, hi_y,
                                pinned_comp, keep_first)
                j = trial.score()[0]
                if j < best_j - 1e-9:
                    plan.__dict__.update(trial.__dict__)
                    best_j, killed, accepted = j, killed + 1, True
                    break
            if accepted:
                break
        if not accepted:
            break
    return best_j, killed


def lighten_critical(plan, lo_x, hi_x, lo_y, hi_y, c_step, c_type,
                     movable, pinned_comp, keep_first, budget, clock,
                     max_moves=10, initial_j=None):
    """Exact best-improvement moves from the critical closing region.

    The window scales with the route instead of being fixed at twelve shots.
    For each heavy/side-sensitive part the earliest, lightest and nearest legal
    targets are checked.  FOV centres and ordering classes are repaired before
    exact scoring, so every committed move is both legal and non-worsening.
    """
    best_j = plan.score()[0] if initial_j is None else initial_j
    moves = 0
    while moves < max_moves and clock() < budget:
        n = len(plan.members)
        start = max(keep_first, n - min(30, max(12, int(np.ceil(0.25 * n)))))
        boxes = shot_boxes(plan, lo_x, hi_x, lo_y, hi_y, pinned_comp)
        choice = None
        for f in range(n - 1, start - 1, -1):
            if clock() >= budget:
                break
            ranked = sorted((c for c in plan.members[f] if movable[c]),
                            key=lambda c: (not plan.c_side[c], -plan.c_time[c]))[:6]
            for c in ranked:
                opts = _targets(c, f, boxes, lo_x, hi_x, lo_y, hi_y,
                                c_step, plan, keep_first)
                opts = opts[opts < f]
                if opts.size == 0:
                    continue
                loads = np.array([sum(plan.c_time[i] for i in plan.members[t])
                                  for t in opts])
                shortlist = {int(opts[0]), int(opts[int(np.argmin(loads))]),
                             int(opts[int(np.argmin(np.abs(opts - f)))])}
                for t in shortlist:
                    if clock() >= budget:
                        break
                    trial = plan.copy()
                    trial.members[f].remove(c)
                    trial.members[t].append(c)
                    refresh_blocks(trial, c_type)
                    trial.compact(keep_first)
                    resort(trial, keep_first)
                    relax_positions(trial, lo_x, hi_x, lo_y, hi_y,
                                    pinned_comp, keep_first)
                    j = trial.score()[0]
                    if j < best_j - 1e-9 and (choice is None or j < choice[0]):
                        choice = (j, trial)
        if choice is None:
            break
        best_j, winner = choice
        plan.__dict__.update(winner.__dict__)
        moves += 1
    return best_j
