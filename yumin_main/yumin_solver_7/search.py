"""Feasible structural neighbourhood with exact acceptance and cheap ranking.

Actions reassign a component, swap two components, or merge a pair of shots.
Only affected centres are projected: unlike global midpoint relaxation, an
unaffected incumbent coordinate is never discarded. Pinned tiles/fiducials
cannot be sources or recipients. All candidates keep the type-order blocks.
"""
import json
import math
import numpy as np
from yumin_solver_4 import refine, sim

FEATURES = ['kind', 'source_pos', 'target_pos', 'release_work', 'capture_saved',
            'travel_saved', 'removed', 'source_load', 'target_load', 'moved_load',
            'source_size', 'target_size', 'stage_share', 'inspection_share',
            'recon_share', 'n_fov', 'n_jobs', 'core', 'tail_work', 'tail_pos',
            'slack_x', 'slack_y', 'critical_release_delta']


def from_rows(b, rows):
    ms = [json.loads(r['comp_idx']) for r in rows]
    return refine.Plan(b.inst, [r['x'] for r in rows], [r['y'] for r in rows],
                       [int(b.step[m[0]]) for m in ms],
                       [max(int(b.type[c]) for c in m) for m in ms],
                       ms, b.time, b.side, b.capture, b.side_capture)


def to_rows(b, p):
    return [dict(x=float(p.x[i]), y=float(p.y[i]),
                 comp_idx=json.dumps(sorted(map(int, ms), key=lambda c: -b.time[c])))
            for i, ms in enumerate(p.members)]


def box(b, members):
    return (max(b.lo_x[members]), min(b.hi_x[members]),
            max(b.lo_y[members]), min(b.hi_y[members]))


def intersection(a, b):
    return max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])


def legal(z):
    return z[0] <= z[1] and z[2] <= z[3]


def project(x, y, z):
    return min(max(x, z[0]), z[1]), min(max(y, z[2]), z[3])


def candidates(b, p, limit=240):
    """Return bounded, deterministic actions and state-action feature matrix.

    Features contain no board identifiers or scores of candidate executions.
    Reject barcode-block transitions rather than repairing with a distant sort.
    """
    n = len(p.members)
    if n < 2: return [], np.empty((0, len(FEATURES)))
    j, k = p.score(); ct = max(j - 10*k, 1e-9)
    loads = np.array([sum(b.time[m]) for m in p.members])
    img = p.t_img(); move = p.inst.moves(p.x, p.y)
    chain = np.cumsum(img + move + np.r_[0, np.diff(p.step) != 0]*sim.Z_DELAY)
    tail = np.cumsum(loads[::-1])[::-1]
    critical = int(np.argmax(chain + b.inst.t_recon + tail/(k-3)))
    eligible = [i for i, m in enumerate(p.members) if m and not any(b.big[m] | (b.type[m] == 2))]
    boxes = {i: box(b, p.members[i]) for i in eligible}
    raw = []
    seen = set()
    without = {}
    def removed(i, c):
        key=(i,c)
        if key not in without:
            ms=[v for v in p.members[i] if v!=c]
            without[key]=(ms,box(b,ms) if ms else (-np.inf,np.inf,-np.inf,np.inf))
        return without[key]
    travel=p.inst.travel
    def scalar_travel(px,py,qx,qy):
        def axis(d,v,a):
            return d/v+v/a if d>=v*v/a else 2*math.sqrt(d/a)
        return max(axis(abs(px-qx),travel.v_x,travel.a_x),axis(abs(py-qy),travel.v_y,travel.a_y))
    # Heaviest closing FOVs and small removable FOVs get early consideration.
    sources = sorted(eligible, key=lambda i: (i >= critical, i, -len(p.members[i])), reverse=True)
    span = max(n-1, 1)
    typical = max(float(loads.mean()), 1e-9)
    max_jobs = max(1, sum(map(len, p.members)))
    common = [float(chain[-1]/ct), float(loads.sum()/((k-3)*ct)),
              n*b.inst.t_recon/(2*ct), np.log1p(n), np.log1p(max_jobs), k,
              float(tail[critical]/max(loads.sum(),1e-9)), critical/span]

    def add(kind, f, t, c=-1, d=-1):
        key=(kind,f,t,c,d)
        if key in seen: return
        seen.add(key)
        fm, tm = p.members[f], p.members[t]
        if kind == 0:
            z=intersection(boxes[t], (b.lo_x[c], b.hi_x[c], b.lo_y[c], b.hi_y[c]))
            if not legal(z): return
            a,az=removed(f,c);q=tm+[c]
        elif kind == 1:
            ar,az=removed(f,c);qr,z=removed(t,d)
            az=intersection(az,(b.lo_x[d],b.hi_x[d],b.lo_y[d],b.hi_y[d]))
            z=intersection(z,(b.lo_x[c],b.hi_x[c],b.lo_y[c],b.hi_y[c]))
            if not legal(az) or not legal(z): return
            a=ar+[d];q=qr+[c]
        else:
            a=[];q=tm+fm;z=intersection(boxes[t],boxes[f])
        if not legal(z): return
        # This keeps every affected FOV in the same barcode ordering block.
        if max(b.type[q]) != p.block[t] or (a and max(b.type[a]) != p.block[f]): return
        xy={t:project(p.x[t],p.y[t],z)}
        if a: xy[f]=project(p.x[f],p.y[f],az)
        # Local edge delta; account for deletion and its new connecting edge.
        affected={e for v in (f,t) for e in (v,v+1) if 0<e<n}
        old=float(sum(move[e] for e in affected))
        def pos(i): return xy.get(i,(p.x[i],p.y[i]))
        new=0.
        for e in affected:
            if not a and (e==f or e==f+1): continue
            new+=scalar_travel(*pos(e-1),*pos(e))
        if not a and 0<f<n-1: new+=scalar_travel(*pos(f-1),*pos(f+1))
        capture_saved=float(img[f]+img[t]-b.t_img_of(q)-(b.t_img_of(a) if a else 0))
        work=float(sum(b.time[fm])) if kind==2 else float(b.time[c]-(b.time[d] if kind==1 else 0))
        release=(f-t)/span*work
        critical_delta=work*((f>=critical)-(t>=critical))/(k-3)
        feat=[kind,f/span,t/span,release,capture_saved,old-new,float(not a),loads[f]/typical,loads[t]/typical,work/typical,
              np.log1p(len(fm)),np.log1p(len(tm)),*common,(z[1]-z[0])/b.fov_w,(z[3]-z[2])/b.fov_h,critical_delta]
        raw.append(((kind,f,t,c,d,xy),feat))

    for f in sources:
        # Targets can be earlier or later; later consolidation can remove a shot.
        targets=sorted((t for t in eligible if t!=f and p.step[t]==p.step[f] and p.block[t]==p.block[f]),
                       key=lambda t: abs(p.x[f]-p.x[t])/b.fov_w+abs(p.y[f]-p.y[t])/b.fov_h)[:8]
        for t in targets:
            if f>t: add(2,f,t)
            heavy=sorted(p.members[f],key=lambda c:-b.time[c])
            # Include geometric extremes and side-sensitive components as well.
            chosen=list(dict.fromkeys(heavy[:3]+[min(heavy,key=lambda c:b.lo_x[c]),max(heavy,key=lambda c:b.hi_x[c]),
                                                 min(heavy,key=lambda c:b.lo_y[c]),max(heavy,key=lambda c:b.hi_y[c])]))
            for c in chosen: add(0,f,t,c)
            if f>t:
                for c in heavy[:2]:
                    for d in sorted(p.members[t],key=lambda d:b.time[d])[:2]: add(1,f,t,c,d)
        if len(raw)>=limit*2: break
    if not raw: return [],np.empty((0,len(FEATURES)))
    # Geometry-based prefilter before learned ranking; this can omit actions.
    raw.sort(key=lambda v: -(v[1][4]+v[1][5]+.1*v[1][3]+.25*v[1][-1]))
    raw=raw[:limit]
    return [r[0] for r in raw],np.asarray([r[1] for r in raw],float)


def apply(p, a):
    kind,f,t,c,d,xy=a
    q=p.copy()
    if kind==0:
        q.members[f].remove(c);q.members[t].append(c)
    elif kind==1:
        q.members[f].remove(c);q.members[t].remove(d)
        q.members[f].append(d);q.members[t].append(c)
    else:
        q.members[t].extend(q.members[f]);q.members[f]=[]
    for i,(x,y) in xy.items(): q.x[i]=x;q.y[i]=y
    q.compact(sum(v==2 for v in p.block))
    return q
