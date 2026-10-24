"""Learned structural improvement with feedback and exact objective checks."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from yumin_solver_4 import solver as base
from yumin_solver_5.budget_model import select_bscale
from .search import from_rows, to_rows, candidates, apply
from .model import predict, heuristic

LAST_STATS={}


def improve_rows(b, rows, *, rounds=5, shortlist=24, learned=True, seconds=.4):
    if not np.isfinite(seconds) or seconds < 0 or rounds < 0 or shortlist < 1:
        raise ValueError('seconds/rounds must be nonnegative and shortlist positive')
    started=time.perf_counter()
    p=from_rows(b,rows);initial=p.score()[0];best=initial
    stats=dict(initial_j=initial,candidates=0,evaluated=0,accepted=0,move_accepted=0,swap_accepted=0,merge_accepted=0)
    for iteration in range(rounds):
        if time.perf_counter()-started>=seconds: break
        round_start=time.perf_counter();previous=best
        actions,x=candidates(b,p)
        stats['candidates']+=len(actions)
        if not actions: break
        values=predict(x) if learned else heuristic(x)
        if learned and float(np.max(values)) <= 0: break  # STOP has reward zero
        indices=np.argsort(-values,kind='stable')[:shortlist]
        winner=None
        # Acceptance is exact; predicted reward affects order, never feasibility.
        for idx in indices:
            if time.perf_counter()-started>=seconds: break
            q=apply(p,actions[int(idx)]);j=q.score()[0];stats['evaluated']+=1
            if j<best-1e-9:
                best=j;winner=(q,int(actions[int(idx)][0]))
        if winner is None: break
        p,kind=winner
        stats['accepted']+=1
        stats[['move_accepted','swap_accepted','merge_accepted'][kind]]+=1
        if previous-best < time.perf_counter()-round_start: break
    stats.update(final_j=best,gain=initial-best,refinement_s=time.perf_counter()-started)
    return to_rows(b,p),best,stats


def plan(data_folder, *, learned=True, rounds=5, shortlist=24, seconds=.4, bscale=None):
    b=base.Board(data_folder)
    if b.n==0:
        LAST_STATS.clear();return [],0.
    scale=select_bscale(b) if bscale is None else bscale
    rows,j=base.plan(data_folder,_board=b,bscale=scale)
    rows,j,stats=improve_rows(b,rows,rounds=rounds,shortlist=shortlist,learned=learned,seconds=seconds)
    LAST_STATS.clear();LAST_STATS.update(stats)
    return rows,j


def solve(data_folder: str) -> int:
    rows,_=plan(data_folder)
    pd.DataFrame(rows,columns=['x','y','comp_idx']).to_csv(Path(data_folder)/'output_fov.csv',index=True)
    return len(rows)
