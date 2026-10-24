"""Reproducible paired benchmark, isolated outputs, no penalty repairs.

python3 -m yumin_solver_6.benchmark --data simulation_data --label practice
python3 -m yumin_solver_6.benchmark --data robustness_data/holdout_seed_51001 --label holdout
python3 -m yumin_solver_6.benchmark --data simulation_data --label ablation --ablation --limit 16
"""
import argparse
import csv
import json
import logging
from pathlib import Path
import shutil
import tempfile
import time
import numpy as np
import pandas as pd
from utils.checker import Checker
from run_utils import release_logger
from yumin_solver_4 import solver as v4
from yumin_solver_5 import solver as v5
from . import solver as v6


def validate(source, rows, j, temp):
    for old in temp.glob('input_*.csv'): old.unlink()
    for f in source.glob('input_*.csv'): shutil.copy2(f,temp/f.name)
    pd.DataFrame(rows,columns=['x','y','comp_idx']).to_csv(temp/'output_fov.csv',index=True)
    c=Checker(str(temp))
    try:
        cover,order=c.cover_all_components(),c.fov_inspection_order()
        if not cover or not order: raise AssertionError(f'invalid raw output: {source}')
        ct=c.compute_ct(); core=int(c._optimal_core)
        if abs(ct+10*core-j)>1e-6: raise AssertionError(f'score mismatch {source}: {ct+10*core} != {j}')
        # Dedicated fiducial registration order and exact centres are checked
        # separately: the official type-order check alone does not enforce all.
        fid=list(np.flatnonzero(c.component.type.to_numpy()==2))
        for i,idx in enumerate(fid):
            row=rows[i]; comp=c.component.iloc[idx]
            assert json.loads(row['comp_idx'])==[int(idx)]
            assert abs(row['x']-(comp.tl_x+comp.br_x)/2)<1e-8
            assert abs(row['y']-(comp.tl_y+comp.br_y)/2)<1e-8
        return ct,core,cover,order
    finally: release_logger(c)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='simulation_data')
    ap.add_argument('--label',default='practice')
    ap.add_argument('--limit',type=int)
    ap.add_argument('--ablation',action='store_true')
    args=ap.parse_args()
    logging.disable(logging.CRITICAL)
    boards=sorted(p for p in Path(args.data).iterdir() if (p/'input_component.csv').exists())
    if args.limit:
        # Evenly spaced selection keeps all eight scenario classes represented.
        boards=[boards[i] for i in np.linspace(0,len(boards)-1,min(args.limit,len(boards)),dtype=int)]
    output=Path(__file__).parent/'results'/args.label
    output.mkdir(parents=True,exist_ok=True)
    report=[]
    with tempfile.TemporaryDirectory(prefix='yumin6_bench_') as tmp:
        for bi,board in enumerate(boards):
            if args.ablation:
                b=v4.Board(str(board)); t=time.perf_counter()
                seed,j=v4.plan(str(board),_board=b,bscale=.8)
                seed_rt=time.perf_counter()-t
                variants=[('seed',dict(seconds=0)),('route',dict(do_qp=False)),
                          ('qp',dict(do_route=False)),('combined',{}),
                          ('fixed_pole',dict(poles=(.5,))),
                          ('full_qp',dict(basis_mode='full')),('no_margin',dict(margin=0))]
                for name,kw in variants:
                    t=time.perf_counter();rows,j,stats=v6.improve_rows(b,seed,**kw)
                    rt=seed_rt+time.perf_counter()-t
                    ct,core,cover,order=validate(board,rows,j,Path(tmp))
                    report.append(dict(dataset=board.name,solver=name,n_fov=len(rows),ct=ct,core=core,
                                       solve_s=rt,total=j+rt,cover=cover,order=order,**stats))
            else:
                # Alternate order to reduce consistent cache/thermal advantage.
                variants=[('yumin_solver_5',v5),('yumin_solver_6',v6)]
                if bi%2: variants.reverse()
                for name,module in variants:
                    t=time.perf_counter();rows,j=module.plan(str(board));rt=time.perf_counter()-t
                    ct,core,cover,order=validate(board,rows,j,Path(tmp))
                    dest=output/board.name/name;dest.mkdir(parents=True,exist_ok=True)
                    pd.DataFrame(rows).to_csv(dest/'output_fov.csv',index=True)
                    report.append(dict(dataset=board.name,solver=name,n_fov=len(rows),ct=ct,core=core,
                                       solve_s=rt,total=j+rt,cover=cover,order=order,
                                       **(dict(v6.LAST_STATS) if module is v6 else {})))
            if (bi+1)%8==0: print(f'{args.label}: {bi+1}/{len(boards)} validated',flush=True)
    fields=list(dict.fromkeys(k for row in report for k in row))
    with (output/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(report)
    frame=pd.DataFrame(report)
    totals=frame.groupby('solver')[['ct','core','solve_s','total']].sum()
    totals.to_csv(output/'totals.csv');print(totals.to_string())
    for name,group in frame.groupby('solver'):
        group[['dataset','n_fov','ct','core','solve_s','cover','order']].to_csv(output/f'summary_{name}.csv',index=False)
    print(f'Saved {output}',flush=True)


if __name__=='__main__': main()
