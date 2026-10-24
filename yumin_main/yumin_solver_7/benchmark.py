"""Sequential matched benchmark with official raw-output validation.
No model fitting occurs here. Runtime includes input parsing and output CSV.
"""
import argparse
import csv
import logging
import hashlib
import json
import platform
from pathlib import Path
import tempfile
import time
import numpy as np
import pandas as pd
from yumin_solver_6.benchmark import validate
from yumin_solver_4 import solver as v4
from yumin_solver_5 import solver as v5
from yumin_solver_6 import solver as v6
from yumin_solver_5.budget_model import select_bscale
from . import solver as v7


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',default='simulation_data');ap.add_argument('--label',default='practice')
    ap.add_argument('--limit',type=int);ap.add_argument('--ablation',action='store_true');ap.add_argument('--repeat',type=int,default=1)
    args=ap.parse_args();logging.disable(logging.CRITICAL)
    boards=sorted(p for p in Path(args.data).iterdir() if (p/'input_component.csv').exists())
    if args.limit: boards=[boards[i] for i in np.linspace(0,len(boards)-1,min(args.limit,len(boards)),dtype=int)]
    output=Path(__file__).parent/'results'/args.label;output.mkdir(parents=True,exist_ok=True);report=[]
    files=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).with_name('reward_model.json')]
    manifest=dict(data=str(Path(args.data).resolve()),boards=[p.name for p in boards],
                  python=platform.python_version(),platform=platform.platform(),args=vars(args),
                  sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with tempfile.TemporaryDirectory(prefix='yumin7_bench_') as tmp:
        for rep in range(args.repeat):
            for bi,board in enumerate(boards):
                if args.ablation:
                    t=time.perf_counter();b=v4.Board(str(board));seed,j=v4.plan(str(board),_board=b,bscale=select_bscale(b));seed_rt=time.perf_counter()-t
                    variants=[('seed',dict(rounds=0)),('heuristic',dict(learned=False,rounds=3,shortlist=16,seconds=.18)),('learned',dict(rounds=3,shortlist=16,seconds=.18)),
                              ('learned_short',dict(rounds=1,shortlist=8)),('learned_long',dict(rounds=5,shortlist=24,seconds=.4))]
                else:
                    variants=[('yumin_solver_5',v5),('yumin_solver_6',v6),('yumin_solver_7',v7)]
                shift=(bi+rep)%len(variants);variants=variants[shift:]+variants[:shift]
                for name,mod in variants:
                    t=time.perf_counter()
                    if args.ablation: rows,j,stats=v7.improve_rows(b,seed,**mod)
                    else:
                        rows,j=mod.plan(str(board));stats=dict(mod.LAST_STATS) if mod in (v6,v7) else {}
                    dest=output/f'repeat_{rep}'/board.name/name;dest.mkdir(parents=True,exist_ok=True)
                    pd.DataFrame(rows,columns=['x','y','comp_idx']).to_csv(dest/'output_fov.csv',index=True)
                    rt=time.perf_counter()-t+(seed_rt if args.ablation else 0)
                    ct,core,cover,order=validate(board,rows,j,Path(tmp))
                    report.append(dict(repeat=rep,dataset=board.name,solver=name,n_fov=len(rows),ct=ct,core=core,
                                       solve_s=rt,total=j+rt,cover=cover,order=order,**stats))
                if (bi+1)%8==0: print(f'{args.label} rep{rep}: {bi+1}/{len(boards)} valid',flush=True)
    fields=list(dict.fromkeys(k for row in report for k in row))
    with (output/'comparison.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(report)
    df=pd.DataFrame(report);totals=df.groupby(['repeat','solver'])[['ct','core','solve_s','total']].sum();totals.to_csv(output/'totals.csv');print(totals.to_string(),flush=True)
    for (rep,name),g in df.groupby(['repeat','solver']):
        g[['dataset','n_fov','ct','core','solve_s','cover','order']].to_csv(output/f'summary_{name}_repeat{rep}.csv',index=False)

if __name__=='__main__': main()
