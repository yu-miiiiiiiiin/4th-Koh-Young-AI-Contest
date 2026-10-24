"""Collect simulator feedback on TRAIN ONLY and fit a small reward predictor.

python3 -m yumin_solver_7.train
Each state evaluates feasible actions; the next state follows an improving
oracle action (80%) or an improving random action (20%). Rejected actions get
zero plan gain minus their measured evaluation cost. No hidden/test IDs enter
features. This is contextual-bandit reward regression, not long-horizon DDQN.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .search import FEATURES,from_rows,candidates,apply
from yumin_solver_4 import solver as base
from yumin_solver_5.budget_model import select_bscale


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='robustness_data/train_seed_31001')
    ap.add_argument('--limit',type=int);ap.add_argument('--reuse',action='store_true');a=ap.parse_args()
    root=Path(__file__).parent;dst=root/'training';dst.mkdir(exist_ok=True)
    rng=np.random.default_rng(707)
    if not a.reuse:
        boards=sorted(p for p in Path(a.data).iterdir() if (p/'input_component.csv').exists())
        if a.limit: boards=[boards[i] for i in np.linspace(0,len(boards)-1,a.limit,dtype=int)]
        X=[];Y=[];groups=[];costs=[]
        for bi,path in enumerate(boards):
            b=base.Board(str(path));rows,j=base.plan(str(path),_board=b,bscale=select_bscale(b));p=from_rows(b,rows)
            for step in range(3):
                actions,x=candidates(b,p)
                if not actions: break
                values=[];plans=[]
                for action in actions:
                    t=time.perf_counter();q=apply(p,action);jq=q.score()[0];elapsed=time.perf_counter()-t
                    values.append(j-jq);plans.append(q);costs.append(elapsed)
                    Y.append(max(0.,j-jq)-elapsed)
                X.extend(x);groups.extend([bi]*len(x))
                good=np.flatnonzero(np.asarray(values)>1e-8)
                if not len(good): break
                pick=int(rng.choice(good)) if rng.random()<.2 else int(np.argmax(values))
                p=plans[pick];j-=values[pick]
            if (bi+1)%8==0: print(f'train {bi+1}/{len(boards)}: {len(Y)} actions',flush=True)
        np.savez_compressed(dst/'feedback.npz',X=X,Y=Y,groups=groups,costs=costs)
        (dst/'manifest.json').write_text(json.dumps(dict(source=str(Path(a.data).resolve()),boards=[p.name for p in boards],seed=707,features=FEATURES),indent=2))
    with np.load(dst/'feedback.npz') as data:
        x,y,groups=data['X'],data['Y'],data['groups']
    from sklearn.ensemble import GradientBoostingRegressor
    model=GradientBoostingRegressor(n_estimators=32,max_depth=3,min_samples_leaf=40,learning_rate=.08,loss='squared_error',random_state=707)
    model.fit(x,y)
    trees=[]
    for estimator in model.estimators_.ravel():
        t=estimator.tree_;trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,0,0].tolist()))
    payload=dict(version=1,kind='contextual_bandit_gamma_zero',features=FEATURES,base=float(model.init_.constant_[0,0]),depth=3,rate=.08,trees=trees)
    (root/'reward_model.json').write_text(json.dumps(payload,separators=(',',':')))
    from . import model as deployed
    deployed._CACHE=None
    np.testing.assert_allclose(deployed.predict(x[:500]),model.predict(x[:500]),atol=1e-10)
    report=dict(samples=len(y),boards=len(np.unique(groups)),positive_fraction=float(np.mean(y>0)),mean_reward=float(y.mean()),fit_mae=float(np.mean(abs(model.predict(x)-y))),importances=dict(zip(FEATURES,map(float,model.feature_importances_))))
    (dst/'fit_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__': main()
