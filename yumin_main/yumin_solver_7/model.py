"""Offline contextual-bandit reward predictor; NumPy-only deployment.

This is a one-step reward model (gamma=0), not a reproduction of DDQN.
Tree fitting is offline; the submitted solver never imports sklearn.
"""
from pathlib import Path
import json
import numpy as np

_CACHE = None


def predict(x):
    global _CACHE
    if not len(x): return np.empty(0)
    if _CACHE is None:
        path=Path(__file__).with_name('reward_model.json')
        if not path.exists():
            return heuristic(x)
        _CACHE=json.loads(path.read_text())
    out=np.full(len(x),_CACHE['base'],float)
    xf=x.astype(np.float32)  # match sklearn tree comparison precision
    for tr in _CACHE['trees']:
        left,right,feature,threshold,value=[np.asarray(tr[k]) for k in ('left','right','feature','threshold','value')]
        node=np.zeros(len(x),dtype=int)
        for _ in range(_CACHE['depth']+1):
            active=left[node]>=0
            if not np.any(active): break
            ii=np.flatnonzero(active);nn=node[ii]
            node[ii]=np.where(xf[ii,feature[nn]]<=threshold[nn],left[nn],right[nn])
        out+=_CACHE['rate']*value[node]
    return out


def heuristic(x):
    return x[:,4]+x[:,5]+.1*x[:,3]+.25*x[:,-1]
