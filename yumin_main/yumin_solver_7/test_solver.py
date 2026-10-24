"""Independent geometry/order checks, official CT parity, learned acceptance."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from yumin_solver_4.solver import Board
from yumin_solver_6.benchmark import validate
from .search import candidates,apply,from_rows,to_rows
from .solver import improve_rows,solve
from .model import predict


def fixture(root):
    comps=[];rows=[]
    def comp(x,y,t=0,side=0,step=0,w=3):
        c=len(comps)
        comps.append(dict(tl_x=x-w/2,tl_y=y-1.5,br_x=x+w/2,br_y=y+1.5,type=t,side=side,step=step,time=.2+.13*(c%7)))
        rows.append(dict(x=x,y=y,comp_idx=json.dumps([c])))
    comp(2,2,2);comp(18,2,2,step=1);comp(30,15,1)
    for i in range(24): comp(30+(i%8)*4,40+(i//8)*4,side=int(i%3==0))
    comp(45,65,step=1)
    comp(100,90,w=40)
    rows[-1]['x']=95
    rows.append(dict(x=105,y=90,comp_idx=rows[-1]['comp_idx']))
    pd.DataFrame(comps).to_csv(root/'input_component.csv',index=False)
    pd.DataFrame(dict(pcb_size=[150,150],fov_size=[30,30])).to_csv(root/'input_size.csv',index=False)
    pd.DataFrame([dict(capture_time=.5,side_capture_time=.8,recon_time=.9,max_core=8,
                       v_x=200,v_y=200,a_x=1000,a_y=1000)]).to_csv(root/'input_parameter.csv',index=False)
    return rows


class Tests(unittest.TestCase):
    def test_structural_actions_official_constraints(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as target:
            root=Path(src);rows=fixture(root);b=Board(src);p=from_rows(b,rows)
            actions,x=candidates(b,p,limit=10000)
            self.assertEqual(set(a[0] for a in actions),{0,1,2})
            self.assertTrue(np.isfinite(x).all())
            chosen=[]
            for kind in range(3): chosen.extend([a for a in actions if a[0]==kind][:4])
            original=sorted(c for ms in p.members for c in ms)
            for action in actions:
                q=apply(p,action)
                self.assertEqual(sorted(c for ms in q.members for c in ms),original)
                self.assertEqual(q.members[:2],[[0],[1]])
                for f,ms in enumerate(q.members):
                    self.assertTrue(all(b.step[c]==q.step[f] for c in ms))
                    for c in ms:
                        if b.big[c]: continue
                        self.assertTrue(b.lo_x[c]-1e-8<=q.x[f]<=b.hi_x[c]+1e-8)
                        self.assertTrue(b.lo_y[c]-1e-8<=q.y[f]<=b.hi_y[c]+1e-8)
            for action in chosen:
                q=apply(p,action)
                validate(root,to_rows(b,q),q.score()[0],Path(target))

    def test_exact_acceptance_and_interface(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as target:
            root=Path(src);rows=fixture(root);b=Board(src)
            j0=from_rows(b,rows).score()[0]
            result,j,stats=improve_rows(b,rows,seconds=.5,rounds=3)
            self.assertLessEqual(j,j0+1e-9)
            validate(root,result,j,Path(target))
            n=solve(src);out=pd.read_csv(root/'output_fov.csv')
            self.assertEqual(len(out),n)
            final=out.to_dict('records');jp=from_rows(b,final).score()[0]
            validate(root,final,jp,Path(target))

    def test_empty_and_invalid_budget(self):
        with tempfile.TemporaryDirectory() as src:
            root=Path(src);fixture(root);b=Board(src)
            r,j,_=improve_rows(b,[]);self.assertEqual(r,[]);self.assertEqual(j,0.)
            for v in (-1,float('nan')):
                with self.assertRaises(ValueError): improve_rows(b,[],seconds=v)
            self.assertEqual(predict(np.empty((0,23))).size,0)

if __name__=='__main__': unittest.main()
