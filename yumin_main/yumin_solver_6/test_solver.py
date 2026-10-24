"""Numerical certificates, scheduling parity, and special-rule regression tests.
Run: python3 -m unittest yumin_solver_6.test_solver -v
"""
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from run_utils import release_logger
from utils.checker import Checker
from yumin_solver_4 import sim, refine
from .qp import laguerre_basis, solve_qp, tighten_box
from .planner import schedule_bound, membership_bound
from .solver import solve


class SolverTests(unittest.TestCase):
    def test_laguerre_orthogonality_and_derivative(self):
        for pole in (0.0, 0.4, 0.85):
            L = laguerre_basis(512, 5, pole)
            np.testing.assert_allclose(L.T @ L, np.eye(5), atol=1e-11)
        a, eps = 0.6, 1e-6
        L = laguerre_basis(50, 5, a)
        numeric = (laguerre_basis(50, 4, a+eps)-laguerre_basis(50, 4, a-eps))/(2*eps)
        analytic = np.column_stack([(n*L[:, n] - ((n-1)*L[:, n-2] if n > 1 else 0))/(1-a*a)
                                    for n in range(1, 5)])
        np.testing.assert_allclose(numeric, analytic, atol=1e-8)

    def test_qp_against_enumerated_active_sets(self):
        rng = np.random.default_rng(6406)
        for _ in range(20):
            R = rng.normal(size=(3,3)); H = R.T@R + np.eye(3)
            q = rng.normal(size=3)*3
            A = np.vstack((np.eye(3), -np.eye(3)))
            b = rng.uniform(0.05, 1.0, 6)
            optimum = 0.0
            for count in range(4):
                for active in itertools.combinations(range(6), count):
                    C = A[list(active)]
                    K = np.block([[H,C.T],[C,np.zeros((count,count))]])
                    try: answer = np.linalg.solve(K,np.r_[-q,b[list(active)]])
                    except np.linalg.LinAlgError: continue
                    x = answer[:3]
                    if np.all(A@x <= b+1e-8) and np.all(answer[3:] >= -1e-8):
                        optimum = min(optimum,float(.5*x@H@x+q@x))
            result = solve_qp(H,q,A,b,max_sweeps=400,tolerance=1e-10)
            self.assertLessEqual(result.violation,1e-9)
            self.assertLessEqual(result.dual,optimum+1e-8)
            self.assertGreaterEqual(result.primal,optimum-1e-8)
            self.assertAlmostEqual(result.primal,optimum,places=6)

    def test_pinched_boxes(self):
        lo,hi=tighten_box(np.array([1,2,3]),np.array([1,2+1e-10,5]),.1)
        self.assertEqual(lo[0],1);self.assertEqual(hi[0],1)
        self.assertTrue(np.all(lo<=hi))
        with self.assertRaises(ValueError): tighten_box([0],[1],-1)

    def test_schedule_model_and_bounds(self):
        rng=np.random.default_rng(604)
        # Independent official simulator, including many Z transitions, recon
        # queues, imbalanced tasks, and varying axis profiles.
        checker=Checker.__new__(Checker)
        for _ in range(30):
            n=12
            xy=rng.uniform(0,300,(n,2)); steps=rng.integers(0,2,n).tolist()
            img=rng.uniform(.1,1,n); recon=float(rng.uniform(.1,3))
            times=rng.uniform(.01,5,n*3)
            members=[list(range(3*i,3*i+3)) for i in range(n)]
            inst=sim.Instance(sim.Travel(200,400,1000,2000),recon,10)
            p=refine.Plan(inst,xy[:,0],xy[:,1],steps,[0]*n,members,times,np.zeros(n*3),.5,.2)
            # Use p's constant capture times for bound tests.
            self.assertLessEqual(schedule_bound(p),p.score()[0]+1e-8)
            self.assertLessEqual(membership_bound(p),p.score()[0]+1e-8)
            costs=[]
            for k in range(4,11):
                jobs=p.t_comp()
                actual,_=checker.get_real_cost([200,400,1000,2000],xy,img,np.full(n,recon),jobs,k,steps)
                fast=sim.ct_for_k(inst.moves(xy[:,0],xy[:,1]),img,recon,jobs,steps,k)
                self.assertAlmostEqual(actual,fast,places=9)
                costs.append(actual+10*k)
            j,k=sim.score(inst.moves(xy[:,0],xy[:,1]),img,recon,p.t_comp(),steps,10)
            self.assertAlmostEqual(j,min(costs),places=9)

    def test_special_geometry_official_checker(self):
        with tempfile.TemporaryDirectory(prefix='yumin6_test_') as tmp:
            root=Path(tmp)
            # Dedicated fiducials, mixed barcode/normal, Side, two step levels,
            # and oversized Side tile coverage.
            rows=[(1,1,3,3,2,0,0),(20,2,22,4,2,0,1),
                  (30,30,34,34,1,0,0),(34,31,36,33,0,1,0),
                  (55,40,82,66,0,1,1),(10,70,14,74,0,0,1)]
            pd.DataFrame([dict(tl_x=a,tl_y=b,br_x=c,br_y=d,type=t,side=s,step=z,time=1.3)
                          for a,b,c,d,t,s,z in rows]).to_csv(root/'input_component.csv',index=False)
            pd.DataFrame(dict(pcb_size=[100,100],fov_size=[20,20])).to_csv(root/'input_size.csv',index=False)
            pd.DataFrame([dict(capture_time=.5,side_capture_time=.8,recon_time=.2,max_core=8,
                               v_x=200,v_y=200,a_x=1000,a_y=1000)]).to_csv(root/'input_parameter.csv',index=False)
            count=solve(tmp); c=Checker(tmp)
            try:
                self.assertTrue(c.cover_all_components());self.assertTrue(c.fov_inspection_order())
                out=pd.read_csv(root/'output_fov.csv');self.assertEqual(count,len(out))
                self.assertEqual(json.loads(out.comp_idx.iloc[0]),[0])
                self.assertEqual(json.loads(out.comp_idx.iloc[1]),[1])
                np.testing.assert_allclose(out[['x','y']].iloc[:2],[[2,2],[21,3]])
            finally: release_logger(c)


if __name__=='__main__': unittest.main()
