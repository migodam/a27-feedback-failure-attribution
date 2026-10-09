"""Charged exact backend/online acceptance tests using existing tiny fixtures."""
from pathlib import Path
import tempfile,unittest,inspect
import numpy as np
import torch
from a24.contracts import get_ledger
from a25.torch_physics import mass_pack,pack_data
from a26.encoder import FrozenEncoder
from a27.exact import ExactMaxwell
from a27.correction import correct_prediction
from test_a26_encoder_decoder import _tiny_cache

class A27ExactTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path,_,_=_tiny_cache(Path(self.temp.name))
        self.book=get_ledger()
        self.enc=FrozenEncoder(self.path,{'observable_rank':16,'confidence_relative_cut':.05,'closure':'TASK_R32'},self.book,'cpu')
        self.ref=self.enc.reference;self.exact=ExactMaxwell(self.ref)
        self.x=mass_pack(self.ref.chi_b+.1+.02j,self.ref.volume)[None]
    def tearDown(self):self.temp.cleanup()
    def test_exact_full_constitutive_and_relative_identity(self):
        s=self.exact(self.x)
        torch.testing.assert_close(s.current,s.chi[...,None]*s.field,rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(s.data,self.ref.yb+self.ref.apply_db(s.w),rtol=1e-11,atol=1e-12)
    def test_exact_real_mass_adjoint_jacobian_and_fd(self):
        s=self.exact(self.x);gen=torch.Generator().manual_seed(271010)
        d=torch.randn(self.x.shape,dtype=torch.float64,generator=gen);d/=torch.linalg.vector_norm(d)
        l=torch.randn((1,2*self.ref.q*self.ref.s),dtype=torch.float64,generator=gen)
        j=self.exact.jvp(self.x,d,s);adj=self.exact.adjoint(self.x,l,s)
        torch.testing.assert_close((j*l).sum(),(adj*d).sum(),rtol=1e-10,atol=1e-10)
        matrix=self.exact.material_jacobian(self.x,s)
        torch.testing.assert_close(j[0],matrix@d[0],rtol=1e-11,atol=1e-11)
        eps=1e-6;fd=(pack_data(self.exact(self.x+eps*d).data)-pack_data(self.exact(self.x-eps*d).data))/(2*eps)
        torch.testing.assert_close(j,fd,rtol=1e-6,atol=1e-8)
    def test_state_invalidated_and_foreign_owner_rejected(self):
        s=self.exact(self.x)
        with self.assertRaises(ValueError):self.exact.jvp(self.x+1e-6,torch.ones_like(self.x),s)
        with self.assertRaises(ValueError):ExactMaxwell(self.ref).jvp(self.x,torch.ones_like(self.x),s)
    def test_online_corrector_signature_excludes_truth_and_step_is_data_only(self):
        names=inspect.signature(correct_prediction).parameters
        self.assertNotIn('truth',names);self.assertNotIn('labels',names);self.assertNotIn('family',names)
        config={'tv_multiplier':.1,'trust_radius_mass':.15,'step_cap_background_multiple':4.,'line_trials':8,'armijo':1e-4,
                'prox_algorithm':'accelerated_pdhg','prox_max_iter':20000,'prox_tolerance':1e-6,'prox_feasibility_tolerance':1e-10,
                'prox_material_scale':1.,'prox_wall_limit_seconds':2.,'prox_check_every':10}
        obs=self.exact(mass_pack(self.ref.chi_b+.12+.025j,self.ref.volume)[None]).data
        out=correct_prediction(self.x,obs,0.,self.exact,self.exact,config,owner='tiny-test')
        meta=out['metadata']
        self.assertLessEqual(meta['data_loss'],meta['old_data_loss']+1e-12)
        self.assertLessEqual(meta['objective'],meta['old_objective']+1e-12)
        self.assertLessEqual(meta['update_mass_norm'],.15)
        self.assertTrue(meta['online_full_acceptance_audit'])
    def test_zero_gradient_does_not_invent_update(self):
        s=self.exact(self.x)
        config={'tv_multiplier':.1,'trust_radius_mass':.15,'step_cap_background_multiple':4.,'line_trials':8,'armijo':1e-4}
        out=correct_prediction(self.x,s.data,0.,self.exact,self.exact,config,owner='zero')
        self.assertFalse(out['metadata']['accepted'])
        torch.testing.assert_close(out['mass'],self.x,rtol=0,atol=0)
    def test_authorized_fgp_same_prox_objective_certificate(self):
        import numpy as np
        from a25.prox import ProxProblem,solve_prox
        cfg={'prox_max_iter':20000,'prox_tolerance':1e-6,'prox_feasibility_tolerance':1e-10,
             'prox_material_scale':1.,'prox_wall_limit_seconds':2.,'prox_check_every':10,'fgp_authorized':True}
        bg=np.zeros(4);v=np.array([.14,.07,.04,.02]);grid=(1,2);volume=.25;weight=.005
        first=solve_prox(ProxProblem(v,bg,weight,grid,volume,config={**cfg,'prox_raise_on_failure':False},algorithm='accelerated_pdhg'),budget=self.ref.book)
        second=solve_prox(ProxProblem(v,bg,weight,grid,volume,config=cfg,algorithm='fgp'),budget=self.ref.book)
        # Retain PDHG's possible iteration-cap failure; it is not relabelled
        # certified merely because its primal vector is close to optimum.
        self.assertTrue(second.converged)
        self.assertLessEqual(second.metadata['kkt_residual'],1e-6)
        self.assertLessEqual(second.metadata['normalized_gap'],1e-6)
        direction=np.array([v[0]-v[1],v[2]-v[3]]);direction/=np.linalg.norm(direction)
        closed=v+weight*np.array([-direction[0],direction[0],-direction[1],direction[1]])
        if first.converged:
            np.testing.assert_allclose(first.x,closed,atol=2e-6,rtol=2e-6)
        else:
            self.assertIn(first.metadata['stop_reason'],('ITERATION_LIMIT','WALL_LIMIT'))
        np.testing.assert_allclose(second.x,closed,atol=2e-6,rtol=2e-6)
