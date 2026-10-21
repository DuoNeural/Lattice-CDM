"""Independent malformed-checkpoint and frozen-inference regression tests.
Run: python -m unittest -v test_redteam
"""
import ast
import math
import tempfile
import unittest
from pathlib import Path
import torch
import cdm_tools as cdm


def fixture(tangent=False):
    g = torch.Generator().manual_seed(1994)
    def rand(*shape): return torch.randn(*shape, generator=g) * .1
    d, k, e, ff = 8, 4, 2, 12
    s = {'tok_embed.weight': rand(19, d), 'pos_embed.weight': rand(10, d)}
    s['lm_head.weight'] = s['tok_embed.weight'].clone()
    for name in ['embed_norm', 'final_norm']:
        s[name + '.weight'] = torch.ones(d)
        s[name + '.bias'] = torch.zeros(d)
    for i in range(2):
        p = f'layers.{i}.'
        for name in ['attn_norm', 'ffn_norm']:
            s[p+name+'.weight'] = torch.ones(d)
            s[p+name+'.bias'] = rand(d)
        s[p+'attn.in_proj_weight'] = rand(3*d, d)
        s[p+'attn.in_proj_bias'] = rand(3*d)
        s[p+'attn.out_proj.weight'] = rand(d, d)
        s[p+'attn.out_proj.bias'] = rand(d)
        r = p+'moe.router.'
        s[r+('slots_tangent' if tangent else 'slots')] = rand(k, d)
        s[r+'log_temp'] = torch.tensor(1.4)
        s[r+'slot_to_expert'] = torch.tensor([1, 0, 0, 1])
        if tangent:
            s[r+'theta_c'] = torch.tensor(-.3+i*.5)
            if i: s[r+'W_tele'] = rand(k,k)
        for j in range(e):
            for name, shape in [('gate',(ff,d)), ('up',(ff,d)), ('down',(d,ff))]:
                s[p+f'moe.experts.{j}.{name}.weight'] = rand(*shape)
    cfg = dict(heads=2, geometry='poincare', curvature=1.4427,
               c_min=.25, c_max=4., beta_tele=.5, telepathy_scale=1.)
    return s, cfg


class CheckpointRedTeam(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'model.pt'
    def tearDown(self): self.tmp.cleanup()
    def load(self, s, cfg):
        torch.save({'model':s}, self.path)
        return cdm.load(self.path,cfg)
    def reject(self,s,cfg):
        with self.assertRaises(ValueError): self.load(s,cfg)

    def test_unknown_learned_tensor_rejected(self):
        s,c=fixture();s['layers.0.moe.secret_gate.weight']=torch.ones(8,8)
        self.reject(s,c)
    def test_missing_tensor_rejected(self):
        s,c=fixture();del s['layers.1.attn.out_proj.bias'];self.reject(s,c)
    def test_float32_overflow_rejected(self):
        s,c=fixture();key='layers.0.moe.experts.0.gate.weight'
        s[key]=s[key].double();s[key][0,0]=1e100;self.reject(s,c)
    def test_invalid_curvature_bounds_rejected(self):
        for bounds in [(4.,.25),(.25,float('inf')),(-1.,4.)]:
            with self.subTest(bounds=bounds):
                s,c=fixture(True);c.update(c_min=bounds[0],c_max=bounds[1]);self.reject(s,c)
    def test_bad_shapes_rejected(self):
        for key,value in [('layers.1.attn.in_proj_weight',torch.ones(23,8)),
                          ('layers.1.moe.experts.1.down.weight',torch.ones(8,11)),
                          ('layers.1.moe.router.slots',torch.ones(5,8)),
                          ('final_norm.bias',torch.ones(7))]:
            with self.subTest(key=key):
                s,c=fixture();s[key]=value;self.reject(s,c)
    def test_nonfinite_inference_tensors_rejected(self):
        for key in ['layers.0.moe.router.slots','layers.0.moe.router.log_temp',
                    'layers.0.moe.experts.0.gate.weight']:
            for bad in [float('nan'),float('inf')]:
                with self.subTest(key=key,bad=bad):
                    s,c=fixture();s[key].view(-1)[0]=bad;self.reject(s,c)
    def test_bad_mapping_rejected_before_export(self):
        for mapping in [torch.tensor([-1,0,0,1]),torch.tensor([0,0,0,2]),
                        torch.tensor([0,1]),torch.tensor([0.,1.,0.,1.])]:
            with self.subTest(mapping=mapping):
                s,c=fixture();s['layers.0.moe.router.slot_to_expert']=mapping
                self.reject(s,c)
    def test_conflicting_slot_representations_rejected(self):
        s,c=fixture();s['layers.0.moe.router.slots_tangent']=torch.ones(4,8)
        self.reject(s,c)
    def test_mixed_versions_rejected(self):
        s,c=fixture();p='layers.1.moe.router.';s[p+'slots_tangent']=s.pop(p+'slots')
        self.reject(s,c)
    def test_invalid_semantic_configuration_rejected(self):
        for override in [dict(heads=0),dict(heads=-2),dict(heads=3),
                         dict(geometry='euclidean'),dict(curvature=float('nan')),
                         dict(curvature=0.)]:
            with self.subTest(override=override):
                s,c=fixture();c.update(override);self.reject(s,c)
    def test_nonfinite_telepathy_rejected(self):
        s,c=fixture(True);c['telepathy_scale']=float('nan');self.reject(s,c)
    def test_training_buffers_accepted(self):
        s,c=fixture()
        for i in range(2):
            p=f'layers.{i}.'
            s[p+'_causal']=torch.ones(10,10,dtype=torch.bool).tril()
            for name in ['alpha','tau_star','slot_cum_weight','prev_tau_per_slot']:
                s[p+'moe.router.'+name]=torch.ones(4)
        self.load(s,c)
    def test_v1_ball_and_v3_tangent_are_distinguished(self):
        for tangent in [False,True]:
            with self.subTest(tangent=tangent):
                s,c=fixture(tangent);out,cfg=self.load(s,c)
                for i in range(2):
                    p=f'layers.{i}.moe.router.'
                    raw=s[p+('slots_tangent' if tangent else 'slots')]
                    expected=cdm.expmap(raw,cfg['curvatures'][i]) if tangent else raw
                    torch.testing.assert_close(out[p+'ball_slots'],cdm.clip(expected,cfg['curvatures'][i]))
    def test_frozen_forward_repeatable_and_causal(self):
        for tangent in [False,True]:
            with self.subTest(tangent=tangent),torch.inference_mode():
                s,c=self.load(*fixture(tangent));before={k:v.clone() for k,v in s.items()}
                ids=torch.tensor([1,5,3,7]);first=cdm.forward(s,c,ids)
                torch.testing.assert_close(first,cdm.forward(s,c,ids),rtol=0,atol=0)
                changed=cdm.forward(s,c,torch.tensor([1,5,8,2]))
                torch.testing.assert_close(first[:2],changed[:2],rtol=1e-6,atol=1e-6)
                for key in s: torch.testing.assert_close(s[key],before[key],rtol=0,atol=0)


    def test_source_eval_requires_explicit_freeze_flag(self):
        # Extract only reviewed architecture definitions, never execute training CLI/imports.
        source=Path('/home/ai/duoneural/A26B/scripts/cdm_lm_distill_v3.py')
        if not source.exists(): self.skipTest('Local original source unavailable')
        names={'_sqrt_c','clip_to_ball','exp_map_0','log_map_0','poincare_dist',
               'DualGeometryCDMRouter','ExpertFFN','CDMMoEBlock','CDMLayer','CDMMoELM'}
        tree=ast.parse(source.read_text())
        selected=ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[])
        ns={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'math':math}
        exec(compile(selected,str(source),'exec'),ns)
        torch.manual_seed(1994)
        model=ns['CDMMoELM'](vocab_size=19,d_model=8,n_layers=2,n_heads=2,K=4,num_experts=2,d_ff=12,max_seq=10,dropout=.1).eval()
        slots=[layer.moe.router.slots_tangent.clone() for layer in model.layers]
        ids=torch.tensor([[1,5,3,7]])
        with torch.inference_mode():
            first=model(ids,training=False,telepathy_scale=1.)[0]
            second=model(ids,training=False,telepathy_scale=1.)[0]
            torch.testing.assert_close(first,second,rtol=0,atol=0)
            for layer,old in zip(model.layers,slots):
                torch.testing.assert_close(layer.moe.router.slots_tangent,old,rtol=0,atol=0)
            model(ids)  # Demonstrate the source hazard that the frozen wrapper must avoid.
        self.assertTrue(any(not torch.equal(layer.moe.router.slots_tangent,old) for layer,old in zip(model.layers,slots)))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
