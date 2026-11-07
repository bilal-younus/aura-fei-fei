import copy,math,unittest
import torch
from common import setup,load_part,TinyCausalLM,freeze_reference,encode_batch,TOKENIZER,tensor_hash
p1=load_part('01_DPO_Loss_Core.py');p2=load_part('02_Implicit_Reward_Evaluator.py')
p3=load_part('03_Length_Bias_and_SimPO.py');p4=load_part('04_Activation_Steering_and_Audit.py')

class ObjectiveTests(unittest.TestCase):
    def setUp(self):setup()
    def test_causal_gather_matches_manual(self):
        logits=torch.randn(2,5,7);labels=torch.tensor([[-100,-100,3,4,2],[-100,2,1,-100,-100]])
        expected=[]
        for i in range(2):
            terms=[logits[i,t-1].log_softmax(0)[labels[i,t]] for t in range(1,5) if labels[i,t]!=-100]
            expected.append(torch.stack(terms).sum())
        torch.testing.assert_close(p1.get_batch_logps(logits,labels),torch.stack(expected))
    def test_prompt_and_padding_invariance(self):
        logits=torch.randn(1,6,7);labels=torch.tensor([[-100,-100,3,2,-100,-100]])
        expected=p1.get_batch_logps(logits,labels);logits[:,0,:]+=torch.randn(7)*10;logits[:,3:,:]+=torch.randn(3,7)*10
        torch.testing.assert_close(p1.get_batch_logps(logits,labels),expected)
    def test_averaging(self):
        logits=torch.randn(1,4,5);labels=torch.tensor([[-100,-100,3,2]])
        torch.testing.assert_close(p1.get_batch_logps(logits,labels,True),p1.get_batch_logps(logits,labels)/2)
    def test_empty_response_rejected(self):
        with self.assertRaises(ValueError):p1.get_batch_logps(torch.zeros(1,3,5),torch.full((1,3),-100))
    def test_invalid_shape_or_label(self):
        with self.assertRaises(ValueError):p1.get_batch_logps(torch.zeros(1,3,5),torch.zeros(1,2,dtype=torch.long))
        with self.assertRaises(ValueError):p1.get_batch_logps(torch.zeros(1,3,5),torch.tensor([[-100,99,0]]))
    def test_equal_policy_dpo(self):
        x=torch.tensor([-3.,-4.]);torch.testing.assert_close(p1.DPOLoss()(x,x,x,x),torch.tensor(math.log(2)))
    def test_dpo_reference_detached(self):
        pc=torch.tensor([-2.],requires_grad=True);pr=torch.tensor([-3.],requires_grad=True)
        rc=torch.tensor([-2.],requires_grad=True);rr=torch.tensor([-2.],requires_grad=True)
        p1.DPOLoss()(pc,pr,rc,rr).backward()
        self.assertLess(pc.grad.item(),0);self.assertGreater(pr.grad.item(),0)
        self.assertIsNone(rc.grad);self.assertIsNone(rr.grad)
    def test_extreme_margins_finite(self):
        for v in [-10000.,10000.]:
            pc=torch.tensor([v],requires_grad=True);z=torch.zeros(1)
            loss=p1.DPOLoss()(pc,z,z,z);loss.backward()
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(pc.grad).all())
    def test_gradient_demo(self):
        r=p1.gradient_demo();self.assertTrue(r['policy_changed'] and r['reference_unchanged'] and r['reference_gradients_absent'])
    def test_simpo_formula(self):
        a=torch.tensor([-.3,-.8]);b=torch.tensor([-.7,-.2])
        torch.testing.assert_close(p3.SimPOLoss()(a,b),-torch.nn.functional.logsigmoid(2*(a-b)-.5).mean())
    def test_simpo_length_invariance(self):
        self.assertEqual(p3.SimPOLoss()(torch.tensor([-6./3]),torch.tensor([-12./6])).item(),p3.SimPOLoss()(torch.tensor([-12./6]),torch.tensor([-24./12])).item())
    def test_calibration_counts(self):
        r=p2.calibration([0,.2,.6,1],[0,0,1,1]);self.assertEqual(sum(b['count'] for b in r['bins']),4)
    def test_pearson_undefined(self):
        self.assertIsNone(p3.pearson([1,1],[2,3]));self.assertAlmostEqual(p3.pearson([1,2,3],[3,2,1]),-1)
    def test_hook_alpha_zero_and_cleanup(self):
        m=TinyCausalLM().eval();ids=torch.tensor([[1,20,30]])
        before=m(ids).detach();weights=tensor_hash(m)
        with p4.ActivationSteeringHook(m.layers[1],torch.ones(48),0):torch.testing.assert_close(m(ids),before,rtol=0,atol=0)
        self.assertEqual(len(m.layers[1]._forward_pre_hooks),0);self.assertEqual(weights,tensor_hash(m))
    def test_hook_cleanup_on_exception(self):
        m=TinyCausalLM()
        with self.assertRaises(RuntimeError):
            with p4.ActivationSteeringHook(m.layers[0],torch.ones(48)):raise RuntimeError('test failure')
        self.assertFalse(m.layers[0]._forward_pre_hooks)
    def test_contrast_identical_zero(self):
        m=TinyCausalLM();v=p4.extract_steering_vector(m,TOKENIZER,[{'honest':'same','sycophantic':'same'}],1)
        torch.testing.assert_close(v,torch.zeros(48),rtol=0,atol=0)
    def test_hook_changes_activations_without_weight_change(self):
        m=TinyCausalLM().eval();ids=torch.tensor([[1,20,30]]);before=m(ids).detach();weights=tensor_hash(m)
        with p4.ActivationSteeringHook(m.layers[1],torch.arange(48).float()*.01):after=m(ids).detach()
        self.assertFalse(torch.equal(before,after));self.assertEqual(weights,tensor_hash(m))
    def test_implicit_reward_ties_not_wins(self):
        m=TinyCausalLM();r=freeze_reference(m);batch={'chosen':encode_batch([('Q',' A')]),'rejected':encode_batch([('Q',' B')])}
        out=p2.evaluate_implicit_rewards(m,r,batch);self.assertEqual(out['accuracy'],0);self.assertEqual(out['probabilities'],[.5])

if __name__=='__main__':unittest.main(verbosity=2)
