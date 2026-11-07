"""Causal response-token likelihoods and DPO, directly in PyTorch."""
import math
import torch
from torch import nn
import torch.nn.functional as F

def get_batch_logps(logits,labels,average_log_prob=False):
    """logits [B,T,V] predict labels one step later; -100 masks prompt and padding."""
    if logits.ndim!=3 or labels.shape!=logits.shape[:2]:raise ValueError('Expected [B,T,V] logits and [B,T] labels')
    if logits.size(1)<2:raise ValueError('Need a next-token prediction')
    labels=labels[:,1:].clone();logits=logits[:,:-1,:]
    mask=labels.ne(-100);safe=labels.masked_fill(~mask,0)
    if torch.any((safe<0)|(safe>=logits.size(-1))):raise ValueError('Label outside vocabulary')
    counts=mask.sum(-1)
    if torch.any(counts==0):raise ValueError('Each response must contain a scored token')
    token_logps=torch.gather(F.log_softmax(logits.float(),dim=-1),-1,safe.unsqueeze(-1)).squeeze(-1)
    sums=(token_logps*mask).sum(-1)
    return sums/counts if average_log_prob else sums

class DPOLoss(nn.Module):
    def __init__(self,beta=.1):
        super().__init__()
        if beta<=0:raise ValueError('beta must be positive')
        self.beta=beta
    def forward(self,policy_chosen_logps,policy_rejected_logps,reference_chosen_logps,reference_rejected_logps):
        values=[policy_chosen_logps,policy_rejected_logps,reference_chosen_logps,reference_rejected_logps]
        if any(v.shape!=values[0].shape for v in values):raise ValueError('Mismatched pair shapes')
        if any(not torch.isfinite(v).all() for v in values):raise ValueError('Nonfinite log probabilities')
        chosen_delta=policy_chosen_logps-reference_chosen_logps.detach()
        rejected_delta=policy_rejected_logps-reference_rejected_logps.detach()
        # logsigmoid is stable at extreme margins: no exp->log underflow or gradient-destroying clamp.
        return -F.logsigmoid(self.beta*(chosen_delta-rejected_delta)).mean()

def gradient_demo():
    from common import setup,TinyCausalLM,freeze_reference,paired_batch,synthetic_rows,response_logps,tensor_hash,save_json
    setup();policy=TinyCausalLM();ref=freeze_reference(policy);batch=paired_batch(synthetic_rows()[:4])
    before=tensor_hash(ref);active=tensor_hash(policy)
    pc=response_logps(policy,batch['chosen']);pr=response_logps(policy,batch['rejected'])
    with torch.no_grad():rc=response_logps(ref,batch['chosen']);rr=response_logps(ref,batch['rejected'])
    loss=DPOLoss()(pc,pr,rc,rr);optimizer=torch.optim.SGD(policy.parameters(),lr=.01)
    optimizer.zero_grad();loss.backward()
    grad=sum(p.grad.square().sum().item() for p in policy.parameters() if p.grad is not None)**.5
    optimizer.step()
    result=dict(initial_loss=loss.item(),expected_equal_policy_loss=math.log(2),policy_gradient_l2=grad,
        policy_changed=active!=tensor_hash(policy),reference_unchanged=before==tensor_hash(ref),
        reference_gradients_absent=all(p.grad is None for p in ref.parameters()),reference_hash=before)
    assert result['policy_changed'] and result['reference_unchanged'] and result['reference_gradients_absent']
    save_json('dpo_gradient.json',result);return result

if __name__=='__main__':
    import json
    print(json.dumps(gradient_demo(),indent=2))
