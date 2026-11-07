"""Reference-free SimPO, length correlation and measured model/memory accounting."""
import copy,gc,json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class SimPOLoss(nn.Module):
    def __init__(self,beta=2.,gamma=.5):
        super().__init__()
        if beta<=0 or gamma<0:raise ValueError('beta>0 and gamma>=0 required')
        self.beta=beta;self.gamma=gamma
    def forward(self,chosen_average_logps,rejected_average_logps):
        if chosen_average_logps.shape!=rejected_average_logps.shape:raise ValueError('Mismatched pair shapes')
        return -F.logsigmoid(self.beta*(chosen_average_logps-rejected_average_logps)-self.gamma).mean()

def pearson(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)<2 or x.std()==0 or y.std()==0:return None
    return float(np.corrcoef(x,y)[0,1])

def memory_benchmark():
    from common import TinyCausalLM,paired_batch,synthetic_rows,response_logps,freeze_reference,load_part
    model=TinyCausalLM();parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    result=dict(parameter_bytes_per_model=parameter_bytes,static_dpo_model_bytes=2*parameter_bytes,
        static_simpo_model_bytes=parameter_bytes,static_model_storage_reduction_percent=50.,
        note='Exact parameter-storage accounting; optimizer states, activations and allocator overhead are excluded. Not total VRAM.')
    result['cuda_available']=torch.cuda.is_available()
    if not torch.cuda.is_available():
        result['cuda_peak_allocated_MB']=None;result['cuda_status']='Not measured: no CUDA device on this host'
        return result
    measurements={}
    for mode in ['dpo','simpo']:
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        policy=TinyCausalLM().cuda();ref=freeze_reference(policy) if mode=='dpo' else None
        batch=paired_batch(synthetic_rows()[:4]);optimizer=torch.optim.AdamW(policy.parameters(),lr=.001)
        for _ in range(2):
            pc=response_logps(policy,batch['chosen'],mode=='simpo');pr=response_logps(policy,batch['rejected'],mode=='simpo')
            if mode=='dpo':
                with torch.no_grad():rc=response_logps(ref,batch['chosen']);rr=response_logps(ref,batch['rejected'])
                loss=load_part('01_DPO_Loss_Core.py').DPOLoss()(pc,pr,rc,rr)
            else:loss=SimPOLoss()(pc,pr)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        torch.cuda.synchronize();measurements[mode]=torch.cuda.max_memory_allocated()/1e6
        del policy,ref,optimizer,loss,pc,pr
        if mode=='dpo':del rc,rr
    result['cuda_peak_allocated_MB']=measurements
    result['cuda_peak_reduction_percent']=100*(1-measurements['simpo']/measurements['dpo'])
    return result

def run():
    from common import ROOT,setup,ensure_models,paired_batch,response_logps,save_json
    setup();models=ensure_models()
    rows=[json.loads(x) for x in (ROOT/'data/hh_human_validation.jsonl').read_text().splitlines()]
    differences=[];dpo_margins=[];normalized_margins=[]
    for i in range(0,len(rows),8):
        sample=rows[i:i+8];batch=paired_batch(sample)
        with torch.no_grad():
            dc=response_logps(models['dpo'],batch['chosen'])-response_logps(models['ref'],batch['chosen'])
            dr=response_logps(models['dpo'],batch['rejected'])-response_logps(models['ref'],batch['rejected'])
            sc=response_logps(models['simpo'],batch['chosen'],True);sr=response_logps(models['simpo'],batch['rejected'],True)
        lengths=[((batch[k]['labels'][:,1:]!=-100).sum(-1)).tolist() for k in ['chosen','rejected']]
        differences.extend(a-b for a,b in zip(*lengths));dpo_margins+=(dc-dr).tolist();normalized_margins+=(sc-sr).tolist()
    result=dict(n=len(rows),length_unit='Scored byte tokens including EOS when completion is untruncated',
        pearson_dpo_length_margin=pearson(differences,dpo_margins),pearson_simpo_length_margin=pearson(differences,normalized_margins),
        length_differences=differences,dpo_margins=dpo_margins,simpo_average_logp_margins=normalized_margins,
        simpo_beta=2.,simpo_gamma=.5,memory=memory_benchmark())
    save_json('length_and_memory.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for ax,y,title in zip(axes,[dpo_margins,normalized_margins],['DPO log-ratio margin','SimPO mean-logp margin']):
        ax.scatter(differences,y,s=18,alpha=.7);ax.set(xlabel='Chosen minus rejected scored tokens',ylabel='Preference margin',title=title)
    fig.tight_layout();fig.savefig(ROOT/'evidence/length_bias.png',dpi=180);plt.close(fig)
    print('Length correlation DPO:',result['pearson_dpo_length_margin'],'SimPO:',result['pearson_simpo_length_margin'])
    print(json.dumps(result['memory'],indent=2));return result
if __name__=='__main__':run()
