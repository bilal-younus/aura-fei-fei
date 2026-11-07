"""Implicit DPO rewards and reliability against held-out human pairwise labels."""
import json,math,random
import numpy as np
import torch
from common import ROOT,SEED,setup,paired_batch,response_logps,ensure_models,save_json

@torch.no_grad()
def evaluate_implicit_rewards(policy,ref,batch,beta=.1):
    policy.eval();ref.eval()
    pc=response_logps(policy,batch['chosen']);pr=response_logps(policy,batch['rejected'])
    rc=response_logps(ref,batch['chosen']);rr=response_logps(ref,batch['rejected'])
    rw=beta*(pc-rc);rl=beta*(pr-rr)
    return dict(chosen_rewards=rw.cpu().tolist(),rejected_rewards=rl.cpu().tolist(),
        probabilities=torch.sigmoid(rw-rl).cpu().tolist(),accuracy=(rw>rl).float().mean().item(),
        margins=((pc-rc)-(pr-rr)).cpu().tolist())

def calibration(probabilities,targets,n_bins=5):
    p=np.asarray(probabilities);y=np.asarray(targets);bins=[]
    for i in range(n_bins):
        lo=i/n_bins;hi=(i+1)/n_bins;mask=(p>=lo)&((p<=hi) if i==n_bins-1 else (p<hi))
        if mask.any():bins.append(dict(lower=lo,upper=hi,count=int(mask.sum()),predicted=float(p[mask].mean()),empirical=float(y[mask].mean())))
    return dict(bins=bins,brier=float(((p-y)**2).mean()),ece=sum(b['count']/len(y)*abs(b['predicted']-b['empirical']) for b in bins))

def run():
    setup();models=ensure_models();rows=[json.loads(x) for x in (ROOT/'data/hh_human_validation.jsonl').read_text().splitlines()]
    all_results=[]
    for start in range(0,len(rows),8):
        batch=rows[start:start+8];r=evaluate_implicit_rewards(models['dpo'],models['ref'],paired_batch(batch))
        for i,row in enumerate(batch):
            all_results.append(dict(id=row['id'],p_chosen=r['probabilities'][i],chosen_reward=r['chosen_rewards'][i],
                rejected_reward=r['rejected_rewards'][i],margin=r['margins'][i],
                chosen_bytes=min(len(row['chosen'].encode()),96),rejected_bytes=min(len(row['rejected'].encode()),96),
                prompt_truncated=len(row['prompt'].encode())>160,
                chosen_truncated=len(row['chosen'].encode())>96,rejected_truncated=len(row['rejected'].encode())>96))
    # Randomize orientation with exactly 32 of 64 chosen-first, independent of score.
    orientation=[0]*(len(rows)//2)+[1]*(len(rows)-len(rows)//2);random.Random(SEED).shuffle(orientation)
    probabilities=[x['p_chosen'] if y else 1-x['p_chosen'] for x,y in zip(all_results,orientation)]
    cal=calibration(probabilities,orientation)
    result=dict(dataset='Anthropic HH-RLHF helpful-base test',n=len(rows),label_type='Human chosen/rejected, not repeated vote fractions',
        pairwise_accuracy=sum(x['chosen_reward']>x['rejected_reward'] for x in all_results)/len(rows),
        calibration=cal,orientation_targets=orientation,oriented_probabilities=probabilities,pairs=all_results,
        truncation='Final 160 prompt bytes and first 96 response bytes; EOS scored only for untruncated completions')
    save_json('implicit_rewards.json',result)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4));ax.plot([0,1],[0,1],'--',color='gray',label='Ideal')
    bins=cal['bins'];ax.plot([b['predicted'] for b in bins],[b['empirical'] for b in bins],'o-',label='64 human-labelled pairs')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted P(first preferred)',ylabel='Observed fraction first preferred',title='Bradley-Terry reliability (five fixed bins)')
    ax.legend();fig.tight_layout();fig.savefig(ROOT/'evidence/reliability.png',dpi=180);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['dataset','n','pairwise_accuracy']},indent=2));print('Brier:',cal['brier'],'ECE:',cal['ece'])
    return result
if __name__=='__main__':run()
