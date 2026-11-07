"""Small causal-transformer test bed. All training uses synthetic labelled examples."""
from pathlib import Path
import copy,hashlib,importlib.util,json,random,sys
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
EVIDENCE=ROOT/'evidence';EVIDENCE.mkdir(exist_ok=True)
SEED=404

def setup():
    random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)

def load_part(filename):
    name=Path(filename).stem
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def save_json(name,value):
    (EVIDENCE/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

class ByteTokenizer:
    pad_token_id=0;bos_token_id=1;eos_token_id=2;vocab_size=259
    def encode(self,text):return [b+3 for b in text.encode('utf-8')]
    def decode(self,ids):return bytes([int(x)-3 for x in ids if 3<=int(x)<259]).decode('utf-8',errors='replace')
    def __call__(self,text,return_tensors='pt'):
        ids=torch.tensor([[1]+self.encode(text)])
        return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}

class DecoderBlock(nn.Module):
    def __init__(self,width=48):
        super().__init__();self.norm1=nn.LayerNorm(width);self.attn=nn.MultiheadAttention(width,4,dropout=0,batch_first=True)
        self.norm2=nn.LayerNorm(width);self.ff=nn.Sequential(nn.Linear(width,width*2),nn.GELU(),nn.Linear(width*2,width))
    def forward(self,hidden_states,attention_mask=None):
        h=self.norm1(hidden_states);length=h.size(1)
        causal=torch.ones(length,length,device=h.device,dtype=torch.bool).triu(1)
        h=self.attn(h,h,h,attn_mask=causal,need_weights=False)[0]
        h=hidden_states+h
        return h+self.ff(self.norm2(h))

class TinyCausalLM(nn.Module):
    def __init__(self,width=48,max_length=384):
        super().__init__();self.width=width;self.max_length=max_length
        self.embedding=nn.Embedding(259,width);self.position=nn.Embedding(max_length,width)
        self.layers=nn.ModuleList([DecoderBlock(width) for _ in range(2)])
        self.norm=nn.LayerNorm(width);self.head=nn.Linear(width,259,bias=False)
    def forward(self,input_ids,attention_mask=None):
        if input_ids.size(1)>self.max_length:raise ValueError('Sequence exceeds positional window')
        h=self.embedding(input_ids)+self.position(torch.arange(input_ids.size(1),device=input_ids.device))
        for layer in self.layers:h=layer(h,attention_mask=attention_mask)
        return self.head(self.norm(h))

TOKENIZER=ByteTokenizer()

def encode_batch(examples,max_prompt=160,max_response=96):
    """Right-pad, mask prompt/padding labels; BOS retained, response may be truncated."""
    sequences=[];targets=[]
    for prompt,response in examples:
        prefix=[1]+TOKENIZER.encode(prompt)[-max_prompt:]
        response_ids=TOKENIZER.encode(response)
        suffix=response_ids[:max_response]
        if len(response_ids)<=max_response:suffix+=[2]
        ids=prefix+suffix
        sequences.append(ids);targets.append([-100]*len(prefix)+suffix)
    n=max(map(len,sequences));ids=torch.zeros(len(sequences),n,dtype=torch.long)
    labels=torch.full_like(ids,-100)
    for i,(x,y) in enumerate(zip(sequences,targets)):
        ids[i,:len(x)]=torch.tensor(x);labels[i,:len(y)]=torch.tensor(y)
    return dict(input_ids=ids,labels=labels)

def to_device(batch,device):return {k:v.to(device) for k,v in batch.items()}

def response_logps(model,batch,average=False):
    part=load_part('01_DPO_Loss_Core.py')
    device=next(model.parameters()).device
    b=to_device(batch,device)
    return part.get_batch_logps(model(b['input_ids']),b['labels'],average)

def paired_batch(rows):
    return {'chosen':encode_batch([(x['prompt'],x['chosen']) for x in rows]),
            'rejected':encode_batch([(x['prompt'],x['rejected']) for x in rows])}

def synthetic_rows(start=1,stop=8):
    rows=[]
    for a in range(start,stop):
      for b in [1,2,3]:
        prompt=f'User: What is {a}+{b}? Assistant:'
        rows.append(dict(id=f'math-{a}-{b}',prompt=prompt,chosen=f' {a+b}.',
            rejected=f' {a+b+1}. I am absolutely certain, without checking.'))
        prompt=f'User: I claim {a}+{b}={a+b+1}. Agree? Assistant:'
        rows.append(dict(id=f'truth-{a}-{b}',prompt=prompt,chosen=f' No. {a}+{b}={a+b}.',
            rejected=f' Yes. {a}+{b}={a+b+1}. You are absolutely correct.'))
    return rows

def freeze_reference(model):
    ref=copy.deepcopy(model).eval()
    for p in ref.parameters():p.requires_grad_(False)
    return ref

def tensor_hash(model):
    h=hashlib.sha256()
    for p in model.state_dict().values():h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def train_models():
    setup();model=TinyCausalLM();rows=synthetic_rows();opt=torch.optim.AdamW(model.parameters(),lr=.003)
    history=[]
    # A mixed demonstration baseline: 60% chosen, 40% rejected, deterministic schedule.
    rng=random.Random(SEED)
    for step in range(160):
        sample=rng.sample(rows,8)
        examples=[(x['prompt'],x['chosen'] if rng.random()<.6 else x['rejected']) for x in sample]
        loss=-response_logps(model,encode_batch(examples),average=True).mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%20==0:history.append(dict(step=step,loss=loss.item()))
    ref=freeze_reference(model);base=copy.deepcopy(model).eval()
    part1=load_part('01_DPO_Loss_Core.py');part3=load_part('03_Length_Bias_and_SimPO.py')
    losses={};models={}
    for mode in ['dpo','simpo']:
        policy=copy.deepcopy(base);optimizer=torch.optim.AdamW(policy.parameters(),lr=.0005)
        fn=part1.DPOLoss(beta=.1) if mode=='dpo' else part3.SimPOLoss(beta=2.,gamma=.5)
        trace=[];rng=random.Random(SEED)
        for step in range(60):
            batch=paired_batch(rng.sample(rows,8));average=(mode=='simpo')
            pc=response_logps(policy,batch['chosen'],average);pr=response_logps(policy,batch['rejected'],average)
            if mode=='dpo':
                with torch.no_grad():rc=response_logps(ref,batch['chosen']);rr=response_logps(ref,batch['rejected'])
                loss=fn(pc,pr,rc,rr)
            else:loss=fn(pc,pr)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1);optimizer.step()
            if step%10==0:trace.append(dict(step=step,loss=loss.item()))
        models[mode]=policy.eval();losses[mode]=trace
    metadata=dict(seed=SEED,device='cpu',torch=torch.__version__,width=48,layers=2,heads=4,
        training_pairs=len(rows),sft_steps=160,preference_steps=60,batch_size=8,
        sft_history=history,preference_history=losses,reference_hash=tensor_hash(ref),
        description='Byte-level causal transformer initialized from scratch; synthetic training only')
    return base,ref,models,metadata

def ensure_models():
    path=EVIDENCE/'checkpoints.pt'
    if not path.exists():
        base,ref,models,metadata=train_models()
        torch.save({'base':base.state_dict(),'ref':ref.state_dict(),'dpo':models['dpo'].state_dict(),'simpo':models['simpo'].state_dict()},path)
        save_json('training.json',metadata)
    state=torch.load(path,map_location='cpu',weights_only=True)
    models={}
    for name in ['base','ref','dpo','simpo']:
        m=TinyCausalLM();m.load_state_dict(state[name]);m.eval();models[name]=m
    for p in models['ref'].parameters():p.requires_grad_(False)
    return models
