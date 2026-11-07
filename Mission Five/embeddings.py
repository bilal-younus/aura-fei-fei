"""Pretrained MiniLM sentence embeddings, cached for offline reproduction of all fixtures."""
import hashlib,importlib.util,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
MODEL='sentence-transformers/all-MiniLM-L6-v2'

def load_file(name):
    s=importlib.util.spec_from_file_location(name,ROOT/name);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def input_texts():
    hybrid=load_file('02_BM25_and_RRF_Combiner.py');safety=load_file('04_RAG_Provenance_and_Safety.py')
    chunks=json.loads((ROOT/'data/chunks.json').read_text())
    texts=[c['text'] for c in chunks]+[q['query'] for q in hybrid.QUERIES]+safety.ATTACKS+safety.BENIGN+safety.PROTOTYPES+safety.SECURITY_QUERIES
    texts += [s for c in chunks for s in safety.sentences(c['text'])]
    return sorted(set(texts))

class CachedEncoder:
    model=MODEL
    def __init__(self):
        self.texts=json.loads((ROOT/'data/embedding_texts.json').read_text());self.vectors=np.load(ROOT/'data/embeddings.npz')['vectors']
        self.lookup={t:i for i,t in enumerate(self.texts)}
    def encode(self,texts):
        try:return np.stack([self.vectors[self.lookup[t]] for t in texts])
        except KeyError as e:raise ValueError('Uncached text. Run python embeddings.py --rebuild after adding it to input_texts().') from e

def build_cache():
    import torch
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(MODEL);model=AutoModel.from_pretrained(MODEL,use_safetensors=True).eval()
    texts=input_texts();out=[]
    with torch.no_grad():
        for i in range(0,len(texts),16):
            batch=tokenizer(texts[i:i+16],padding=True,truncation=True,max_length=256,return_tensors='pt')
            hidden=model(**batch).last_hidden_state;mask=batch['attention_mask'].unsqueeze(-1)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(pooled,p=2,dim=1).numpy())
    vectors=np.concatenate(out).astype('float32')
    (ROOT/'data/embedding_texts.json').write_text(json.dumps(texts,indent=2)+'\n');np.savez_compressed(ROOT/'data/embeddings.npz',vectors=vectors)
    config=dict(model=MODEL,revision=getattr(model.config,'_commit_hash',None),dimension=int(vectors.shape[1]),
        pooling='attention-mask mean pooling followed by L2 normalization',max_tokens=256,text_count=len(texts),
        corpus_hash=hashlib.sha256((ROOT/'data/chunks.json').read_bytes()).hexdigest())
    (ROOT/'data/embedding_provenance.json').write_text(json.dumps(config,indent=2)+'\n');print(config)
if __name__=='__main__':build_cache()
