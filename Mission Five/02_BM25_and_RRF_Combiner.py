"""Exact identifiers, BM25 and one-based rank fusion. No score-scale mixing."""
import collections,importlib.util,json,math,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def tokens(text):return re.findall(r'[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*',text.casefold())

class BM25:
    def __init__(self,documents,k1=1.5,b=.75):
        if k1<=0 or not 0<=b<=1:raise ValueError('Invalid BM25 parameters')
        self.k1=k1;self.b=b;self.docs=[tokens(d) for d in documents]
        self.tf=[collections.Counter(d) for d in self.docs];self.lengths=np.array([len(d) for d in self.docs]);self.avgdl=float(self.lengths.mean()) if len(self.docs) else 0
        self.df=collections.Counter(t for d in self.docs for t in set(d))
    def scores(self,query):
        result=np.zeros(len(self.docs));n=len(self.docs)
        if not n or not self.avgdl:return result
        for term in set(tokens(query)):
            df=self.df.get(term,0)
            if not df:continue
            idf=math.log(1+(n-df+.5)/(df+.5))
            f=np.array([d.get(term,0) for d in self.tf],float)
            result+=idf*f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*self.lengths/self.avgdl))
        return result
    def rank(self,query,top_k=10):
        scores=self.scores(query)
        return [(int(i),float(scores[i])) for i in np.argsort(-scores,kind='stable')[:top_k] if scores[i]>0]

def reciprocal_rank_fusion(rankings,k=60):
    if k<0:raise ValueError('k must be nonnegative')
    scores=collections.defaultdict(float)
    for ranking in rankings:
        unique=list(dict.fromkeys(item[0] if isinstance(item,(tuple,list)) else item for item in ranking))
        for rank,doc in enumerate(unique,1):scores[doc]+=1/(k+rank)
    return sorted(scores.items(),key=lambda x:(-x[1],str(x[0])))

QUERIES=[
 dict(id='concept_memory',query='How can an assistant remember relevant past facts without identical wording?',relevant=['semantic_memory','rag_memory'],kind='conceptual'),
 dict(id='concept_global',query='How do we synthesize the main themes across the entire document collection?',relevant=['graph_global','graph_reduce'],kind='conceptual'),
 dict(id='exact_serial',query='XQ-17B-904',relevant=['ship_cooling'],kind='exact identifier'),
 dict(id='exact_sensor',query='ST-83-112',relevant=['ship_sensor'],kind='exact identifier'),
 dict(id='mixed',query='Which power controller feeds the Beryl coolant pump?',relevant=['ship_power'],kind='mixed'),
]

def load_hnsw():
    spec=importlib.util.spec_from_file_location('hnsw05',ROOT/'01_HNSW_Vector_Index.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.HNSWIndex

class HybridRetriever:
    def __init__(self,chunks,encoder):
        self.chunks=chunks;self.encoder=encoder
        vectors=encoder.encode([c['text'] for c in chunks])
        self.index=load_hnsw()(dim=vectors.shape[1],M=6,ef_construction=24,capacity=max(len(chunks),1))
        for i,v in enumerate(vectors):self.index.insert(i,v)
        self.bm25=BM25([c['text'] for c in chunks])
    def retrieve(self,query,k=5,candidates=10):
        dense=self.index.query(self.encoder.encode([query])[0],candidates,ef=max(24,candidates));sparse=self.bm25.rank(query,candidates)
        fused=reciprocal_rank_fusion([dense,sparse],60)[:k]
        return dict(dense=dense,sparse=sparse,fused=fused)

def run():
    from embeddings import CachedEncoder
    chunks=json.loads((ROOT/'data/chunks.json').read_text());encoder=CachedEncoder();retriever=HybridRetriever(chunks,encoder);rows=[]
    for case in QUERIES:
        result=retriever.retrieve(case['query']);row={**case}
        for method in ['dense','sparse','fused']:
            ids=[chunks[i]['id'] for i,_ in result[method]];row[method]=ids
            row[method+'_hit_at_1']=bool(ids and ids[0] in case['relevant']);row[method+'_hit_at_3']=bool(set(ids[:3])&set(case['relevant']));row[method+'_recall_at_3']=len(set(ids[:3])&set(case['relevant']))/len(case['relevant'])
        rows.append(row);print(case['id'],{k:row[k][:3] for k in ['dense','sparse','fused']})
    out=dict(embedding_model=encoder.model,corpus_chunks=len(chunks),k1=1.5,b=.75,rrf_k=60,queries=rows)
    (ROOT/'evidence/hybrid_results.json').write_text(json.dumps(out,indent=2)+'\n');return out
if __name__=='__main__':run()
