"""Pure NumPy/Python HNSW with bounded per-layer degree and heap-based ef search."""
import argparse,heapq,json,math,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parent

def normalize(x):
    x=np.asarray(x,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite vector')
    n=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(n==0):raise ValueError('Zero vectors have undefined cosine')
    return x/n

def cosine_similarity(u,v):
    a=normalize(u);b=normalize(v)
    return np.sum(a*b,axis=-1)

class HNSWIndex:
    def __init__(self,dim=768,M=12,ef_construction=32,capacity=25000,seed=505):
        if dim<1 or M<2 or ef_construction<M or capacity<1:raise ValueError('Invalid index parameters')
        self.dim=dim;self.M=M;self.ef_construction=ef_construction;self.m_L=1/math.log(M)
        self.vectors=np.empty((capacity,dim),np.float32);self.ids=[];self.id_to_pos={};self.graphs=[]
        self.levels=[];self.entry_point=None;self.max_level=-1;self.rng=np.random.default_rng(seed);self.last_distance_evaluations=0
    def _random_level(self):return int(-math.log(max(self.rng.random(),np.finfo(float).tiny))*self.m_L)
    def _dist(self,q,positions):
        self.last_distance_evaluations+=len(positions)
        return 1-self.vectors[positions]@q
    def search_layer(self,entry_point,query,ef,layer):
        if ef<1 or layer<0 or layer>self.max_level:raise ValueError('Invalid layer or ef')
        q=normalize(query)
        ep=int(entry_point);d=float(self._dist(q,[ep])[0]);seen={ep}
        candidates=[(d,ep)];best=[(-d,ep)]
        while candidates:
            distance,node=heapq.heappop(candidates)
            if len(best)>=ef and distance>-best[0][0]:break
            unseen=[n for n in self.graphs[layer].get(node,[]) if n not in seen]
            seen.update(unseen)
            if not unseen:continue
            for n,nd in zip(unseen,self._dist(q,unseen)):
                nd=float(nd)
                if len(best)<ef or nd<-best[0][0]:
                    heapq.heappush(candidates,(nd,n));heapq.heappush(best,(-nd,n))
                    if len(best)>ef:heapq.heappop(best)
        return sorted([(-d,n) for d,n in best])
    def insert(self,node_id,vector):
        if node_id in self.id_to_pos:raise ValueError('Duplicate ID')
        v=normalize(vector)
        if v.shape!=(self.dim,):raise ValueError('Dimension mismatch')
        p=len(self.ids)
        if p>=len(self.vectors):self.vectors=np.concatenate([self.vectors,np.empty_like(self.vectors)],axis=0)
        level=self._random_level();self.vectors[p]=v;self.ids.append(node_id);self.id_to_pos[node_id]=p;self.levels.append(level)
        for l in range(len(self.graphs),level+1):self.graphs.append({})
        for l in range(level+1):self.graphs[l][p]=[]
        if self.entry_point is None:self.entry_point=p;self.max_level=level;return
        ep=self.entry_point
        for l in range(self.max_level,level,-1):ep=self.search_layer(ep,v,1,l)[0][1]
        for l in range(min(level,self.max_level),-1,-1):
            candidates=self.search_layer(ep,v,self.ef_construction,l)
            selected=[n for _,n in candidates[:self.M]]
            self.graphs[l][p]=selected
            for n in selected:
                links=self.graphs[l][n];links.append(p);limit=2*self.M if l==0 else self.M
                if len(links)>limit:
                    # Nearest-M pruning: a documented simplified HNSW neighbor selector.
                    distances=self._dist(self.vectors[n],links)
                    keep=np.argsort(distances,kind='stable')[:limit]
                    self.graphs[l][n]=[links[j] for j in keep]
            if candidates:ep=candidates[0][1]
        if level>self.max_level:self.entry_point=p;self.max_level=level
    def query(self,query,top_k=5,ef=64):
        if top_k<1 or ef<1:raise ValueError('top_k and ef must be positive')
        if self.entry_point is None:return []
        q=normalize(query)
        if q.shape!=(self.dim,):raise ValueError('Dimension mismatch')
        self.last_distance_evaluations=0;ep=self.entry_point
        for l in range(self.max_level,0,-1):ep=self.search_layer(ep,q,1,l)[0][1]
        result=self.search_layer(ep,q,max(ef,top_k),0)
        return [(self.ids[n],float(1-d)) for d,n in result[:top_k]]

def benchmark(n=25000,dim=768,queries=40):
    rng=np.random.default_rng(505)
    # Clustered synthetic embeddings; held-out queries are independent samples, never indexed rows.
    centers=normalize(rng.normal(size=(64,dim)))
    x=normalize(centers[rng.integers(0,64,n)]+rng.normal(0,.035,(n,dim)))
    q=normalize(centers[rng.integers(0,64,queries)]+rng.normal(0,.035,(queries,dim)))
    rows=[];start=time.perf_counter();index=HNSWIndex(dim=dim,capacity=n)
    with threadpool_limits(limits=1):
        for i,v in enumerate(x):
            index.insert(i,v)
            if (i+1)%5000==0:print(f'Indexed {i+1}/{n}',flush=True)
        build=time.perf_counter()-start
        # Warm both paths and use the same normalized vectors and cosine metric.
        for query in q[:3]:index.query(query);_ = x@query
        for ef in [32,64,128]:
            times=[];flat_times=[];recalls=[];counts=[]
            for query in q:
                t=time.perf_counter_ns();approx=index.query(query,10,ef);times.append((time.perf_counter_ns()-t)/1e6);counts.append(index.last_distance_evaluations)
                t=time.perf_counter_ns();scores=x@query;top=np.argpartition(-scores,9)[:10];top=top[np.argsort(-scores[top])];flat_times.append((time.perf_counter_ns()-t)/1e6)
                recalls.append(len({i for i,_ in approx}&set(top))/10)
            rows.append(dict(ef=ef,hnsw_median_ms=float(np.median(times)),hnsw_p95_ms=float(np.percentile(times,95)),
                flat_median_ms=float(np.median(flat_times)),flat_p95_ms=float(np.percentile(flat_times,95)),recall_at_10=float(np.mean(recalls)),
                mean_distance_evaluations=float(np.mean(counts)),hnsw_times_ms=times,flat_times_ms=flat_times))
    result=dict(n=n,dim=dim,queries=queries,seed=505,M=index.M,ef_construction=index.ef_construction,build_seconds=build,
        level_population=[len(g) for g in index.graphs],vector_bytes=int(x.nbytes),threads=1,distribution='64 normalized Gaussian clusters; independent held-out queries',rows=rows)
    out=ROOT/'evidence';out.mkdir(exist_ok=True);(out/'hnsw_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in rows:print({k:v for k,v in r.items() if not k.endswith('times_ms')})
    return result
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--n',type=int,default=25000);parser.add_argument('--dim',type=int,default=768);parser.add_argument('--queries',type=int,default=40);args=parser.parse_args()
    benchmark(args.n,args.dim,args.queries)
