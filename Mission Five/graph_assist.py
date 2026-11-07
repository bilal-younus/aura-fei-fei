"""Transparent graph-assisted retrieval, not a reimplementation of Microsoft's GraphRAG."""
import json
from pathlib import Path
import networkx as nx
ROOT=Path(__file__).resolve().parent

def run():
    chunks=json.loads((ROOT/'data/chunks.json').read_text());g=nx.Graph()
    for c in chunks:g.add_node(c['id'])
    for i,a in enumerate(chunks):
        for b in chunks[i+1:]:
            common=set(a['entities'])&set(b['entities'])
            if common:g.add_edge(a['id'],b['id'],weight=len(common),shared_entities=sorted(common))
    communities=nx.community.louvain_communities(g,weight='weight',seed=505)
    byid={c['id']:c for c in chunks};summaries=[]
    for i,community in enumerate(communities):
        ids=sorted(community);summaries.append(dict(community=i,chunk_ids=ids,summary=' '.join(byid[n]['text'].split('. ')[0]+'.' for n in ids[:2])))
    neighbors=list(g.neighbors('ship_cooling'));beryl=[n for n in neighbors if 'Beryl' in g.edges['ship_cooling',n]['shared_entities']]
    result=dict(nodes=g.number_of_nodes(),edges=g.number_of_edges(),community_algorithm='Louvain on hand-labelled entity co-occurrence; not Leiden',
        summaries=summaries,local_graph_expansion=dict(seed='ship_cooling',shared_entity='Beryl',neighbors=sorted(beryl)),
        limitation='Extractive community summaries and fixed entity labels; no LLM entity extraction or global generative map-reduce.')
    (ROOT/'evidence/graph_assist.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Entity graph:',result['nodes'],'nodes',result['edges'],'edges;',len(summaries),'communities');print('Beryl expansion:',beryl)
    return result
if __name__=='__main__':run()
