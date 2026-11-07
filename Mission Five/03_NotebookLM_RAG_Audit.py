"""Score the actual browser-observed lab records; never fabricate service responses."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def citation_precision(records):
    if not records:return None
    return sum(bool(r['entailed']) for r in records)/len(records)

def run():
    source=ROOT/'evidence/notebooklm_observations.json'
    if not source.exists():raise FileNotFoundError('The observed NotebookLM audit record is required')
    data=json.loads(source.read_text());citations=data['citations'];negatives=data['negative_probes']
    result=dict(notebook_url=data['notebook_url'],sources=data['sources'],citation_occurrences=len(citations),
        entailed_citations=sum(x['entailed'] for x in citations),citation_precision=citation_precision(citations),
        scoring_rule=data['scoring_rule'],negative_probes=len(negatives),
        refusals=sum(n['refused_without_inventing_requested_fact'] for n in negatives),audio=data.get('audio',{}))
    (ROOT/'evidence/notebooklm_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='audio'},indent=2));print('Audio status:',result['audio'].get('status'));return result
if __name__=='__main__':run()
