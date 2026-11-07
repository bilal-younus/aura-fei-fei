from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parent
b=json.loads((r/'evidence/hnsw_benchmark.json').read_text())
fig,ax=plt.subplots(figsize=(6,3.5));rows=b['rows']
ax.plot([x['ef'] for x in rows],[x['hnsw_median_ms'] for x in rows],'o-',label='HNSW median')
ax.plot([x['ef'] for x in rows],[x['flat_median_ms'] for x in rows],'s--',label='Flat median')
ax.set(xlabel='ef search beam',ylabel='Milliseconds/query',title='25,000 vectors x 768 dimensions');ax.legend();fig.tight_layout();fig.savefig(r/'evidence/latency.png',dpi=160);plt.close(fig)
