"""Word-substitution channel with optional repetition and positional MI estimates."""
import math,random
from collections import Counter
NOISE=('Banana','Labyrinth','Quantum','Midnight','Velocity','Green','Cosmological')

def entropy(values):
    c=Counter(values);n=sum(c.values())
    return -sum((v/n)*math.log2(v/n) for v in c.values()) if n else 0.0

def mutual_information(xs,ys):
    if len(xs)!=len(ys) or not xs:raise ValueError('Need nonempty aligned symbol sequences')
    n=len(xs);cx=Counter(xs);cy=Counter(ys);joint=Counter(zip(xs,ys))
    return sum(v/n*math.log2(v*n/(cx[x]*cy[y])) for (x,y),v in joint.items())

def capacity(bandwidth,snr):
    if bandwidth<=0 or snr<0:raise ValueError('Bandwidth must be positive; SNR nonnegative')
    return bandwidth*math.log2(1+snr)

class NoisyChannel:
    def __init__(self,seed=42):self.rng=random.Random(seed)
    def transmit(self,text,redundant=True,substitutions=1):
        words=text.split()
        if not words or len(words)>7:raise ValueError('Payload must contain 1..7 whitespace words')
        if any(w in NOISE for w in words):raise ValueError('Reserved noise vocabulary in payload')
        # At most two words can receive full triple protection under seven-word limit.
        repeat=3 if redundant and len(words)<=2 else 1
        encoded=[w for w in words for _ in range(repeat)]
        if not 1<=substitutions<=len(encoded):raise ValueError('Invalid substitution count')
        received=encoded.copy()
        for i in self.rng.sample(range(len(encoded)),substitutions):received[i]=self.rng.choice(NOISE)
        decoded=[]
        for i in range(0,len(received),repeat):
            group=[w for w in received[i:i+repeat] if w not in NOISE]
            decoded.append(Counter(group).most_common(1)[0][0] if group else '[ERASURE]')
        assert len(received)<=7 and any(w in NOISE for w in received)
        return dict(payload=text,encoded=' '.join(encoded),received=' '.join(received),decoded=' '.join(decoded),
            words=len(received),substitutions=substitutions,repetition=repeat,recovered=words==decoded)
