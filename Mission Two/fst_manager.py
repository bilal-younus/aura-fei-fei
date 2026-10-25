"""Deterministic dialogue controller; no model output can authorize navigation."""
import re
from enum import Enum

class State(str, Enum):
    INIT='INIT'; GREETING='GREETING'; COLLECT_SLOTS='COLLECT_SLOTS'
    CONFIRMATION='CONFIRMATION'; TERMINAL='TERMINAL'; FALLBACK='FALLBACK'

class FSTDialogueManager:
    def __init__(self, max_retries=3):
        self.state=State.INIT
        self.slot_memory={}
        self.retries=0
        self.max_retries=max_retries
        self.authorized=False
        self.trace=[]

    def _fallback(self, reason):
        self.retries+=1
        self.state=State.TERMINAL if self.retries>=self.max_retries else State.FALLBACK
        return 'Session aborted' if self.state==State.TERMINAL else 'Please clarify'

    def step(self, text):
        before=self.state
        response=self._step(text)
        self.trace.append(dict(input=text,before=before.value,after=self.state.value,
            output=response,slots=dict(self.slot_memory),retries=self.retries,authorized=self.authorized))
        return response

    def _step(self,text):
        if self.state==State.TERMINAL:return 'Session closed'
        text=text.strip()
        low=text.lower()
        if low in {'cancel','abort'}:
            self.state=State.TERMINAL;return 'Navigation cancelled'
        # Conservative grammar: rejects obvious instruction overrides and negated slot requests.
        if re.search(r'ignore|override|system prompt|bypass|actually|sarcas|yeah right|\bnot\b|\bdon.t\b',low):
            return self._fallback('unsafe or ambiguous utterance')
        if not text:return self._fallback('empty input')
        if self.state==State.INIT and re.fullmatch(r'(hello|hi|hello aura online|aura online)',low):
            self.state=State.GREETING;self.retries=0;return 'Destination please'
        # Restrict all values to finite domains, making the slot-augmented machine finite.
        found={}
        for name in ['Alpha Centauri','Orion Arm','Sirius']:
            if re.search(r'\b'+re.escape(name)+r'\b',text,re.I):found['destination']=name
        destinations=[name for name in ['Alpha Centauri','Orion Arm','Sirius'] if re.search(r'\b'+re.escape(name)+r'\b',text,re.I)]
        speeds=re.findall(r'\b(?:warp|speed)\s*=?\s*(\d+(?:\.\d+)?)\b',low)
        if len(destinations)>1 or len(set(speeds))>1:return self._fallback('conflicting values')
        if speeds:
            value=float(speeds[0])
            if value not in {i/10 for i in range(1,100)}:return self._fallback('invalid speed')
            found['speed']=f'{value:g}'
        codes=re.findall(r'\b(?:ALPHA|BETA)-\d+\b',text,re.I)
        if codes:
            if {c.upper() for c in codes}!={'ALPHA-7'}:return self._fallback('invalid clearance')
            found['clearance_code']='ALPHA-7'
        if found:
            self.slot_memory.update(found);self.retries=0
            if len(self.slot_memory)==3:
                self.state=State.CONFIRMATION
                return f"Confirm {self.slot_memory['destination'].replace(' ','_')}/{self.slot_memory['speed']}/ALPHA-7"
            self.state=State.COLLECT_SLOTS
            for slot,prompt in [('destination','Destination please'),('speed','Speed please'),('clearance_code','Clearance please')]:
                if slot not in self.slot_memory:return prompt
        if low in {'yes','confirm','confirmed'} and self.state in {State.CONFIRMATION,State.FALLBACK} and len(self.slot_memory)==3:
            self.state=State.TERMINAL;self.authorized=True;self.retries=0;return 'Navigation authorized'
        if low in {'no','reject'} and len(self.slot_memory)==3:
            self.state=State.COLLECT_SLOTS;self.slot_memory.clear();self.retries=0;return 'Destination please'
        return self._fallback('unrecognized input')
