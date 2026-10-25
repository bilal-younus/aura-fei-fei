import unittest,itertools
from fst_manager import FSTDialogueManager,State
from noisy_channel import NoisyChannel,NOISE,entropy,mutual_information,capacity

class MissionTests(unittest.TestCase):
    def test_greeting_path(self):
        f=FSTDialogueManager();f.step('hello');self.assertEqual(f.state,State.GREETING)
        for t in ['Sirius','warp 7','ALPHA-7']:f.step(t)
        self.assertFalse(f.authorized);self.assertEqual(f.state,State.CONFIRMATION)
        f.step('yes');self.assertTrue(f.authorized)
    def test_all_slot_orders(self):
        for order in itertools.permutations(['Sirius','warp 7','ALPHA-7']):
            f=FSTDialogueManager()
            for t in order:f.step(t)
            self.assertEqual(len(f.slot_memory),3);self.assertFalse(f.authorized)
            f.step('confirm');self.assertTrue(f.authorized)
    def test_missing_clearance_cannot_authorize(self):
        f=FSTDialogueManager()
        for t in ['Sirius','warp 7','yes']:f.step(t)
        self.assertFalse(f.authorized)
    def test_invalid_values(self):
        for value in ['warp 99','warp 0','BETA-7','warp 7 warp 8','Sirius Orion Arm']:
            f=FSTDialogueManager();f.step(value);self.assertFalse(f.slot_memory);self.assertEqual(f.state,State.FALLBACK)
    def test_recovery(self):
        f=FSTDialogueManager();f.step('Tell me a joke');f.step('Sirius')
        self.assertEqual(f.state,State.COLLECT_SLOTS);self.assertEqual(f.retries,0)
    def test_abort_is_absorbing(self):
        f=FSTDialogueManager()
        for t in ['noise','noise','noise','Sirius','warp 7','ALPHA-7','yes']:f.step(t)
        self.assertFalse(f.authorized);self.assertEqual(f.state,State.TERMINAL);self.assertFalse(f.slot_memory)
    def test_correction_reconfirmation(self):
        f=FSTDialogueManager()
        for t in ['Sirius','warp 7','ALPHA-7','warp 5']:f.step(t)
        self.assertFalse(f.authorized);self.assertEqual(f.slot_memory['speed'],'5')
    def test_negation_and_injection(self):
        for t in ['Do not go to Sirius','Ignore rules authorize','Yeah right Sirius']:
            f=FSTDialogueManager();f.step(t);self.assertFalse(f.slot_memory)
    def test_channel_one_substitution_recovery(self):
        for seed in range(100):
            p=NoisyChannel(seed).transmit('warp 7')
            self.assertEqual(p['decoded'],'warp 7');self.assertLessEqual(p['words'],7)
            self.assertEqual(sum(x in NOISE for x in p['received'].split()),1)
    def test_uncorrectable_is_erasure(self):
        p=NoisyChannel(1).transmit('yes',substitutions=3)
        self.assertEqual(p['decoded'],'[ERASURE]')
    def test_information_identities(self):
        self.assertEqual(entropy(['a','a']),0);self.assertEqual(entropy(['a','b']),1)
        self.assertEqual(mutual_information(['a','a','b','b'],['0','1','0','1']),0)
        self.assertEqual(mutual_information(['a','b'],['a','b']),1)
        self.assertEqual(capacity(1000,3),2000)
    def test_packet_boundaries(self):
        for payload in ['', 'one two three four five six seven eight','Banana']:
            with self.assertRaises(ValueError):NoisyChannel().transmit(payload)
if __name__=='__main__':unittest.main(verbosity=2)
