"""Offline control-flow tests. These are NOT live LLM evaluation evidence."""
import importlib.util
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
spec = importlib.util.spec_from_file_location('engine', Path(__file__).with_name('04_dialogic_engine.py'))
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)


def turn(status='REFINE', premise='Safer premise'):
    return f'QUESTIONS: Which constraints?\nCOUNTEREXAMPLE: Harmful command.\nTEST: Obedience may harm.\nREFINED PREMISE: {premise}\nSTATUS: {status}'


class EngineTests(unittest.TestCase):
    def engine(self, replies):
        obj = e.AuraEngine(client=object())
        self.prompts = []
        def ask(system, user):
            self.prompts.append(user)
            return next(replies)
        obj.ask = ask
        return obj

    def test_budget_ends_in_aporia(self):
        result = self.engine(iter([turn(premise=f'P{i}') for i in range(3)])).socratic_loop('Initial')
        self.assertEqual(result.status, 'aporia')
        self.assertEqual(len(result.turns), 3)

    def test_no_progress_ends_in_aporia(self):
        result = self.engine(iter([turn(premise='Initial')])).socratic_loop('Initial')
        self.assertEqual(result.status, 'aporia')
        self.assertEqual(len(result.turns), 1)

    def test_resolution(self):
        result = self.engine(iter([turn('RESOLVED_WITH_CAVEATS')])).socratic_loop('Initial')
        self.assertEqual(result.status, 'resolved_with_caveats')

    def test_explicit_aporia(self):
        self.assertEqual(self.engine(iter([turn('APORIA')])).socratic_loop('Initial').status, 'aporia')

    def test_bad_status_rejected(self):
        with self.assertRaises(ValueError):
            self.engine(iter([turn('MAYBE')])).socratic_loop('Initial')

    def test_missing_sections_rejected(self):
        with self.assertRaises(ValueError):
            self.engine(iter(['STATUS: APORIA'])).socratic_loop('Initial')

    def test_history_is_carried(self):
        self.engine(iter([turn(), turn('APORIA')])).socratic_loop('Initial')
        self.assertIn('Harmful command', self.prompts[1])

    def test_debate_passes_rebuttal_to_mediator(self):
        text = self.engine(iter(['benefits', 'risks', 'updated case', 'conditional conclusion'])).galilean_debate('Topic')
        self.assertIn('updated case', self.prompts[3])
        self.assertIn('REBUTTAL', text)

    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self.engine(iter([])).socratic_loop(' ')

    def test_api_empty_and_truncated(self):
        for content, reason in [('', 'stop'), ('partial', 'length')]:
            response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=reason, message=SimpleNamespace(content=content, refusal=None))])
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
            with self.assertRaises(RuntimeError):
                e.AuraEngine(client=client).ask('system', 'user')

if __name__ == '__main__':
    unittest.main()
