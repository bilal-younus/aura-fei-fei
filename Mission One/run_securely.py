"""Use a key in process memory without saving or displaying it."""
import getpass, os, runpy
from pathlib import Path
if __name__ == '__main__':
    provider = os.getenv('AURA_PROVIDER', 'gemini')
    key_name = 'GEMINI_API_KEY' if provider == 'gemini' else 'OPENAI_API_KEY'
    if not os.getenv(key_name):
        os.environ[key_name] = getpass.getpass(f'{provider} key (hidden): ').strip()
    if not os.environ[key_name]:
        raise SystemExit('No key entered.')
    runpy.run_path(str(Path(__file__).with_name('04_dialogic_engine.py')), run_name='__main__')
