"""Convert mission-03-doc.md to mission-03-doc.pdf with headless Chrome (browser Print-to-PDF). Requires markdown."""
import re
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1d2b33; }
h1 { color: #153B50; font-size: 18pt; border-bottom: 2px solid #153B50; padding-bottom: 4px; }
h2 { color: #153B50; font-size: 13pt; margin-top: 18px; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 8.5pt; page-break-inside: avoid; }
th { background: #DFECF1; }
th, td { border: 1px solid #BCCDD5; padding: 4px 6px; vertical-align: top; }
code { font-family: Menlo, Courier, monospace; font-size: 8.5pt; background: #f2f5f7; }
pre { background: #f2f5f7; padding: 8px; font-size: 7.5pt; line-height: 1.3; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; }
img { max-width: 100%; }
a { color: #17698B; }
hr { border: none; border-top: 1px solid #BCCDD5; }
"""

source = (ROOT / "mission-03-doc.md").read_text()
# GitHub starts a list directly after a paragraph line; python-markdown needs a blank line first.
source = re.sub(r"^((?!```)[^\n\-|].*)\n(- )", r"\1\n\n\2", source, flags=re.M)
html = markdown.markdown(source, extensions=["tables", "fenced_code"])
html = html.replace('src="evidence/', f'src="{(ROOT / "evidence").as_uri()}/')
with tempfile.TemporaryDirectory() as tmp:
    page = Path(tmp) / "mission-03-doc.html"
    page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
                    f"<body>{html}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={ROOT / 'mission-03-doc.pdf'}", page.as_uri()],
                   check=True, capture_output=True)
print("Built mission-03-doc.pdf")
