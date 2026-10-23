"""Render docs/REPORT.md to docs/REPORT.pdf with headless Chrome (images and the SVG diagram inlined)."""
import pathlib
import subprocess

import markdown

HERE = pathlib.Path(__file__).parent
md = (HERE / "REPORT.md").read_text()
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
arch = (HERE / "architecture.svg").read_text()
html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{font:11pt/1.45 -apple-system,Helvetica,Arial,sans-serif;max-width:780px;margin:24px auto;color:#1d1d1f}}
h1{{font-size:20pt}} h2{{font-size:14pt;margin-top:22px;border-bottom:1px solid #ddd}} h3{{font-size:12pt}}
table{{border-collapse:collapse;font-size:9.5pt;margin:8px 0}} td,th{{border:1px solid #ccc;padding:3px 6px}}
code,pre{{font-size:9pt;background:#f4f4f4}} pre{{padding:8px;overflow-wrap:anywhere;white-space:pre-wrap}}
svg{{width:100%;height:auto}} img{{max-width:100%}}
</style></head><body>{body}<h2>Architecture diagram</h2>{arch}</body></html>"""
out = HERE / "REPORT.html"
out.write_text(html)
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={HERE / 'REPORT.pdf'}", out.as_uri()], check=True, capture_output=True)
out.unlink()
print(HERE / "REPORT.pdf")
