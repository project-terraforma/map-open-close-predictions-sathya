"""
Test server for vision-based business detection.
Upload an image → model tells you what text and logos/brands it sees.

Usage: python scripts/test_ocr.py
Then open http://localhost:5001
"""

import os
import sys
import base64
import tempfile
from flask import Flask, request

# Add scripts dir to path so we can import ocr_model
sys.path.insert(0, os.path.dirname(__file__))
from ocr_model import analyze_image

app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TERRAFORMA - Vision Detection Test</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #0f172a; color: #e2e8f0; min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 40px 20px;
  }
  h1 { font-size: 28px; margin-bottom: 8px; color: #fff; }
  .subtitle { color: #64748b; font-size: 14px; margin-bottom: 32px; }
  .card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 32px; width: 100%%; max-width: 640px;
    margin-bottom: 24px;
  }
  label { display: block; font-size: 13px; color: #94a3b8; margin-bottom: 6px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
  input[type="file"] { margin-bottom: 20px; color: #94a3b8; }
  button {
    width: 100%%; padding: 14px; border: none; border-radius: 10px;
    background: linear-gradient(135deg, #3b82f6, #2563eb); color: #fff;
    font-size: 16px; font-weight: 700; cursor: pointer; transition: 0.2s;
  }
  button:hover { background: linear-gradient(135deg, #60a5fa, #3b82f6); }
  .result-card {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 24px; width: 100%%; max-width: 640px;
  }
  .verdict {
    font-size: 22px; font-weight: 800; text-align: center;
    padding: 16px; border-radius: 12px; margin-bottom: 20px;
  }
  .verdict.most-likely { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
  .verdict.likely { background: rgba(250,204,21,0.15); color: #facc15; border: 1px solid rgba(250,204,21,0.3); }
  .verdict.not-found { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
  .preview { width: 100%%; border-radius: 12px; margin-bottom: 20px; }
  .section-label { font-size: 12px; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; margin-bottom: 8px; margin-top: 20px; }
  .text-list { list-style: none; padding: 0; }
  .text-list li {
    padding: 8px 12px; margin-bottom: 4px; border-radius: 8px;
    background: rgba(255,255,255,0.05); font-family: monospace; font-size: 14px;
    display: flex; justify-content: space-between; align-items: center;
  }
  .text-list .conf { color: #64748b; font-size: 12px; }
  .brand-bar-container { margin-bottom: 6px; }
  .brand-bar-label {
    display: flex; justify-content: space-between; font-size: 13px;
    margin-bottom: 3px;
  }
  .brand-bar-label .name { color: #e2e8f0; font-weight: 600; }
  .brand-bar-label .score { color: #64748b; }
  .brand-bar {
    height: 8px; border-radius: 4px; background: rgba(255,255,255,0.08);
    overflow: hidden;
  }
  .brand-bar-fill {
    height: 100%%; border-radius: 4px; transition: width 0.3s;
  }
  .brand-bar-fill.top { background: linear-gradient(90deg, #34d399, #10b981); }
  .brand-bar-fill.other { background: rgba(148,163,184,0.4); }
  .meta { display: flex; justify-content: space-between; margin-top: 16px; font-size: 13px; color: #94a3b8; }
  a { color: #60a5fa; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .info-box { padding: 10px 14px; border-radius: 8px; background: rgba(255,255,255,0.05); font-size: 13px; color: #94a3b8; margin-bottom: 12px; }
  .info-box strong { color: #e2e8f0; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 600px) { .two-col { grid-template-columns: 1fr; } }
</style>
</head>
<body>
  <h1>TERRAFORMA: Vision Detection</h1>
  <p class="subtitle">Upload a street-level image — the model identifies text and brand logos automatically</p>
  %%CONTENT%%
</body>
</html>"""

UPLOAD_FORM = """
  <div class="card">
    <form method="POST" action="/analyze" enctype="multipart/form-data">
      <label>Street-Level Image</label>
      <input type="file" name="image" accept="image/*" required>
      <button type="submit">Analyze Image</button>
    </form>
  </div>
"""


@app.route("/")
def index():
    return HTML_TEMPLATE.replace("%%CONTENT%%", UPLOAD_FORM)


@app.route("/analyze", methods=["POST"])
def analyze():
    import traceback as tb
    tmp_path = None
    try:
        image_file = request.files.get("image")
        if not image_file:
            return HTML_TEMPLATE.replace("%%CONTENT%%", '<p style="color:#f87171;">Please upload an image.</p>')

        # Save uploaded image to temp file
        suffix = os.path.splitext(image_file.filename)[1] or ".jpg"
        tmp_path = tempfile.mktemp(suffix=suffix)
        image_file.save(tmp_path)
        print(f"Saved image to: {tmp_path}", flush=True)

        # Run full analysis (OCR + CLIP)
        result = analyze_image(tmp_path)
        print(f"Result: verdict={result['verdict']}, texts={len(result['texts'])}, top_brand={result['top_brand']}", flush=True)

        # Encode image for preview
        with open(tmp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        img_src = f"data:image/{suffix.lstrip('.')};base64,{img_b64}"

        # Build detected text list
        text_items = ""
        for t in result["texts"]:
            text_items += f'<li><span>{t["text"]}</span><span class="conf">{t["confidence"]}</span></li>'
        if not text_items:
            text_items = '<li style="color:#64748b;">No text detected</li>'

        # Build brand recognition bars (cosine similarity 0-0.4 scale)
        brand_bars = ""
        for i, b in enumerate(result["brands"]):
            # Scale bar: 0.4 cosine sim = 100% bar width
            bar_width = max(2, min(100, (b["score"] / 0.4) * 100))
            fill_class = "top" if i == 0 else "other"
            brand_bars += f"""
            <div class="brand-bar-container">
                <div class="brand-bar-label">
                    <span class="name">{'&#9733; ' if i == 0 else ''}{b['name']}</span>
                    <span class="score">{b['score']:.3f}</span>
                </div>
                <div class="brand-bar">
                    <div class="brand-bar-fill {fill_class}" style="width:{bar_width}%%"></div>
                </div>
            </div>"""

        # Verdict display
        verdict = result["verdict"]
        if verdict == "most_likely_open":
            verdict_class = "most-likely"
            verdict_text = f'DETECTED: {result["best_match"]}'
        elif verdict == "likely_open":
            verdict_class = "likely"
            verdict_text = f'LIKELY: {result["best_match"]}'
        else:
            verdict_class = "not-found"
            verdict_text = "NO KNOWN BRAND DETECTED"

        content = f"""
        <div class="result-card">
            <div class="verdict {verdict_class}">{verdict_text}</div>
            <img src="{img_src}" class="preview" alt="Uploaded image">

            <div class="two-col">
                <div>
                    <div class="section-label">Brand Recognition (CLIP)</div>
                    {brand_bars}
                </div>
                <div>
                    <div class="section-label">Text Detection (OCR)</div>
                    <ul class="text-list">{text_items}</ul>
                </div>
            </div>

            <div class="section-label">Raw OCR Output</div>
            <div class="info-box" style="font-family: monospace; white-space: pre-wrap;">{result['all_text'] or '(no text detected)'}</div>

            <div class="meta">
                <span>Confidence: <strong>{result['confidence']}</strong></span>
                <span>Text fragments: <strong>{len(result['texts'])}</strong></span>
            </div>

            <div style="margin-top: 20px; text-align: center;">
                <a href="/">Analyze another image</a>
            </div>
        </div>
        """
        return HTML_TEMPLATE.replace("%%CONTENT%%", content)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        tb.print_exc()
        err_msg = f'<p style="color:#f87171;">Error: {str(e)}</p><pre style="color:#94a3b8;font-size:12px;">{tb.format_exc()}</pre>'
        return HTML_TEMPLATE.replace("%%CONTENT%%", f'<div class="card">{err_msg}<a href="/">Try again</a></div>')
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("Starting vision detection test server...")
    print("Open http://localhost:5001 in your browser")
    print("Upload a street-level image to see what the model detects\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
