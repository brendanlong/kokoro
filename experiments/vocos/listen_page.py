"""The listening page's HTML fragment, from listen.py's samples.json (fill it with the html-doc template).

usage: listen_page.py SAMPLES_JSON OUT_HTML [PAGE_JSON]

PAGE_JSON replaces the page's own text: {"labels": {variant: label}, "blind": [reference, other],
"intro": the HTML before the samples}."""
import html
import json
import sys

SAMPLES, OUT = sys.argv[1:3]
PAGE = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
items = json.load(open(SAMPLES))

LABELS = {
    "original": "Original",
    "original_seed1": "Original, another noise seed",
    "lstmfree": "LSTM-free",
    "teacher": "Batch-first (per-frame normalization, Kokoro's decoder layout)",
    "vocos": "Vocos-style decoder, spectral losses only",
}
BLIND = ("original", "vocos")
if PAGE:
    LABELS, BLIND = PAGE["labels"], tuple(PAGE["blind"])

parts = ["""
<style>
.sample { border: 1px solid var(--border); border-radius: 8px; padding: 0.8em 1em; margin: 1.2em 0; }
.sample .meta { color: var(--muted); font-size: 0.9em; }
.sample .text { margin: 0.3em 0 0.7em; }
.clips { display: grid; grid-template-columns: max-content 1fr; gap: 0.4em 1em; align-items: center; }
.clips audio { width: 100%; min-width: 0; height: 2.2em; }
.blind-controls { display: none; gap: 0.6em; align-items: center; margin-top: 0.6em; flex-wrap: wrap; }
body.blind .blind-controls { display: flex; }
body.blind .clip-row:not(.ab) { display: none; }
.verdict { font-weight: 600; }
@media (max-width: 520px) { .clips { grid-template-columns: 1fr; gap: 0.1em; } .clips .label { margin-top: 0.4em; } }
#score { font-weight: 600; }
</style>

"""]
INTRO = """
<p>Each sentence is read by the original Kokoro-82M, the LSTM-free model, the batch-first model (per-frame
normalization in Kokoro's decoder layout, from November 26), and a new Vocos-style decoder. The Vocos-style
decoder replaces Kokoro's 53M-parameter decoder, which upsamples 300-fold through many convolution layers, with
17M parameters of ConvNeXt blocks at 80 frames per second and one inverse STFT; it runs behind the LSTM-free
text side and the batch-first pitch/loudness predictor. This is stage one of its training: spectral losses
only, no adversarial training yet, which in Paradee's experience sounds robotic or buzzy until the adversarial
stage. All clips use the same noise seed except "another noise seed".</p>

<h2 id="numbers">Measurements</h2>
<p>100 held-out sentences of <cite>The Time Machine</cite>, fp32, against the original. Word error rate is
Whisper's transcript against its transcript of the original. The others force the timing to the original's:
mean loudness difference per sound (dB) and spectrogram correlation. "Teacher" here is what the Vocos decoder
was trained to imitate: the LSTM-free text side, the batch-first pitch/loudness predictor and Kokoro's own
decoder.</p>
<table>
<thead><tr><th>Variant</th><th>Word error rate</th><th>Loudness difference</th><th>Spectrogram correlation</th></tr></thead>
<tbody>
<tr><td>Original, another noise seed</td><td>0.11%</td><td>0.09 dB</td><td>0.994</td></tr>
<tr><td>LSTM-free</td><td>0.77%</td><td>0.51 dB</td><td>0.954</td></tr>
<tr><td>Teacher (batch-first predictor, Kokoro's decoder)</td><td>0.83%</td><td>0.64 dB</td><td>0.944</td></tr>
<tr><td>Batch-first (Nov 26)</td><td>0.66%</td><td>1.34 dB</td><td>0.915</td></tr>
<tr><td>Vocos-style, stage one</td><td>0.88%</td><td>1.31 dB</td><td>0.923</td></tr>
</tbody>
</table>
<p>The Vocos-style decoder against its teacher (same inputs): 1.11 dB, 0.942. Training took 37 minutes on the
local GPU (60,000 steps of 16 crops of 1.6 s from 22 hours of teacher audio in all 54 voices, English text).</p>

<h2 id="samples">Samples</h2>
<p><label><input type="checkbox" id="blind"> Blind test</label>: shows only the original and the batch-first model, as A and B in random order, so you can guess which is which before revealing.
<span id="score"></span></p>
"""
parts.append(PAGE.get("intro", INTRO))


for i, it in enumerate(items):
    rows = []
    for key, label in LABELS.items():
        if key not in it["files"]:
            continue
        ab = " ab" if key in BLIND else ""
        rows.append(f'<div class="clip-row{ab}" data-key="{key}" style="display:contents">'
                    f'<span class="label">{label}</span>'
                    f'<audio controls preload="none" src="{html.escape(it["files"][key])}"></audio></div>')
    parts.append(f"""
<div class="sample" id="s{i}">
<div class="meta">{html.escape(it["source"])} · voice {html.escape(it["voice"])}</div>
<div class="text">{html.escape(it["text"])}</div>
<div class="clips">{"".join(rows)}</div>
<div class="blind-controls">Which is the {html.escape(LABELS[BLIND[0]])}?
<button data-guess="A">A</button><button data-guess="B">B</button>
<button data-guess="?">Can't tell</button>
<span class="verdict"></span></div>
</div>""")

parts.append("<script>const KEYS = " + json.dumps(list(LABELS)) + "; const REF = " + json.dumps(BLIND[0]) + ";</script>")
parts.append("""
<script>
let right = 0, answered = 0;
function setBlind(on) {
  document.body.classList.toggle("blind", on);
  right = answered = 0;
  document.getElementById("score").textContent = "";
  for (const s of document.querySelectorAll(".sample")) {
    const clips = s.querySelector(".clips");
    const rows = [...clips.querySelectorAll(".clip-row")];
    const ab = rows.filter(r => r.classList.contains("ab"));
    if (on && Math.random() < 0.5) ab.reverse();
    const order = on ? [...ab, ...rows.filter(r => !r.classList.contains("ab"))] : rows.sort(
      (a, b) => KEYS.indexOf(a.dataset.key) - KEYS.indexOf(b.dataset.key));
    order.forEach(r => clips.appendChild(r));
    ab.forEach((r, j) => {
      const l = r.querySelector(".label");
      l.dataset.real ??= l.textContent;
      l.textContent = on ? "ABCD"[j] : l.dataset.real;
    });
    s.dataset.answered = "";
    s.querySelector(".verdict").textContent = "";
  }
}
document.getElementById("blind").addEventListener("change", e => setBlind(e.target.checked));
for (const b of document.querySelectorAll("[data-guess]")) b.addEventListener("click", () => {
  const s = b.closest(".sample");
  if (s.dataset.answered) return;
  s.dataset.answered = "1";
  const ab = [...s.querySelectorAll(".clip-row.ab")];
  const orig = "ABCD"[ab.findIndex(r => r.dataset.key === REF)];
  const g = b.dataset.guess;
  answered++;
  if (g === orig) right++;
  s.querySelector(".verdict").textContent =
    (g === "?" ? "" : g === orig ? "Right. " : "Wrong. ") + `It was ${orig}.`;
  ab.forEach(r => { const l = r.querySelector(".label"); l.textContent += " = " + l.dataset.real; });
  document.getElementById("score").textContent = `${right} of ${answered} right`;
});
</script>
""")
open(OUT, "w").write("".join(parts))
