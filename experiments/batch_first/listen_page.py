"""The listening page's HTML fragment, from listen.py's samples.json (fill it with the html-doc template).

usage: listen_page.py SAMPLES_JSON OUT_HTML"""
import html
import json
import sys

SAMPLES, OUT = sys.argv[1:3]
items = json.load(open(SAMPLES))

LABELS = {
    "original": "Original",
    "original_seed1": "Original, another noise seed",
    "lstmfree": "LSTM-free (today's candidate, the teacher)",
    "batchfirst": "Batch-first (per-frame normalization)",
}
BLIND = ("original", "batchfirst")

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

<p>Each sentence is read by the original Kokoro-82M; by the LSTM-free model (its six LSTMs replaced by
distilled convolution blocks); and by a batch-first model, the LSTM-free one with per-frame normalization in its
pitch/loudness predictor and decoder instead of Kokoro's per-sentence normalization. Batch-first is what would
let sentences be batched with no special code. All use the same noise seed except "another noise seed", which
shows how much two renders of the original differ on their own.</p>

<h2 id="numbers">Measurements</h2>
<p>100 held-out sentences of <cite>The Time Machine</cite>, fp32, against the original. Word error rate is
Whisper's transcript of each against its transcript of the original. The others force the timing to the
original's: mean loudness difference per sound (dB), spectrogram correlation and pitch difference on voiced
frames.</p>
<table>
<thead><tr><th>Variant</th><th>Word error rate</th><th>Loudness difference</th><th>Spectrogram correlation</th><th>Pitch difference</th></tr></thead>
<tbody>
<tr><td>Original, another noise seed</td><td>0.11%</td><td>0.09 dB</td><td>0.994</td><td>0 Hz</td></tr>
<tr><td>LSTM-free</td><td>0.77%</td><td>0.51 dB</td><td>0.954</td><td>8.7 Hz</td></tr>
<tr><td>Batch-first</td><td>0.66%</td><td>1.34 dB</td><td>0.915</td><td>14.9 Hz</td></tr>
</tbody>
</table>
<p>Batch-first against its own teacher (LSTM-free), same timing: 1.30 dB, 0.921, 13.1 Hz. It was trained for
about 2.5 hours on the local GPU (2,000 steps block by block, then 6,000 end to end with a spectrogram loss),
on sentences of up to 64 sounds, with no adversarial training yet; its loss was still falling slowly.</p>

<h2 id="samples">Samples</h2>
<p><label><input type="checkbox" id="blind"> Blind test</label>: shows only the original and the batch-first model, as A and B in random order, so you can guess which is which before revealing.
<span id="score"></span></p>
"""]

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
<div class="blind-controls">Which is the original?
<button data-guess="A">A</button><button data-guess="B">B</button>
<button data-guess="?">Can't tell</button>
<span class="verdict"></span></div>
</div>""")

parts.append("<script>const KEYS = " + json.dumps(list(LABELS)) + ";</script>")
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
  const orig = "ABCD"[ab.findIndex(r => r.dataset.key === "original")];
  const g = b.dataset.guess;
  answered++;
  if (g === orig) right++;
  s.querySelector(".verdict").textContent =
    (g === "?" ? "" : g === orig ? "Right. " : "Wrong. ") + `The original was ${orig}.`;
  ab.forEach(r => { const l = r.querySelector(".label"); l.textContent += " = " + l.dataset.real; });
  document.getElementById("score").textContent = `${right} of ${answered} right`;
});
</script>
""")
open(OUT, "w").write("".join(parts))
