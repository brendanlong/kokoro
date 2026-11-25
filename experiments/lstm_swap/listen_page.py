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
    "swap5": "5 of 6 LSTMs replaced",
    "swap6": "All 6 LSTMs replaced",
}

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

<p>Each sentence below is read by the original Kokoro-82M and by two distilled variants, where small
convolution blocks stand in for the model's LSTM layers. All clips use the same noise seed except
"Original, another noise seed", which shows how much two renders of the unchanged model differ on
their own. A variant that differs from the original by about that much is as close as it can get.</p>

<ul>
<li><strong>5 of 6 LSTMs replaced</strong> keeps the text encoder's LSTM. This is the candidate.</li>
<li><strong>All 6 LSTMs replaced</strong> also replaces the text encoder's. It made audible
mispronunciations in the earlier measurements, so it's here for comparison.</li>
</ul>

<h2 id="numbers">Earlier measurements</h2>
<p>From <a href="https://github.com/brendanlong/kokoro/pull/3">brendanlong/kokoro#3</a>, on the RTX 3060 Ti with
torch 2.8. Speed is fp16 and compiled, on 35 sentences of 53-259 phonemes. Quality is fp32, on 100 held-out
sentences of <cite>The Time Machine</cite>. Word error rate is Whisper's transcript of each variant against its
transcript of the original.</p>
<table>
<thead><tr><th>Variant</th><th>Median time per sentence</th><th>Seconds of audio per second</th><th>Word error rate</th><th>Pitch difference</th></tr></thead>
<tbody>
<tr><td>Original</td><td>50.5 ms</td><td>189</td><td>0.1% (another seed)</td><td>0 Hz</td></tr>
<tr><td>5 of 6 replaced</td><td>37.6 ms</td><td>259</td><td>0.55%</td><td>8.7 Hz</td></tr>
<tr><td>All 6 replaced</td><td>36.0 ms</td><td>272</td><td>3.75%</td><td>8.7 Hz</td></tr>
</tbody>
</table>
<p>These clips come from the reference Kokoro code in fp32 at batch 1, not from optimized-kokoro's worker
(fp16, its own convolution kernels, packed batches). The students were trained once, for about 75 minutes,
and their losses were still falling.</p>

<h2 id="samples">Samples</h2>
<p><label><input type="checkbox" id="blind"> Blind test</label>: shows only the original and the
5-replaced variant, as A and B in random order, so you can guess which is which before revealing.
<span id="score"></span></p>
"""]

for i, it in enumerate(items):
    rows = []
    for key, label in LABELS.items():
        ab = " ab" if key in ("original", "swap5") else ""
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

parts.append("""
<script>
const labels = {original: "Original", swap5: "5 of 6 LSTMs replaced"};
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
      (a, b) => ["original", "original_seed1", "swap5", "swap6"].indexOf(a.dataset.key) -
                ["original", "original_seed1", "swap5", "swap6"].indexOf(b.dataset.key));
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
