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
    "swap5": "5 of 6 replaced",
    "swap6_audio": "All 6, text encoder trained on the audio",
    "swap6_context": "All 6, text encoder with wider context",
    "swap6": "All 6, first try",
}
BLIND = ("original", "swap6_audio")

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

<p>Each sentence below is read by the original Kokoro-82M and by distilled variants, where small
convolution blocks stand in for the model's LSTM layers. All clips use the same noise seed except
"Original, another noise seed", which shows how much two renders of the unchanged model differ on
their own. A variant that differs from the original by about that much is as close as it can get.</p>

<ul>
<li><strong>5 of 6 replaced</strong> keeps the text encoder's LSTM (November 1).</li>
<li><strong>All 6, text encoder trained on the audio</strong> (November 25): the same five, plus the first
try's text-encoder replacement fine-tuned so that the audio made from its output matches the audio made from
the LSTM's (a spectrogram loss through the frozen decoder). This is the new candidate.</li>
<li><strong>All 6, text encoder with wider context</strong> (November 25): a text-encoder replacement that sees
about 120 sounds either side instead of about 25, trained the old way. A control: does context fix it?</li>
<li><strong>All 6, first try</strong> (November 1): the one with the obvious loudness changes.</li>
</ul>

<h2 id="numbers">Measurements</h2>
<p>On 100 held-out sentences of <cite>The Time Machine</cite> (99 sounds on average), fp32, against the original
model. Word error rate is Whisper's transcript of each variant against its transcript of the original. Loudness
difference is the mean difference, in decibels, between each input sound's stretch of audio and the original's,
with the timing forced to the original's (sounds more than 40 dB below the loudest are left out). Spectrogram
correlation is also with the timing forced. Pitch and the predicted loudness are the same for all four
replaced variants, since they share the five predictor replacements.</p>
<table>
<thead><tr><th>Variant</th><th>Word error rate</th><th>Loudness difference</th><th>Spectrogram correlation</th></tr></thead>
<tbody>
<tr><td>Original, another noise seed</td><td>0.11%</td><td>0.09 dB</td><td>0.994</td></tr>
<tr><td>5 of 6 replaced</td><td>0.55%</td><td>0.40 dB</td><td>0.957</td></tr>
<tr><td><strong>All 6, trained on the audio</strong></td><td><strong>0.77%</strong></td><td><strong>0.51 dB</strong></td><td><strong>0.954</strong></td></tr>
<tr><td>All 6, wider context</td><td>1.71%</td><td>2.04 dB</td><td>0.908</td></tr>
<tr><td>All 6, first try</td><td>3.75%</td><td>2.23 dB</td><td>0.897</td></tr>
</tbody>
</table>
<p>Speed, from <a href="https://github.com/brendanlong/kokoro/pull/3">brendanlong/kokoro#3</a> (fp16, compiled,
RTX 3060 Ti, 35 sentences of 53-259 sounds): the original took 50.5 ms per sentence (189 seconds of audio per
second), 5 of 6 replaced 37.6 ms (259), all 6 replaced 36.0 ms (272). The new all-6 variants have the same shape
as the first try (the wider-context one has two more blocks), so they should be about as fast; not measured.</p>
<p>These clips come from the reference Kokoro code in fp32 at batch 1, not from optimized-kokoro's worker
(fp16, its own convolution kernels, packed batches). Each variant was trained once. The audio-trained one was
fine-tuned for 70 minutes on sentences of up to 100 sounds (longer ones didn't fit in GPU memory beside the
desktop), and its loss was still falling slowly.</p>

<h2 id="samples">Samples</h2>
<p><label><input type="checkbox" id="blind"> Blind test</label>: shows only the original and the
all-6 variant trained on the audio, as A and B in random order, so you can guess which is which before revealing.
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
