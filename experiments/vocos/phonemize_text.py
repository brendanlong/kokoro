"""wiki_text.py's paragraphs as Kokoro input ids, read by optimized-kokoro's own pipeline, so a
teacher corpus holds what the server sends the model.

usage: phonemize_text.py PHONEMIZE_BIN TEXT_DIR OUT_JSONL LANG ...

PHONEMIZE_BIN is optimized-kokoro's `cargo build --release -p kokoro-pipeline --example phonemize`
(Japanese needs $KOKORO_UNIDIC). Half of each language's paragraphs are cut into chunks of up to
the pipeline's default length, as a request's later chunks are, and half into chunks of up to 150
ids, about a sentence or two. Each line of OUT_JSONL is {"lang", "ids", "text"}.
"""
import json
import subprocess
import sys

BIN, TEXT, OUT = sys.argv[1:4]
with open(OUT, "w") as out:
    for lang in sys.argv[4:]:
        lines = open(f"{TEXT}/{lang}.txt").read().split("\n")[:-1]
        kept = errors = ids_total = 0
        for part, max_tokens in ((lines[::2], None), (lines[1::2], 150)):
            cmd = [BIN, lang] + ([str(max_tokens)] if max_tokens else [])
            result = subprocess.run(cmd, input="\n".join(part) + "\n", capture_output=True, text=True, check=True)
            for record in map(json.loads, result.stdout.split("\n")[:-1]):
                errors += record["error"] is not None
                for chunk in record["chunks"]:
                    if 10 <= len(chunk["ids"]) <= 510:
                        out.write(json.dumps({"lang": lang, "ids": chunk["ids"], "text": chunk["text"]}, ensure_ascii=False) + "\n")
                        kept += 1
                        ids_total += len(chunk["ids"])
        print(lang, kept, "chunks,", ids_total, "ids,", errors, "paragraphs with an error", flush=True)
