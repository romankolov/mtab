#!/usr/bin/env python3
"""Run the CLI across its whole surface and build a browsable gallery.

Every artifact in `samples/showcase/output/` comes from one `mtab` invocation
whose exact command line is recorded next to it in `index.html`. This script
only orchestrates and formats — it never fabricates note data, so whatever the
model produced is what you see.

Usage:
    python samples/showcase/generate_gallery.py \
        --binary target/release/mtab --audio samples/showcase/audio/sample.wav \
        --weights weights/muscriptor-large.safetensors --size large --out samples/showcase/output

Requires the CC BY-NC 4.0 weights; see samples/showcase/README.md.
"""

import argparse
import html
import json
import os
import subprocess
import sys
import time

# One entry per demonstrable capability of the CLI. `--format` accepts a single
# format; for multi-format output the extension of each -o path decides.
DEMOS = [
    {
        "id": "full",
        "title": "Everything at once",
        "why": "The happy path from the acceptance criteria: one command, MIDI + Guitar Pro + MusicXML + ASCII tab + the intermediate IR.",
        # No --format here: with several -o paths the format comes from each
        # extension, and --format takes exactly one value.
        "args": ["--tuning", "auto", "--capo", "auto", "--dynamics", "--techniques",
                 "--chords", "--emit-ir", "{out}/full.ir.json"],
        "outputs": ["full.gp5", "full.musicxml", "full.mid", "full.txt"],
    },
    {
        "id": "strict",
        "title": "Greedy decoding, strict EOS",
        "why": "The deterministic default: no sampling, fail loudly if a chunk never emits EOS. Same flags twice must give identical bytes.",
        "args": ["--seed", "7", "--strict-eos", "--format", "musicxml"],
        # A dense 5-second chunk needs more than the default token budget here,
        # so this demo is meant to end in a refusal rather than a truncated tab.
        "expect": "failure",
        "note": "Expected: strict EOS refuses a chunk that runs past the token "
                "budget instead of silently truncating the notes.",
        "outputs": ["strict.musicxml"],
    },
    {
        "id": "beam",
        "title": "Beam search",
        "why": "Wider search over the same chunk, at a multiple of the decode cost.",
        "args": ["--beam-size", "4", "--format", "midi"],
        "outputs": ["beam.mid"],
    },
    {
        "id": "sampled",
        "title": "Seeded sampling",
        "why": "Temperature sampling is deterministic given --seed, so a sample is reproducible without being greedy.",
        "args": ["--temperature", "0.9", "--seed", "42", "--format", "midi"],
        "outputs": ["sampled.mid"],
    },
    {
        "id": "tuning",
        "title": "Fixed tuning and capo",
        "why": "Skip detection and fretting for a drop-D guitar with a capo at fret 2.",
        "args": ["--tuning", "drop-d", "--capo", "2", "--format", "gp5"],
        "outputs": ["tuning.gp5"],
    },
    {
        "id": "raw",
        "title": "Unquantized timing",
        "why": "Raw model timing with no grid at all, for rubato passages that a fixed grid would flatten.",
        "args": ["--grid", "off", "--format", "midi"],
        "outputs": ["raw.mid"],
    },
    {
        "id": "dynamic",
        "title": "Detected tempo, not a fixed grid",
        "why": "Beat tracking instead of a constant BPM: MIDI keeps the rubato rather than snapping to a grid.",
        "args": ["--experimental-beat-tracking", "--format", "midi"],
        "outputs": ["dynamic.mid"],
    },
    {
        "id": "jsonl",
        "title": "Machine-readable notes",
        "why": "The same decode as JSON Lines, one NoteEvent per line, for downstream tooling.",
        "args": ["--format", "jsonl"],
        "outputs": ["jsonl.jsonl"],
    },
]

INSPECTS = [
    ("models", "Model variants", ["--models"]),
    ("tunings", "Known tunings", ["detect-tuning", "--list"]),
]


def run(binary, argv, log):
    started = time.time()
    proc = subprocess.run([binary, *argv], capture_output=True, text=True)
    elapsed = time.time() - started
    log.write(f"\n=== mtab {' '.join(argv)}  ({elapsed:.1f}s, exit {proc.returncode})\n")
    log.write(proc.stdout)
    if proc.stderr:
        log.write("--- stderr ---\n")
        log.write(proc.stderr)
    return proc, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="target/release/mtab")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--size", default="large")
    parser.add_argument("--out", default="samples/showcase/output")
    parser.add_argument("--only", help="comma-separated demo ids to run")
    parser.add_argument("--max-gen-len", default="2000")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    wanted = set(args.only.split(",")) if args.only else None
    results = []

    with open(os.path.join(out, "commands.log"), "w") as log:
        # Informational commands first: they need no weights.
        for demo_id, title, argv in INSPECTS:
            proc, elapsed = run(args.binary, argv, log)
            results.append({
                "id": demo_id, "title": title, "why": "Built-in catalogue, no weights needed.",
                "command": f"mtab {' '.join(argv)}", "status": proc.returncode,
                "seconds": elapsed, "dir": None, "text": proc.stdout, "outputs": [],
            })

        for demo in DEMOS:
            if wanted and demo["id"] not in wanted:
                continue
            stem = demo["id"]
            outputs = [os.path.join(out, o) for o in demo["outputs"]]
            argv = ["transcribe", args.audio, "--size", args.size,
                    "--weights", os.path.abspath(args.weights),
                    "--max-gen-len", args.max_gen_len,
                    "-o", ",".join(outputs), *demo["args"]]
            argv = [a.format(out=out) for a in argv]
            proc, elapsed = run(args.binary, argv, log)
            produced = [p for p in outputs if os.path.exists(p)]
            for path in outputs:
                if os.path.exists(path) and os.path.getsize(path) == 0:
                    os.remove(path)
            results.append({
                "id": stem, "title": demo["title"], "why": demo["why"],
                "expect": demo.get("expect", "ok"), "note": demo.get("note", ""),
                "command": f"mtab {' '.join(argv)}", "status": proc.returncode,
                "signal": (-proc.returncode) if proc.returncode < 0 else None,
                "seconds": elapsed, "dir": out,
                "text": (proc.stderr or "") if proc.returncode else "",
                "outputs": [os.path.basename(p) for p in produced if os.path.exists(p)],
            })
            print(f"{stem:8s} exit {proc.returncode}  {elapsed:6.1f}s  "
                  f"{len(produced)} file(s)", flush=True)

        # Determinism check (§6.14.6): the same command twice, compared byte for byte.
        a = os.path.join(out, "det_a.musicxml")
        b = os.path.join(out, "det_b.musicxml")
        base = ["transcribe", args.audio, "--size", args.size,
                "--weights", os.path.abspath(args.weights), "--seed", "11",
                "--max-gen-len", args.max_gen_len]
        run(args.binary, [*base, "-o", a], log)
        run(args.binary, [*base, "-o", b], log)
        identical = (os.path.exists(a) and os.path.exists(b)
                     and open(a, "rb").read() == open(b, "rb").read())
        for path in (a, b):
            if os.path.exists(path):
                os.remove(path)
        results.append({
            "id": "determinism", "title": "Determinism",
            "why": "Same audio, weights and --seed, run twice: byte-identical output is required by §6.14.6.",
            "command": f"mtab {' '.join([*base, '-o', '<a>'])}  (twice)",
            "status": 0 if identical else 1, "seconds": None, "dir": None,
            "text": "identical" if identical else "OUTPUTS DIFFER", "outputs": [],
        })

        # Error-path demos: what the product says when it cannot help (§6.4).
        bad = os.path.join(out, "not-audio.wav")
        with open(bad, "wb") as handle:
            handle.write(b"not audio\n")
        proc, elapsed = run(args.binary, ["transcribe", bad, "--size", args.size,
                                          "--weights", os.path.abspath(args.weights)], log)
        results.append({
            "id": "error", "title": "Actionable failure", "expect": "failure", "note": "",
            "why": "A malformed audio file: the error names the problem and the next step, as §6.4 requires.",
            "expect": "failure",
            "command": f"mtab transcribe not-audio.wav ...", "status": proc.returncode,
            "seconds": elapsed, "dir": None,
            "text": (proc.stderr or proc.stdout).strip(), "outputs": [],
        })
        os.remove(bad)

    summary = {"size": args.size, "audio": os.path.basename(args.audio),
               "weights": os.path.basename(args.weights), "demos": results}
    with open(os.path.join(out, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    render(out, summary)
    print(f"gallery written to {os.path.join(out, 'index.html')}")


CSS = """
body{font:15px/1.55 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     margin:0;padding:2.5rem 1.5rem;max-width:60rem;color:#1a1a1a;background:#fafafa}
h1{margin:0 0 .3rem;font-size:1.9rem;letter-spacing:-.02em}
.sub{color:#666;margin:0 0 2rem}
.card{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:1.2rem 1.35rem;
      margin:0 0 1.1rem;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.card h2{margin:0 0 .35rem;font-size:1.1rem}
.why{color:#555;margin:0 0 .8rem}
code{background:#f3f3f3;border-radius:4px;padding:.15rem .35rem;font-size:.86em}
pre{background:#1e1e1e;color:#e8e8e8;border-radius:7px;padding:.85rem;overflow-x:auto;
    font-size:.82rem;margin:.6rem 0}
.badge{display:inline-block;font-size:.74rem;font-weight:600;padding:.16rem .5rem;
       border-radius:99px;margin-left:.5rem;vertical-align:middle}
.ok{background:#e3f6e5;color:#116b25}.bad{background:#fde8e8;color:#9b1c1c}
.expected{background:#fff3d6;color:#8a5a00}
.note{color:#8a5a00;background:#fffaf0;border-left:3px solid #f0c36d;
      padding:.45rem .7rem;margin:0 0 .8rem;font-size:.9rem}
.files{display:flex;flex-wrap:wrap;gap:.45rem;margin:.7rem 0 0}
.files a{display:inline-block;background:#eef3fb;color:#1c4fd8;text-decoration:none;
         border-radius:6px;padding:.32rem .6rem;font-size:.85rem}
.files a:hover{background:#dde7fa}
table{border-collapse:collapse;font-size:.88rem}
td{padding:.15rem .8rem .15rem 0;vertical-align:top}
.k{color:#777}
audio{width:100%;margin:.5rem 0 0}
"""


def render(out, summary):
    rows = []
    for demo in summary["demos"]:
        ok = demo["status"] == 0
        expected_failure = demo.get("expect") == "failure"
        if demo.get("signal"):
            # Negative return codes are signals, not exit codes: say so rather
            # than printing a number nobody recognises.
            label = f"killed by signal {demo['signal']}"
        else:
            label = f"exit {demo['status']}"
        if ok and not expected_failure:
            badge = '<span class="badge ok">ok</span>'
            note = demo.get("note") or ""
        elif not ok and expected_failure:
            # Not hidden and not dressed up as success: a refusal like this is
            # the product working as designed.
            badge = f'<span class="badge expected">{label}, as designed</span>'
            note = demo.get("note") or ""
        elif ok and expected_failure:
            # The demo was meant to be refused and was not. Say that plainly:
            # the label would otherwise read "exit 0" on a red card.
            badge = '<span class="badge bad">expected a refusal, got success</span>'
            note = "This run did not exercise the refusal path."
        else:
            badge = f'<span class="badge bad">{label}</span>'
            note = demo.get("note") or ""
        timing = f' &middot; {demo["seconds"]:.1f}s' if demo["seconds"] else ""
        files = ""
        if demo["outputs"]:
            links = "".join(
                f'<a href="{html.escape(name)}">{html.escape(name)}</a>'
                for name in demo["outputs"])
            files = f'<div class="files">{links}</div>'
        text = ""
        if demo["text"]:
            text = f"<pre>{html.escape(demo['text'].strip())}</pre>"
        rows.append(f"""
<div class="card">
  <h2>{html.escape(demo['title'])}{badge}</h2>
  <p class="why">{html.escape(demo['why'])}{timing}</p>
  {f'<p class="note">{html.escape(note)}</p>' if note else ''}
  <pre>$ {html.escape(demo['command'])}</pre>
  {text}
  {files}
</div>""")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mtab sample gallery</title><style>{CSS}</style></head><body>
<h1>mtab &mdash; what the CLI can do</h1>
<p class="sub">Every artifact below was produced by the command shown above it,
using the <strong>{html.escape(summary['size'])}</strong> MuScriptor checkpoint on
<code>{html.escape(summary['audio'])}</code>. Weights are CC BY-NC 4.0 and are
never bundled; the workflow downloads them with a repository secret.</p>
<table>
<tr><td class="k">model</td><td>{html.escape(summary['size'])}</td></tr>
<tr><td class="k">weights</td><td>{html.escape(summary['weights'])}</td></tr>
<tr><td class="k">input</td><td>{html.escape(summary['audio'])}</td></tr>
</table>
{"".join(rows)}
</body></html>"""
    with open(os.path.join(out, "index.html"), "w") as handle:
        handle.write(page)


if __name__ == "__main__":
    main()
