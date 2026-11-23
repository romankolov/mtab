#!/usr/bin/env python3
"""Download and convert a gated MuScriptor checkpoint for the sample workflow.

This is the "weights acquisition" step of PROMPT.md §10.1 made explicit: the
download happens in CI, on the user's own credentials, into a scratch directory
that is never committed. `mtab` itself never downloads weights.

Why a conversion step at all: upstream ships a slightly different safetensors
layout than the loader's fast path. This script rewrites the file into the
layout `mtab` reads directly, dropping the conditioning tensors the CLI never
needs (the mel filterbank is recomputed by the frontend, so keeping it would
only add 8 MB of noise to the artifact).

Usage:
    python samples/showcase/fetch_weights.py --size large --out weights/
"""

import argparse
import json
import os
import struct
import sys
import urllib.request

REPOS = {
    "small": (103_000_000, 14, 768),
    "medium": (307_000_000, 24, 1024),
    "large": (1_400_000_000, 48, 1536),
}

# Conditioning tensors the CLI does not need. `self_wav.output_proj` *is* needed
# (it projects the mel stream); the stored filterbank and window are not, because
# MelFrontend::upstream computes them itself.
DROPPED_PREFIXES = (
    "condition_provider.conditioners.self_wav.mel_spec_transform.",
)


def hf_get(url: str, token: str, byte_range: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    if byte_range:
        request.add_header("Range", f"bytes={byte_range}")
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def download(url: str, token: str, destination: str) -> None:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, timeout=600) as response, open(destination, "wb") as out:
        total = int(response.headers.get("X-Linked-Size") or response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 22):
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r  {done / 1e9:6.2f} / {total / 1e9:.2f} GB  {pct:5.1f}%", end="", flush=True)
        print()


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return header, length


def convert(source: str, destination: str, repo: str) -> tuple[int, int]:
    """Rewrite the checkpoint with sorted, contiguous tensor data."""
    header, header_len = read_header(source)
    keep = {k: v for k, v in header.items() if not k.startswith(DROPPED_PREFIXES)}
    dropped = len(header) - len(keep)
    base = 8 + header_len

    new_header = {}
    offset = 0
    with open(source, "rb") as src, open(destination, "wb") as out:
        out.write(b"\0" * 8)  # header length written last
        for name in sorted(keep):
            info = keep[name]
            begin, end = info["data_offsets"]
            new_header[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": [offset, offset + (end - begin)],
            }
            offset += end - begin
        encoded = json.dumps(new_header, separators=(",", ":")).encode()
        out.write(encoded)
        for name in sorted(keep):
            begin, end = keep[name]["data_offsets"]
            src.seek(base + begin)
            remaining = end - begin
            while remaining:
                block = src.read(min(remaining, 1 << 22))
                if not block:
                    sys.exit(f"{source} truncated reading {name}")
                out.write(block)
                remaining -= len(block)
        end = out.tell()
        out.seek(0)
        out.write(struct.pack("<Q", len(encoded)))
    print(f"  converted {len(keep)} tensors ({dropped} conditioning tensors dropped), {end / 1e9:.2f} GB")
    return len(keep), end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="large", choices=sorted(REPOS))
    parser.add_argument("--repo", help="override the repository (default MuScriptor/muscriptor-<size>)")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--out", default="weights/")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        sys.exit(
            f"{args.token_env} is not set. The weights are CC BY-NC 4.0 and gated:\n"
            f"  accept the license at https://huggingface.co/{args.repo or f'MuScriptor/muscriptor-{args.size}'}\n"
            f"  then export {args.token_env}=hf_..."
        )
    repo = args.repo or f"MuScriptor/muscriptor-{args.size}"
    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "upstream.safetensors")
    final = os.path.join(args.out, f"muscriptor-{args.size}.safetensors")

    sha = json.loads(hf_get(f"https://huggingface.co/api/models/{repo}", token))["sha"]
    url = f"https://huggingface.co/{repo}/resolve/{sha}/model.safetensors"
    print(f"downloading {repo} @ {sha[:12]} -> {raw}")
    download(url, token, raw)

    header, _ = read_header(raw)
    params = 0
    for info in header.values():
        count = 1
        for dim in info["shape"]:
            count *= dim
        params += count
    approx, layers, dim = REPOS[args.size]
    print(f"  {len(header)} tensors, {params / 1e6:.1f}M parameters, dim {dim}, {layers} layers")
    if abs(params - approx) / approx > 0.15:
        sys.exit(f"parameter count {params} does not look like the {args.size} variant ({approx} expected)")

    print(f"converting -> {final}")
    convert(raw, final, repo)
    os.remove(raw)
    print(f"done: {final}")


if __name__ == "__main__":
    main()
