#!/usr/bin/env python3
"""Generate an opt-in small-checkpoint parity reference using upstream PyTorch.

Run from any directory after installing numpy, torch, einops and safetensors,
with an upstream MuScriptor checkout (and its dependencies) available:

  python tools/gen_real_small_reference.py --muscriptor-root /path/to/muscriptor \
      --weights /path/to/model.safetensors --output /tmp/mtab-small-reference.json

The weights and generated reference are user-provided local files: they must
not be checked in. The input is the repository's synthetic plucked-string
fixture. The reference uses upstream float32; Rust compares within 1e-3 scaled
error and requires exact tokens for greedy and forced-prompt generation.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muscriptor-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    sys.path.insert(0, str(args.muscriptor_root.resolve()))

    import numpy as np
    import torch
    from safetensors.torch import load_file
    from muscriptor.modules.conditioners import (
        ClassConditioner, ConditioningAttributes, ConditioningProvider,
        MelSpectrogramConditioner, WavCondition,
    )
    from muscriptor.modules.streaming import increment_steps, init_states
    try:
        from muscriptor.models.lm import LMModel
    except ModuleNotFoundError as exc:
        # Also supports the lightweight local reference layout used for tests.
        if exc.name not in ("muscriptor.models", "muscriptor.models.lm"):
            raise
        from muscriptor.lm import LMModel

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    dim, card = 768, 1393
    provider = ConditioningProvider(conditioners={
        "self_wav": MelSpectrogramConditioner(output_dim=dim, device="cpu", sample_rate=16000,
            n_fft=2048, frame_rate=100, n_mel_bins=512, log_scale=True, eps=1e-6,
            normalize_audio=False),
        "instrument_group": ClassConditioner(1000, dim),
        "dataset_name": ClassConditioner(4, dim),
    })
    model = LMModel(condition_provider=provider, card=card, dim=dim, num_heads=12,
        hidden_scale=4, cfg_coef=1.0, num_layers=14, max_period=10000)
    tensors = load_file(str(args.weights), device="cpu")
    if any(k.startswith(("emb.1.", "linears.1.")) for k in tensors):
        raise ValueError("only single-codebook checkpoints are supported")
    def remap(key):
        if key.startswith("emb.0."):
            return "emb." + key[len("emb.0."):]
        if key.startswith("linears.0."):
            return "linear." + key[len("linears.0."):]
        return key
    model.load_state_dict({remap(k): v for k, v in tensors.items()}, strict=True)
    del tensors
    model.eval()

    fixture = Path(__file__).resolve().parent.parent / "crates/mtab-core/tests/fixtures/model/guitar_plucks.f32"
    wav = np.fromfile(fixture, dtype="<f4").copy()
    if len(wav) != 80000:
        raise ValueError("expected 80000 mono 16 kHz input samples")
    attrs = ConditioningAttributes(
        wav={"self_wav": WavCondition(wav=torch.from_numpy(wav).view(1, 1, -1),
            length=torch.tensor([80000]), sample_rate=[16000], path=[None], seek_time=[0.0])},
        text={"instrument_group": "4 5 6", "dataset_name": None},
    )
    conds = provider(provider.tokenize([attrs]))
    prep = sum(c.shape[1] for c, _ in conds.values())
    maxlen = 80
    # Guitar representatives 24/26/29; exclude all other program and drum tokens.
    forbidden = [1135 + i for i in range(130) if i not in (24, 26, 29)] + list(range(1265, 1393))
    scenarios = {}
    for name, prompt in [("greedy", []), ("forced_prompt", [1134, 1159, 1133])]:
        state = init_states(model, batch_size=1, sequence_length=prep + maxlen)
        seq = torch.tensor([[card] + prompt], dtype=torch.long)
        rows, tokens = [], prompt.copy()
        for step in range(maxlen - len(prompt)):
            logits = model._compute_logits(seq, conds, state, step == 0, 1.0, forbidden_tokens=forbidden)
            rows.append([v if np.isfinite(v) else None for v in logits[0].tolist()])
            token = int(logits.argmax(-1)[0])
            tokens.append(token)
            increment_steps(model.transformer, state,
                increment=seq.shape[-1] + (prep if step == 0 else 0))
            seq = torch.tensor([[token]], dtype=torch.long)
            if token == 1:
                break
        scenarios[name] = {"prompt": prompt, "tokens": tokens, "logits": rows}
        print(f"{name}: {len(tokens)} tokens, {len(rows)} logit rows")
    result = {"source": "MuScriptor small; upstream PyTorch float32", "torch": torch.__version__,
        "checkpoint_sha256": sha256(args.weights), "fixture_sha256": sha256(fixture),
        "max_gen_len": maxlen, "scenarios": scenarios}
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
    print(f"Reference saved to {args.output}; do not commit this model-derived file.")


if __name__ == "__main__":
    main()
