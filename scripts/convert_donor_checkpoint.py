# -*- coding: utf-8 -*-
"""One-time: DFM-Mimir (PyTorch, trust_remote_code) -> plain arrays for JAX.

    python scripts/convert_donor_checkpoint.py \
        --donor-model-id danish-foundation-models/DFM-Mimir \
        --out outputs/donor_jax

Writes into ``--out``:

* ``donor_tables.npz`` — the two frozen ``[vocab_size, d_donor]`` tables
  (``embed``, ``head``) plus the donor id, at the requested dtype.
* the donor's tokenizer, saved locally.

Why this exists as a separate, run-once script rather than a function the
training path calls: the donor is a custom ``trust_remote_code`` Hub model with
no Flax-native counterpart, so reading its tables *requires* torch and
transformers. Doing that once, offline, keeps them off the JAX training
critical path entirely — ``model_jax.donor_projection.load_donor_tables`` reads
only NumPy, and a training run never imports torch.

The tokenizer comes along for the ride because token ids have to index the
donor tables: the two models are not merely compatible but required to share a
tokenizer. Saving it locally means the JAX path can load it with a plain
``AutoTokenizer.from_pretrained(<dir>)`` — no ``trust_remote_code``, no Hub
round-trip.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

log = logging.getLogger("convert_donor")

DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-model-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    import torch
    from transformers import AutoTokenizer

    from model.hf_compat import load_model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trust = not args.no_trust_remote_code

    log.info("Loading donor %s (once)", args.donor_model_id)
    donor = load_model(args.donor_model_id, trust_remote_code=trust, dtype=torch.float32)

    with torch.no_grad():
        embed = donor.get_input_embeddings().weight.detach().to("cpu", torch.float32)
        out_embed = donor.get_output_embeddings()
        # A tied-embedding donor has no separate head — the embedding table
        # *is* the head. DFM-Mimir is untied, so this is just defensiveness.
        head = (
            embed.clone()
            if out_embed is None
            else out_embed.weight.detach().to("cpu", torch.float32)
        )

    # Saved via ml_dtypes so bfloat16 survives the .npz round-trip; NumPy has
    # no native bfloat16 and would otherwise silently upcast (doubling the file).
    if args.dtype == "bf16":
        import ml_dtypes

        cast = lambda t: t.numpy().astype(ml_dtypes.bfloat16)  # noqa: E731
    else:
        cast = lambda t: t.numpy().astype(DTYPES[args.dtype])  # noqa: E731

    embed_np, head_np = cast(embed), cast(head)
    del donor, out_embed, embed, head

    npz = out / "donor_tables.npz"
    np.savez(npz, embed=embed_np, head=head_np, model_id=np.asarray(args.donor_model_id))
    log.info(
        "Wrote %s — vocab=%d d_donor=%d dtype=%s (%.1f MiB per table)",
        npz,
        embed_np.shape[0],
        embed_np.shape[1],
        args.dtype,
        embed_np.nbytes / 2**20,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.donor_model_id, trust_remote_code=trust)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(out))
    log.info("Wrote tokenizer to %s", out)


if __name__ == "__main__":
    main()
