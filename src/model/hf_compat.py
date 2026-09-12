from __future__ import annotations

import logging
import os

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Dtype names as they appear in config (``model.dtype``), shared by
# ``dfm_mimir`` and ``mimir_mamba2`` so the two model kinds can't drift on
# what "bf16" means.
DTYPE_MAP = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _is_cached_locally(model_id: str) -> bool:
    """True when the model's config.json is present in the HF cache."""
    if os.path.isdir(model_id):
        return True
    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


def load_tokenizer(
    model_id: str, trust_remote_code: bool = True
) -> PreTrainedTokenizerBase:
    """Load a tokenizer and make sure it has a pad token."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=_is_cached_locally(model_id),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str, trust_remote_code: bool = True, **kwargs
) -> PreTrainedModel:
    """Load a causal LM, skipping the Hub cache-freshness HEAD when warm. """
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=_is_cached_locally(model_id),
        **kwargs,
    )


# Cold-start ``from_pretrained`` still emits httpx/huggingface_hub INFO lines
# for the initial fetch. Keep both loggers quiet so those don't clutter the
# per-fold output; warm-cache loads via ``load_model`` / ``load_tokenizer``
# skip the HEAD requests outright.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Workaround for a transformers>=5.3 regression.
#
# In transformers 5.3.x, ``TokenizersBackend.__init__`` (tokenization_utils_
# tokenizers.py, ~line 442) calls ``self._patch_mistral_regex(...)`` passing
# ``fix_mistral_regex=kwargs.get("fix_mistral_regex")`` EXPLICITLY while also
# forwarding ``**kwargs`` — which itself still contains ``fix_mistral_regex``.
# Python rejects the call with::
#
#     TypeError: _patch_mistral_regex() got multiple values for keyword
#                argument 'fix_mistral_regex'
#
# The crash is raised at the *call site* (argument-binding time), so patching
# ``_patch_mistral_regex`` itself cannot help. Instead we wrap
# ``TokenizersBackend.__init__`` and strip the duplicate ``fix_mistral_regex``
# from its ``**kwargs`` before the buggy call runs. This disables only the
# (mistral-specific) regex patch, which DFM-Mimir does not need, and is a no-op
# when the kwarg is absent — so it stays safe across transformers versions.
# ---------------------------------------------------------------------------
from transformers.tokenization_utils_tokenizers import (  # noqa: E402
    TokenizersBackend,
)

# Guard so a re-import (or an explicit second apply) doesn't wrap the wrapper.
if not getattr(TokenizersBackend.__init__, "_distiller_patched", False):
    _original_tokenizers_init = TokenizersBackend.__init__

    def _patched_tokenizers_init(self, *args, **kwargs):
        # Drop the kwarg that the internal _patch_mistral_regex call
        # double-passes.
        kwargs.pop("fix_mistral_regex", None)
        return _original_tokenizers_init(self, *args, **kwargs)

    _patched_tokenizers_init._distiller_patched = True
    TokenizersBackend.__init__ = _patched_tokenizers_init
