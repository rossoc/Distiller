# -*- coding: utf-8 -*-
"""Shared HuggingFace compatibility shims.

Imported for its side effects by every module that loads a DFM-Mimir
tokenizer (``model.dfm_mimir``, ``model.donor_projection``). Importing it
twice is a no-op — the patch is applied exactly once, at first import.
"""

from __future__ import annotations

import logging

# Every AutoTokenizer/AutoModelForCausalLM.from_pretrained() call re-checks
# the Hub for cache freshness (HEAD requests) even when the model is fully
# cached locally — each K-fold run reloads the model once per fold, so this
# would otherwise print a burst of "HTTP Request: HEAD ..." lines per fold.
# Harmless noise when the cache is warm; bump both loggers to WARNING/ERROR.
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
