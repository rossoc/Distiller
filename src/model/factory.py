# -*- coding: utf-8 -*-
"""Model-module selection, so the training/prediction entrypoints stay generic.

``cfg.model.kind`` names which LightningModule to build; every other key under
``cfg.model`` is forwarded to that module's constructor if — and only if — it
names one of its arguments. That keeps ``train.py``/``predict.py`` free of
per-architecture branching, and lets each model's config group carry exactly
the knobs that model actually has.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, Type

import lightning as L

log = logging.getLogger(__name__)

# Imported lazily inside module_class(): both modules pull in transformers and
# (for dfm_mimir) trust_remote_code machinery, and there is no reason to pay
# for the one you are not using.
_KINDS = {
    "dfm_mimir": ("model.dfm_mimir", "DFMMimirModule"),
    "mamba2": ("model.mimir_mamba2", "MimirMamba2Module"),
}

DEFAULT_KIND = "dfm_mimir"


def module_class(kind: str) -> Type[L.LightningModule]:
    """Resolve a ``cfg.model.kind`` string to its LightningModule class."""
    try:
        module_path, class_name = _KINDS[kind]
    except KeyError:
        raise ValueError(
            f"Unknown model kind {kind!r} — expected one of {sorted(_KINDS)}."
        ) from None
    imported = __import__(module_path, fromlist=[class_name])
    return getattr(imported, class_name)


def filter_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop entries *func* does not name as a parameter.

    Config groups carry documentation-only or downstream-only keys (``kind``,
    ``use_peft``, ...) alongside real arguments, and the two model kinds name
    their donor differently (``model_id`` vs ``donor_model_id``); this is what
    lets them share one call site. A ``**kwargs`` catch-all in *func* is
    deliberately NOT treated as "accepts anything" — the point is to pass only
    what the callee explicitly declares.
    """
    accepted = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in accepted}


def supported_kwargs(cls: Type[L.LightningModule], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop entries ``cls.__init__`` does not accept."""
    return filter_kwargs(cls.__init__, kwargs)


def build_module(kind: str, kwargs: Dict[str, Any]) -> L.LightningModule:
    """Instantiate the module named by *kind* with whatever of *kwargs* it takes."""
    cls = module_class(kind)
    accepted = supported_kwargs(cls, kwargs)
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        log.debug("%s ignores config keys: %s", cls.__name__, ", ".join(dropped))
    return cls(**accepted)
