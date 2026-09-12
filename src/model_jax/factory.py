# -*- coding: utf-8 -*-
"""Model selection for the JAX path.

Mirrors ``model.factory``'s dispatch shape, but as a **separate registry**, not
an entry added to that module's ``_KINDS``.

That is a deliberate departure from the original plan, which said both that
``model/factory.py``'s ``_KINDS`` would gain a ``mamba2_jax`` sibling *and*
that the registry would be kept separate so the PyTorch path never imports
jax/flax. Those cannot both hold, and the first one breaks things:
``model.factory.module_class`` is annotated ``-> Type[L.LightningModule]`` and
``predict.py`` calls ``cls.from_pretrained(...)`` on whatever it returns, so a
JAX kind reachable through that dispatch would either import jax into every
PyTorch entrypoint or blow up at ``predict`` time. Keeping two registries costs
~20 lines and keeps both paths honest.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Callable, Dict

log = logging.getLogger(__name__)

_KINDS = {
    "mamba2_jax": ("model_jax.mimir_mamba2", "MimirMamba2Model"),
}

DEFAULT_KIND = "mamba2_jax"


def model_class(kind: str):
    """Resolve a ``cfg.model.kind`` string to its nnx.Module class."""
    try:
        module_path, class_name = _KINDS[kind]
    except KeyError:
        raise ValueError(
            f"Unknown JAX model kind {kind!r} — expected one of {sorted(_KINDS)}. "
            "PyTorch kinds live in model.factory and are not interchangeable."
        ) from None
    return getattr(importlib.import_module(module_path), class_name)


def filter_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop entries *func* does not name as a parameter.

    Same contract as ``model.factory.filter_kwargs``: a ``**kwargs`` catch-all
    is deliberately NOT treated as "accepts anything", so config groups can
    carry documentation-only keys without every entrypoint knowing about them.
    """
    accepted = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in accepted}


def build_model(kind: str, kwargs: Dict[str, Any], **required):
    """Instantiate the model named by *kind* with whatever of *kwargs* it takes.

    *required* carries the arguments that are not config-derived (``tables``,
    ``rngs``) and are always passed through.
    """
    cls = model_class(kind)
    accepted = filter_kwargs(cls.__init__, kwargs)
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        log.debug("%s ignores config keys: %s", cls.__name__, ", ".join(dropped))
    return cls(**accepted, **required)
