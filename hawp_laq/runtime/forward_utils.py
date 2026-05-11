from __future__ import annotations

import inspect


def _model_forward_has_arg(model, name: str) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def prefill_forward_last_logits(model, **kwargs):
    """Run a prefill forward while asking compatible models for last logits only.

    Recent Transformers causal LM implementations can avoid materializing the
    full ``[batch, seq_len, vocab]`` logits tensor during prefill.  This helper
    keeps older versions compatible by falling back to the original call when
    the model forward signature does not expose such an argument.
    """
    forward_kwargs = dict(kwargs)
    if _model_forward_has_arg(model, "logits_to_keep"):
        forward_kwargs["logits_to_keep"] = 1
    elif _model_forward_has_arg(model, "num_logits_to_keep"):
        forward_kwargs["num_logits_to_keep"] = 1
    return model(**forward_kwargs)
