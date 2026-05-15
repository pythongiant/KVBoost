"""Convenience entry point for loading a :class:`StreamingCausalLM`."""

from __future__ import annotations

from typing import Any, Optional

from .config import StreamingConfig
from .model_shell import StreamingCausalLM


def load_streaming_model(
    model_id: str,
    *,
    awq_path: Optional[str] = None,
    streaming_config: Optional[StreamingConfig] = None,
    device: str = "auto",
    **kwargs: Any,
) -> StreamingCausalLM:
    """Load a streaming-capable causal LM. Thin wrapper over
    :meth:`StreamingCausalLM.from_pretrained`.
    """
    if streaming_config is None:
        streaming_config = StreamingConfig()
    return StreamingCausalLM.from_pretrained(
        model_id,
        awq_path=awq_path,
        streaming_config=streaming_config,
        device=device,
        **kwargs,
    )


__all__ = ["load_streaming_model"]
