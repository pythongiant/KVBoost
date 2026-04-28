"""
OpenAI-compatible request / response schemas.

Implements the subset of the OpenAI API used by most clients:
  POST /v1/completions       — text completion
  POST /v1/chat/completions  — chat completion (messages list)
  GET  /v1/models            — model listing

Streaming (stream=True) returns Server-Sent Events in the same format as
the OpenAI API so that any SSE-aware client works without changes.

References:
  https://platform.openai.com/docs/api-reference/completions
  https://platform.openai.com/docs/api-reference/chat
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ── Shared ────────────────────────────────────────────────────────────────────

def _request_id() -> str:
    return f"kvboost-{uuid.uuid4().hex[:12]}"


# ── /v1/completions ───────────────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=16)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs: Optional[int] = None
    echo: bool = False
    best_of: int = 1
    user: Optional[str] = None

    @field_validator("prompt")
    @classmethod
    def normalise_prompt(cls, v):
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("prompt list must not be empty")
        return v

    @property
    def prompts(self) -> List[str]:
        if isinstance(self.prompt, str):
            return [self.prompt]
        return self.prompt

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: Literal["stop", "length"] = "stop"
    logprobs: None = None


class UsageStats(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=_request_id)
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageStats


# Streaming chunk
class CompletionChunk(BaseModel):
    id: str = Field(default_factory=_request_id)
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]


# ── /v1/chat/completions ──────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=8192)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=16)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None

    @field_validator("messages")
    @classmethod
    def at_least_one_message(cls, v):
        if not v:
            raise ValueError("messages must not be empty")
        return v

    def to_prompt(self, tokenizer=None) -> str:
        """
        Render the messages list to a single prompt string.

        If the tokenizer exposes apply_chat_template (HF ≥ 4.34) we use that
        so the model sees the correct special tokens.  Otherwise we fall back
        to a simple role: content format that works well for instruction-tuned
        models.
        """
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                msg_dicts = [{"role": m.role, "content": m.content} for m in self.messages]
                return tokenizer.apply_chat_template(
                    msg_dicts,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        # Fallback
        parts = []
        for m in self.messages:
            parts.append(f"{m.role.capitalize()}: {m.content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=_request_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: UsageStats


# Streaming chunk
class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=_request_id)
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[str, Any]]


# ── /v1/models ────────────────────────────────────────────────────────────────

class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "kvboost"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard]


# ── Internal (not exposed to clients) ────────────────────────────────────────

class PendingRequest(BaseModel):
    """Internal wrapper carried through the batch queue."""
    model_config = {"arbitrary_types_allowed": True}

    request_id: str = Field(default_factory=_request_id)
    prompt: str
    max_tokens: int
    temperature: float
    do_sample: bool
    stream: bool
    model_name: str
    # asyncio.Future set by the worker when generation completes
    future: Any = None
