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

import logging
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)


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

class ContentPart(BaseModel):
    type: str
    text: Optional[str] = None


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string per OpenAI spec


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Union[str, List[ContentPart]]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    @field_validator("content")
    @classmethod
    def flatten_content(cls, v):
        if v is None:
            return v
        # Accept OpenAI's multimodal parts format and flatten text parts.
        # Non-text parts (images, etc.) are dropped — this server is text-only.
        if isinstance(v, list):
            return "".join(p.text for p in v if p.type == "text" and p.text)
        return v


class FunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    n: int = Field(default=1, ge=1, le=16)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: Optional[str] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Union[Literal["auto", "none", "required"], Dict[str, Any]]] = None

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
                msg_dicts = []
                for m in self.messages:
                    d: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
                    if m.name is not None:
                        d["name"] = m.name
                    if m.tool_calls:
                        d["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in m.tool_calls
                        ]
                    if m.tool_call_id is not None:
                        d["tool_call_id"] = m.tool_call_id
                    msg_dicts.append(d)
                tools_arg = None
                if self.tools:
                    tools_arg = [t.model_dump() for t in self.tools]
                return tokenizer.apply_chat_template(
                    msg_dicts,
                    tools=tools_arg,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                if tools_arg:
                    log.warning(
                        "apply_chat_template failed with tools=%d defs (%s); "
                        "tools will NOT reach the model. Make sure the tokenizer "
                        "has a chat template that accepts a `tools` argument "
                        "(Qwen2.5/3, Hermes 2/3 do; older Llama2 does not).",
                        len(tools_arg), exc,
                    )
                else:
                    log.debug("apply_chat_template failed: %s — using fallback", exc)
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
    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"


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
