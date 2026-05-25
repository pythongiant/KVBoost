"""
Tool-call parsers for OpenAI-compatible function calling.

Models emit tool calls in their own native syntax inside the assistant
output. This module converts that text into a list of ToolCall objects and
returns the cleaned text content (with the tool-call markup stripped).

Supported parsers
-----------------
hermes : <tool_call>{"name": "...", "arguments": {...}}</tool_call>
         Used by Qwen2.5/Qwen3, Hermes 2/3, NousResearch fine-tunes.
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Tuple

from .schema import FunctionCall, ToolCall

log = logging.getLogger(__name__)


# Accept both the Hermes-canonical `<tool_call>` AND the variant `<tools>`
# that some Qwen2.5 builds (notably -Coder-14B-AWQ) emit. Qwen's chat template
# uses `<tools>...</tools>` for tool DEFINITIONS in the system prompt and
# `<tool_call>...</tool_call>` for tool INVOCATIONS in the response — the
# model sometimes conflates the two and wraps its call in `<tools>`. The
# named-group + backreference ensures the closing tag matches the opening.
_HERMES_RE = re.compile(
    r"<(?P<tag>tool_call|tools)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.DOTALL,
)


def parse_hermes(text: str) -> Tuple[str, List[ToolCall]]:
    """
    Extract Hermes-style <tool_call>{...}</tool_call> blocks (and the Qwen
    quirk variant <tools>{...}</tools>).

    Returns (cleaned_text, tool_calls). If no calls are found, the input text
    is returned unchanged with an empty list.
    """
    calls: List[ToolCall] = []
    cleaned_parts: List[str] = []
    cursor = 0

    for match in _HERMES_RE.finditer(text):
        cleaned_parts.append(text[cursor:match.start()])
        cursor = match.end()

        body = match.group("body").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            log.warning("hermes parser: malformed JSON in tool_call: %s", e)
            cleaned_parts.append(match.group(0))
            continue

        name = obj.get("name")
        if not isinstance(name, str):
            log.warning("hermes parser: tool_call missing 'name': %r", body)
            cleaned_parts.append(match.group(0))
            continue

        args = obj.get("arguments", {})
        # OpenAI spec: arguments must be a JSON-encoded string
        args_str = json.dumps(args) if not isinstance(args, str) else args

        calls.append(ToolCall(function=FunctionCall(name=name, arguments=args_str)))

    cleaned_parts.append(text[cursor:])
    cleaned = "".join(cleaned_parts).strip()
    return cleaned, calls


PARSERS = {
    "hermes": parse_hermes,
}


def parse(text: str, parser_name: str) -> Tuple[str, List[ToolCall]]:
    fn = PARSERS.get(parser_name)
    if fn is None:
        raise ValueError(
            f"Unknown tool-call parser: {parser_name!r}. "
            f"Available: {sorted(PARSERS)}"
        )
    return fn(text)


# ── Streaming parser ──────────────────────────────────────────────────────────


class StreamingHermesParser:
    """
    Incremental Hermes tool-call parser for SSE streaming.

    Usage::

        parser = StreamingHermesParser()
        for delta_text in stream:
            for event in parser.feed(delta_text):
                # event is ("text", str) or ("tool_call", ToolCall)
                ...
        for event in parser.flush():
            ...

    The parser holds back any text that *could* be the start of a `<tool_call>`
    tag, so the caller never sees partial markup. Once a complete
    `<tool_call>...</tool_call>` block is buffered, the JSON body is parsed
    and a single ("tool_call", ToolCall) event is emitted with the full
    name + arguments. Malformed blocks fall through as plain text.
    """

    # Accept either tag. The `_active_close` tracks which closer to look for
    # once we've seen an opener (so `<tool_call>` doesn't get matched by
    # `</tools>` or vice versa).
    OPEN_TAGS = ("<tool_call>", "<tools>")
    CLOSE_MAP = {"<tool_call>": "</tool_call>", "<tools>": "</tools>"}
    MAX_OPEN_LEN = max(len(t) for t in OPEN_TAGS)

    def __init__(self):
        self._buf = ""              # text awaiting emission (non-tool-call mode)
        self._json = ""             # JSON body accumulator (in tool-call mode)
        self._in_call = False
        self._active_open: str = ""   # the opener we matched on this call
        self._active_close: str = ""  # its matching closer

    def feed(self, delta: str):
        events = []
        if self._in_call:
            self._json += delta
            i = self._json.find(self._active_close)
            if i == -1:
                return events
            body = self._json[:i]
            remainder = self._json[i + len(self._active_close):]
            self._json = ""
            self._in_call = False
            events.extend(self._parse_call(body))
            events.extend(self._consume_text(remainder))
            return events
        return self._consume_text(delta)

    def flush(self):
        events = []
        if self._buf:
            events.append(("text", self._buf))
            self._buf = ""
        if self._in_call:
            log.warning("hermes streaming parser: stream ended mid tool_call")
            events.append(("text", self._active_open + self._json))
            self._json = ""
            self._in_call = False
            self._active_open = self._active_close = ""
        return events

    def _find_earliest_open(self, s: str):
        """Return (index, opener_string) of the earliest opening tag in s,
        or (-1, '') if none. When multiple tags match at different positions,
        the earliest wins; ties broken by appearance order in OPEN_TAGS."""
        best_i = -1
        best_tag = ""
        for tag in self.OPEN_TAGS:
            i = s.find(tag)
            if i == -1:
                continue
            if best_i == -1 or i < best_i:
                best_i = i
                best_tag = tag
        return best_i, best_tag

    def _consume_text(self, text: str):
        events = []
        self._buf += text
        while True:
            i, opener = self._find_earliest_open(self._buf)
            if i == -1:
                # Maybe the buffer ends with a partial tag — hold those chars.
                hold = self._suffix_prefix_of_any_open(self._buf)
                if hold and len(self._buf) > hold:
                    events.append(("text", self._buf[:-hold]))
                    self._buf = self._buf[-hold:]
                elif hold:
                    pass  # entire buffer could be a partial tag — hold it all
                else:
                    if self._buf:
                        events.append(("text", self._buf))
                    self._buf = ""
                return events
            head = self._buf[:i]
            tail = self._buf[i + len(opener):]
            self._buf = ""
            if head:
                events.append(("text", head))
            self._in_call = True
            self._active_open = opener
            self._active_close = self.CLOSE_MAP[opener]
            j = tail.find(self._active_close)
            if j == -1:
                self._json = tail
                return events
            body = tail[:j]
            remainder = tail[j + len(self._active_close):]
            self._in_call = False
            self._active_open = self._active_close = ""
            events.extend(self._parse_call(body))
            self._buf = remainder
            # loop to look for more tool calls in remainder

    def _suffix_prefix_of_any_open(self, s: str) -> int:
        """How many trailing chars of s might be the prefix of ANY known opener?
        Used to decide how much buffer to hold back to avoid emitting partial
        markup as text."""
        max_hold = min(len(s), self.MAX_OPEN_LEN - 1)
        for i in range(max_hold, 0, -1):
            tail = s[-i:]
            if any(tag.startswith(tail) for tag in self.OPEN_TAGS):
                return i
        return 0

    def _parse_call(self, body: str):
        body = body.strip()
        wrapper = self._active_open or "<tool_call>"
        wrapper_close = self.CLOSE_MAP.get(wrapper, "</tool_call>")
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            log.warning("hermes streaming: malformed JSON: %s body=%r", e, body)
            return [("text", f"{wrapper}{body}{wrapper_close}")]
        name = obj.get("name")
        if not isinstance(name, str):
            return [("text", f"{wrapper}{body}{wrapper_close}")]
        args = obj.get("arguments", {})
        args_str = json.dumps(args) if not isinstance(args, str) else args
        return [("tool_call", ToolCall(function=FunctionCall(name=name, arguments=args_str)))]


STREAMING_PARSERS = {
    "hermes": StreamingHermesParser,
}


def make_streaming_parser(parser_name: str):
    cls = STREAMING_PARSERS.get(parser_name)
    if cls is None:
        raise ValueError(
            f"Unknown streaming tool-call parser: {parser_name!r}. "
            f"Available: {sorted(STREAMING_PARSERS)}"
        )
    return cls()
