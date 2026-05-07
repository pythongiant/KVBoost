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


_HERMES_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
    re.DOTALL,
)


def parse_hermes(text: str) -> Tuple[str, List[ToolCall]]:
    """
    Extract Hermes-style <tool_call>{...}</tool_call> blocks.

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

    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"

    def __init__(self):
        self._buf = ""        # text awaiting emission (non-tool-call mode)
        self._json = ""       # JSON body accumulator (in tool-call mode)
        self._in_call = False

    def feed(self, delta: str):
        events = []
        if self._in_call:
            self._json += delta
            i = self._json.find(self.CLOSE)
            if i == -1:
                return events
            body = self._json[:i]
            remainder = self._json[i + len(self.CLOSE):]
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
            # Fall back: emit raw partial as text so it isn't lost silently.
            events.append(("text", self.OPEN + self._json))
            self._json = ""
            self._in_call = False
        return events

    def _consume_text(self, text: str):
        events = []
        self._buf += text
        while True:
            i = self._buf.find(self.OPEN)
            if i == -1:
                hold = self._suffix_prefix_of_open(self._buf)
                if hold and len(self._buf) > hold:
                    events.append(("text", self._buf[:-hold]))
                    self._buf = self._buf[-hold:]
                elif hold:
                    pass  # entire buffer is a potential partial tag — hold
                else:
                    if self._buf:
                        events.append(("text", self._buf))
                    self._buf = ""
                return events
            head = self._buf[:i]
            tail = self._buf[i + len(self.OPEN):]
            self._buf = ""
            if head:
                events.append(("text", head))
            self._in_call = True
            j = tail.find(self.CLOSE)
            if j == -1:
                self._json = tail
                return events
            body = tail[:j]
            remainder = tail[j + len(self.CLOSE):]
            self._in_call = False
            events.extend(self._parse_call(body))
            self._buf = remainder
            # loop to look for more tool calls in remainder

    def _suffix_prefix_of_open(self, s: str) -> int:
        max_hold = min(len(s), len(self.OPEN) - 1)
        for i in range(max_hold, 0, -1):
            if self.OPEN.startswith(s[-i:]):
                return i
        return 0

    def _parse_call(self, body: str):
        body = body.strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            log.warning("hermes streaming: malformed JSON: %s body=%r", e, body)
            return [("text", f"<tool_call>{body}</tool_call>")]
        name = obj.get("name")
        if not isinstance(name, str):
            return [("text", f"<tool_call>{body}</tool_call>")]
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
