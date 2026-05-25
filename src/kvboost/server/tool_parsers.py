"""
Tool-call parsers for OpenAI-compatible function calling.

Models emit tool calls in their own native syntax inside the assistant
output. This module converts that text into a list of ToolCall objects and
returns the cleaned text content (with the tool-call markup stripped).

Supported parsers
-----------------
hermes         : <tool_call>{"name": "...", "arguments": {...}}</tool_call>
                 Qwen2.5/Qwen3, Hermes 2/3, NousResearch fine-tunes.
json_codeblock : ```json
                 {"name": "...", "arguments": {...}}
                 ```
                 Qwen2.5-Coder and many instruction-tuned models that weren't
                 trained with Hermes syntax. Single object or array of calls.
                 Language tag optional (matches ``` and ```json / ```tool_call).
qwen3_coder    : <tool_call>
                 <function=name>
                 <parameter=key>value</parameter>
                 ...
                 </function>
                 </tool_call>
                 Qwen3-Coder agent fine-tunes — XML-attribute-style parameters
                 instead of JSON.
llama          : <|python_tag|>{"name": "...", "parameters": {...}}<|eom_id|>
                 Llama 3.1 / 3.2 / 3.3 with the canonical chat template.
mistral        : [TOOL_CALLS][{"name": "...", "arguments": {...}}, ...]
                 Mistral / Mixtral chat-template tool format.
auto           : tries every registered parser in order; the first one that
                 finds at least one tool call wins. Use this when the
                 deployed model's exact format isn't pinned down or varies.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, List, Optional, Set, Tuple

from .schema import FunctionCall, ToolCall

log = logging.getLogger(__name__)

# A parser is ``(text, tool_names) -> (cleaned_text, tool_calls)``.
# ``tool_names`` is an optional allowlist; if provided, calls whose name is
# not in the allowlist are dropped (treats them as model hallucination).
ParserFn = Callable[[str, Optional[Set[str]]], Tuple[str, List[ToolCall]]]


# Heuristic: a JSON object is a tool call only if it has BOTH a string `name`
# AND an `arguments` (or `parameters`) field. Otherwise a benign code example
# like ``{"name": "John", "age": 30}`` would be misread as a tool invocation.
def _looks_like_tool_call(obj: dict, args_keys: Tuple[str, ...] = ("arguments", "parameters")) -> bool:
    if not isinstance(obj, dict):
        return False
    if not isinstance(obj.get("name"), str):
        return False
    if not any(k in obj for k in args_keys):
        return False
    for k in args_keys:
        if k in obj:
            v = obj[k]
            if not isinstance(v, (dict, str, list)):
                return False
    return True


def _extract_args(obj: dict, args_keys: Tuple[str, ...] = ("arguments", "parameters")) -> str:
    """Return the canonical OpenAI ``arguments`` JSON string for a call obj."""
    raw = None
    for k in args_keys:
        if k in obj:
            raw = obj[k]
            break
    if raw is None:
        raw = {}
    if isinstance(raw, str):
        return raw
    return json.dumps(raw)


def _tool_name_allowed(name: str, allowed: Optional[Set[str]]) -> bool:
    """When the request supplies a tools list, drop calls that name a tool the
    caller never declared. When allowed is None (or empty), accept any name."""
    if not allowed:
        return True
    return name in allowed


# ── hermes ───────────────────────────────────────────────────────────────────
#
# Accept both the canonical ``<tool_call>`` AND the variant ``<tools>`` that
# some Qwen2.5 builds (notably -Coder-14B-AWQ) emit. The named-group +
# backreference ensures the closing tag matches the opener.
_HERMES_RE = re.compile(
    r"<(?P<tag>tool_call|tools)>\s*(?P<body>.*?)\s*</(?P=tag)>",
    re.DOTALL,
)


def parse_hermes(
    text: str,
    tool_names: Optional[Set[str]] = None,
) -> Tuple[str, List[ToolCall]]:
    """Extract ``<tool_call>{...}</tool_call>`` blocks (and the ``<tools>``
    Qwen variant).

    Because the XML wrapper is itself a strong intent signal, this parser is
    lenient — a missing ``arguments`` field is treated as ``{}``. The
    ambiguity that ``json_codeblock`` worries about doesn't apply here.
    """
    calls: List[ToolCall] = []
    cleaned_parts: List[str] = []
    cursor = 0

    for match in _HERMES_RE.finditer(text):
        body = match.group("body").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            # Debug level: under `auto` parsing, hermes is tried against
            # every <tool_call> block including qwen3_coder's XML payload.
            # Falling through is expected; reserve warnings for cases where
            # hermes was explicitly chosen.
            log.debug("hermes parser: not JSON in tool_call: %s", e)
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        candidates = _coerce_call_objects(obj)
        valid: List[ToolCall] = []
        for cand in candidates:
            name = cand.get("name") if isinstance(cand, dict) else None
            if not isinstance(name, str):
                continue
            if not _tool_name_allowed(name, tool_names):
                log.debug("hermes: dropping unknown tool %r (not in allowlist)", name)
                continue
            valid.append(ToolCall(
                function=FunctionCall(name=name, arguments=_extract_args(cand)),
            ))

        if not valid:
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        cleaned_parts.append(text[cursor:match.start()])
        cursor = match.end()
        calls.extend(valid)

    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), calls


# ── json_codeblock ───────────────────────────────────────────────────────────
#
# Matches a fenced code block whose body is either a single JSON object with
# {"name": "...", "arguments": {...}} or a JSON array of such objects. The
# language tag is optional and we accept "json", "tool_call", "tool_calls",
# or none. The fence may use 3+ backticks (some models emit 4).
_JSON_CODEBLOCK_RE = re.compile(
    r"```+(?:json|tool_calls?|JSON)?[ \t]*\n?(?P<body>[\s\S]*?)\n?[ \t]*```+",
)


def _coerce_call_objects(obj) -> List[dict]:
    """Normalize parser input to a list of call dicts.

    Accepts:
      * single object: ``{"name": "...", "arguments": {...}}``
      * list of objects: ``[{...}, {...}]``
      * single object with embedded list under "tool_calls" or "calls"
    Returns ``[]`` for anything else.
    """
    if isinstance(obj, list):
        return [c for c in obj if isinstance(c, dict)]
    if isinstance(obj, dict):
        for key in ("tool_calls", "calls"):
            inner = obj.get(key)
            if isinstance(inner, list):
                return [c for c in inner if isinstance(c, dict)]
        return [obj]
    return []


def parse_json_codeblock(
    text: str,
    tool_names: Optional[Set[str]] = None,
) -> Tuple[str, List[ToolCall]]:
    """Extract tool calls from ``` ```json ... ``` ``` fenced blocks.

    The fence isn't a unique tool-call signal — models also use it for
    legitimate JSON code samples. The heuristic for treating a body as a
    tool call is therefore deliberately strict:

      * the body must parse as a JSON object (or array of objects)
      * each object must have a string ``name`` AND an ``arguments`` /
        ``parameters`` key (the latter check is what filters out incidental
        objects like ``{"name": "John", "age": 30}``)
      * when ``tool_names`` is supplied, the call's name must be in it

    If any of those fail, the fence is left in the output text unchanged.
    """
    calls: List[ToolCall] = []
    cleaned_parts: List[str] = []
    cursor = 0

    for match in _JSON_CODEBLOCK_RE.finditer(text):
        body = match.group("body").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        candidates = _coerce_call_objects(obj)
        valid: List[ToolCall] = []
        for cand in candidates:
            if not _looks_like_tool_call(cand):
                # Strict: needs name AND arguments/parameters
                continue
            name = cand["name"]
            if not _tool_name_allowed(name, tool_names):
                continue
            valid.append(ToolCall(
                function=FunctionCall(name=name, arguments=_extract_args(cand)),
            ))

        if not valid:
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        cleaned_parts.append(text[cursor:match.start()])
        cursor = match.end()
        calls.extend(valid)

    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), calls


# ── qwen3_coder ──────────────────────────────────────────────────────────────
#
# Qwen3-Coder agent fine-tunes emit calls as:
#
#   <tool_call>
#   <function=execute_bash>
#   <parameter=command>ls -la</parameter>
#   </function>
#   </tool_call>
#
# Parameter values are raw text (NOT JSON-escaped) and may span multiple lines.
# This format isn't compatible with the Hermes parser, so it gets its own.
_QWEN3_TOOLCALL_RE = re.compile(
    r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>",
    re.DOTALL,
)
_QWEN3_FUNCTION_RE = re.compile(
    r"<function\s*=\s*(?P<name>[^>]+?)>\s*(?P<body>.*?)\s*</function>",
    re.DOTALL,
)
_QWEN3_PARAM_RE = re.compile(
    r"<parameter\s*=\s*(?P<key>[^>]+?)>(?P<value>.*?)</parameter>",
    re.DOTALL,
)


def parse_qwen3_coder(
    text: str,
    tool_names: Optional[Set[str]] = None,
) -> Tuple[str, List[ToolCall]]:
    """Extract Qwen3-Coder XML-attribute-style tool calls."""
    calls: List[ToolCall] = []
    cleaned_parts: List[str] = []
    cursor = 0

    for tc_match in _QWEN3_TOOLCALL_RE.finditer(text):
        inner = tc_match.group("body")
        func_match = _QWEN3_FUNCTION_RE.search(inner)
        if not func_match:
            cleaned_parts.append(text[cursor:tc_match.end()])
            cursor = tc_match.end()
            continue

        name = func_match.group("name").strip()
        if not name or not _tool_name_allowed(name, tool_names):
            cleaned_parts.append(text[cursor:tc_match.end()])
            cursor = tc_match.end()
            continue

        args: dict = {}
        for p in _QWEN3_PARAM_RE.finditer(func_match.group("body")):
            key = p.group("key").strip()
            # Preserve internal whitespace; only strip leading/trailing newlines
            # that come from formatting, not from semantic content.
            value = p.group("value").strip("\n")
            args[key] = value

        cleaned_parts.append(text[cursor:tc_match.start()])
        cursor = tc_match.end()
        calls.append(ToolCall(
            function=FunctionCall(name=name, arguments=json.dumps(args)),
        ))

    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), calls


# ── llama (Llama 3.1 / 3.2 / 3.3) ────────────────────────────────────────────
#
# Llama's tool format uses the ``<|python_tag|>`` special token followed by
# JSON ``{"name": "...", "parameters": {...}}`` and terminated by ``<|eom_id|>``
# (or ``<|eot_id|>`` in some templates). Parameters live under ``parameters``,
# not ``arguments`` — _extract_args handles both.
_LLAMA_RE = re.compile(
    r"<\|python_tag\|>\s*(?P<body>.*?)\s*(?:<\|(?:eom|eot)_id\|>|\Z)",
    re.DOTALL,
)


def parse_llama(
    text: str,
    tool_names: Optional[Set[str]] = None,
) -> Tuple[str, List[ToolCall]]:
    """Extract Llama 3.1+ ``<|python_tag|>...<|eom_id|>`` tool calls."""
    calls: List[ToolCall] = []
    cleaned_parts: List[str] = []
    cursor = 0

    for match in _LLAMA_RE.finditer(text):
        body = match.group("body").strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        candidates = _coerce_call_objects(obj)
        valid: List[ToolCall] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            name = cand.get("name")
            if not isinstance(name, str):
                continue
            if not _tool_name_allowed(name, tool_names):
                continue
            valid.append(ToolCall(
                function=FunctionCall(name=name, arguments=_extract_args(cand)),
            ))

        if not valid:
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        cleaned_parts.append(text[cursor:match.start()])
        cursor = match.end()
        calls.extend(valid)

    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), calls


# ── mistral ──────────────────────────────────────────────────────────────────
#
# Mistral chat templates emit a ``[TOOL_CALLS]`` marker followed by a JSON
# array of call objects: ``[TOOL_CALLS][{"name": "...", "arguments": {...}}]``.
_MISTRAL_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(?P<body>\[.*?\])",
    re.DOTALL,
)


def parse_mistral(
    text: str,
    tool_names: Optional[Set[str]] = None,
) -> Tuple[str, List[ToolCall]]:
    """Extract Mistral ``[TOOL_CALLS][...]`` tool calls."""
    calls: List[ToolCall] = []
    cleaned_parts: List[str] = []
    cursor = 0

    for match in _MISTRAL_RE.finditer(text):
        body = match.group("body").strip()
        try:
            arr = json.loads(body)
        except json.JSONDecodeError:
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        if not isinstance(arr, list):
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        valid: List[ToolCall] = []
        for cand in arr:
            if not isinstance(cand, dict):
                continue
            name = cand.get("name")
            if not isinstance(name, str):
                continue
            if not _tool_name_allowed(name, tool_names):
                continue
            valid.append(ToolCall(
                function=FunctionCall(name=name, arguments=_extract_args(cand)),
            ))

        if not valid:
            cleaned_parts.append(text[cursor:match.end()])
            cursor = match.end()
            continue

        cleaned_parts.append(text[cursor:match.start()])
        cursor = match.end()
        calls.extend(valid)

    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), calls


# ── auto ─────────────────────────────────────────────────────────────────────
#
# Tries each parser in order; first one with at least one accepted call wins.
# Order matters: stricter / less-ambiguous formats go first so a model whose
# output happens to match multiple patterns is interpreted by the closest
# match.
_AUTO_ORDER: Tuple[str, ...] = (
    "hermes",         # <tool_call>{...}</tool_call> — XML-tagged, unambiguous
    "qwen3_coder",    # <tool_call><function=...><parameter=...> — also tagged
    "llama",          # <|python_tag|>...<|eom_id|> — special tokens, unambiguous
    "mistral",        # [TOOL_CALLS][...] — unique prefix
    "json_codeblock", # ```json {...} ``` — last because most likely to false-match
)


def parse_auto(
    text: str,
    tool_names: Optional[Set[str]] = None,
) -> Tuple[str, List[ToolCall]]:
    """Try each parser in turn; first match wins.

    Useful when the deployed model's exact format isn't pinned — saves you
    from having to restart the server when swapping models. Note: this only
    matters when ``tool_names`` is provided, because otherwise
    ``json_codeblock`` will eagerly match anything that has a string ``name``
    and an ``arguments`` field.
    """
    for parser_name in _AUTO_ORDER:
        fn = PARSERS[parser_name]
        cleaned, calls = fn(text, tool_names)
        if calls:
            return cleaned, calls
    return text, []


PARSERS: dict[str, ParserFn] = {
    "hermes":         parse_hermes,
    "json_codeblock": parse_json_codeblock,
    "qwen3_coder":    parse_qwen3_coder,
    "llama":          parse_llama,
    "mistral":        parse_mistral,
    "auto":           parse_auto,
}


def parse(
    text: str,
    parser_name: str,
    tool_names: Optional[Set[str]] = None,
) -> Tuple[str, List[ToolCall]]:
    """Run the named parser. ``tool_names`` is an optional allowlist of
    declared function names — calls naming anything else are dropped."""
    fn = PARSERS.get(parser_name)
    if fn is None:
        raise ValueError(
            f"Unknown tool-call parser: {parser_name!r}. "
            f"Available: {sorted(PARSERS)}"
        )
    return fn(text, tool_names)


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

    def __init__(self, tool_names: Optional[Set[str]] = None):
        self._buf = ""              # text awaiting emission (non-tool-call mode)
        self._json = ""             # JSON body accumulator (in tool-call mode)
        self._in_call = False
        self._active_open: str = ""   # the opener we matched on this call
        self._active_close: str = ""  # its matching closer
        self._tool_names = tool_names

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
        name = obj.get("name") if isinstance(obj, dict) else None
        if not isinstance(name, str):
            return [("text", f"{wrapper}{body}{wrapper_close}")]
        if not _tool_name_allowed(name, self._tool_names):
            # Looks like a tool call by shape but names a function the caller
            # never declared — surface the raw markup so the client sees it.
            return [("text", f"{wrapper}{body}{wrapper_close}")]
        return [("tool_call", ToolCall(
            function=FunctionCall(name=name, arguments=_extract_args(obj)),
        ))]


class StreamingJsonCodeblockParser:
    """Incremental ```json ... ``` tool-call parser for SSE streaming.

    Mirrors :class:`StreamingHermesParser`: ``feed(delta)`` returns events,
    one of ``("text", str)`` or ``("tool_call", ToolCall)``. ``flush()`` is
    called once at end of stream to drain any held-back text and warn on
    unclosed fences.

    The parser holds back trailing ` ` or `` `` so a partial fence never
    leaks as text. Once an opening fence is seen, the body accumulates until
    the closing fence; a single body may yield multiple tool calls (if the
    JSON inside is a list).

    Known limitation: a 4+ backtick fence (`````json...`````) emits one
    stray backtick as text. Rare in practice; the non-streaming parser
    handles it cleanly via regex.
    """

    FENCE = "```"
    LANG_TAGS = ("json", "tool_calls", "tool_call", "JSON")

    def __init__(self, tool_names: Optional[Set[str]] = None):
        self._buf = ""        # text awaiting emission (outside fence)
        self._body = ""       # body accumulator (inside fence)
        self._in_fence = False
        self._tool_names = tool_names

    def feed(self, delta: str):
        events = []
        if self._in_fence:
            self._body += delta
            i = self._body.find(self.FENCE)
            if i == -1:
                return events
            body = self._body[:i]
            remainder = self._body[i + len(self.FENCE):]
            self._body = ""
            self._in_fence = False
            events.extend(self._parse_body(body))
            events.extend(self._consume_text(remainder))
            return events
        return self._consume_text(delta)

    def flush(self):
        events = []
        if self._buf:
            events.append(("text", self._buf))
            self._buf = ""
        if self._in_fence:
            log.warning("json_codeblock streaming: stream ended mid fence")
            events.append(("text", self.FENCE + self._body))
            self._body = ""
            self._in_fence = False
        return events

    def _consume_text(self, text: str):
        events = []
        self._buf += text
        while True:
            i = self._buf.find(self.FENCE)
            if i == -1:
                # Hold trailing 1-2 backticks: they might be the start of a
                # fence completing in the next delta.
                hold = 0
                if self._buf.endswith("``"):
                    hold = 2
                elif self._buf.endswith("`"):
                    hold = 1
                if hold and len(self._buf) > hold:
                    events.append(("text", self._buf[:-hold]))
                    self._buf = self._buf[-hold:]
                elif not hold and self._buf:
                    events.append(("text", self._buf))
                    self._buf = ""
                return events

            head = self._buf[:i]
            tail = self._buf[i + len(self.FENCE):]
            self._buf = ""
            if head:
                events.append(("text", head))
            self._in_fence = True

            j = tail.find(self.FENCE)
            if j == -1:
                self._body = tail
                return events

            body = tail[:j]
            remainder = tail[j + len(self.FENCE):]
            self._in_fence = False
            events.extend(self._parse_body(body))
            self._buf = remainder
            # loop in case the remainder contains another fence

    def _parse_body(self, body: str):
        # Strip optional language tag at the start (json / tool_call / tool_calls).
        stripped = body.lstrip()
        for tag in self.LANG_TAGS:
            if stripped.startswith(tag):
                after = stripped[len(tag):]
                # Tag must be followed by whitespace, newline, or end
                if after == "" or after[0] in (" ", "\t", "\n", "\r"):
                    stripped = after.lstrip()
                    break
        stripped = stripped.strip()

        if not stripped:
            return [("text", self.FENCE + body + self.FENCE)]

        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            # Real code sample, not a tool call — emit fence intact so the
            # client sees what the model wrote.
            return [("text", self.FENCE + body + self.FENCE)]

        candidates = _coerce_call_objects(obj)
        events = []
        for cand in candidates:
            if not _looks_like_tool_call(cand):
                continue
            name = cand["name"]
            if not _tool_name_allowed(name, self._tool_names):
                continue
            events.append((
                "tool_call",
                ToolCall(function=FunctionCall(name=name, arguments=_extract_args(cand))),
            ))

        if not events:
            return [("text", self.FENCE + body + self.FENCE)]
        return events


class _BufferedFallbackStreamingParser:
    """Generic streaming parser that defers parsing to end of stream.

    Used for formats without a dedicated incremental parser (qwen3_coder,
    llama, mistral, auto). Buffers every delta; emits nothing during the
    stream; at flush() runs the non-streaming parser on the full buffer and
    emits the resulting text + tool_call events in order.

    Trade: tool calls don't appear progressively for these formats. For the
    block-based formats this matters approximately zero — the call markup is
    a single block at the end of the response anyway, so there's nothing
    useful to emit before it completes.
    """

    def __init__(
        self,
        parse_fn: ParserFn,
        parser_name: str,
        tool_names: Optional[Set[str]] = None,
    ):
        self._parse_fn = parse_fn
        self._parser_name = parser_name
        self._tool_names = tool_names
        self._buf: List[str] = []

    def feed(self, delta: str):
        if delta:
            self._buf.append(delta)
        return []  # no progressive events

    def flush(self):
        text = "".join(self._buf)
        self._buf.clear()
        if not text:
            return []
        cleaned, calls = self._parse_fn(text, self._tool_names)
        events: List[Tuple[str, object]] = []
        if cleaned:
            events.append(("text", cleaned))
        for call in calls:
            events.append(("tool_call", call))
        return events


# Dedicated streaming parsers — emit events progressively.
_DEDICATED_STREAMING_PARSERS = {
    "hermes":         StreamingHermesParser,
    "json_codeblock": StreamingJsonCodeblockParser,
}


def make_streaming_parser(
    parser_name: str,
    tool_names: Optional[Set[str]] = None,
):
    """Construct a streaming parser. Dedicated incremental parsers are used
    for ``hermes`` and ``json_codeblock``; everything else (including
    ``auto``) falls back to buffer-then-parse-at-flush."""
    if parser_name in _DEDICATED_STREAMING_PARSERS:
        return _DEDICATED_STREAMING_PARSERS[parser_name](tool_names=tool_names)
    parse_fn = PARSERS.get(parser_name)
    if parse_fn is None:
        raise ValueError(
            f"Unknown tool-call parser: {parser_name!r}. "
            f"Available: {sorted(PARSERS)}"
        )
    return _BufferedFallbackStreamingParser(
        parse_fn=parse_fn,
        parser_name=parser_name,
        tool_names=tool_names,
    )


# Backwards-compat alias — earlier code referenced STREAMING_PARSERS directly.
# It now lists everything, with non-dedicated parsers using the fallback shape.
STREAMING_PARSERS = dict(_DEDICATED_STREAMING_PARSERS)
