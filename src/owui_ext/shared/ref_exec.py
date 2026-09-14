"""Externalized-ref engine for the sub_agent tool loop.

Ported from the production Auto Compact pipe (commit
54bb3153659368a607df51c53216167de7f41021, functions/pipe/auto_compact.py blob
b3617497411c5518c840b9fa3b3959b00d4e0d8d).
"""

import asyncio
import codecs
import copy
import hashlib
import json
import logging
import re
import shlex
import threading
import time
from array import array
from bisect import bisect_left, bisect_right
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Iterable, Literal, TypeAlias

try:
    import regex as _REGEX
except ImportError:
    _REGEX = None

ref_exec_log = logging.getLogger("owui_ext.shared.ref_exec")

LOG = ref_exec_log

RefKind = Literal["history", "tool"]

REF_TEXT_HASH_CHUNK_CHARS = 16 * 1024
REF_EXEC_TOOL_NAME = "agent_ref_exec"
REF_EXEC_COMMAND_MAX_BYTES = 1_024
REF_EXEC_RESPONSE_MAX_BYTES = 65_536
REF_EXEC_TAIL_MAX_BYTES = 8 * 1024 * 1024
# Retained line objects cost ~200 bytes each regardless of text length,
# so cap the window by line count (~15 MiB) as well.
REF_EXEC_TAIL_MAX_LINES = 65_536
REF_EXEC_USAGE_ERROR = (
    "Error: usage: agent_ref_exec(command). Expected REF: "
    "tool:<64 hex> or history:<64 hex>"
)
REF_EXEC_REGEX_BUDGET_SECONDS = 2.0
REF_EXEC_COMMANDS = ("cat", "grep", "head", "ls", "sed", "stat", "tail", "wc")
REF_EXEC_CLASSIFY_EXACT_ENCODE_MAX_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class ParsedRef:
    kind: RefKind
    value: str


@dataclass(frozen=True, slots=True)
class RefManifest:
    ref: str
    utf8_bytes: int | None
    sha256: str


@dataclass(frozen=True, slots=True)
class ZeroCopySourceHandle:
    text: str


@dataclass(frozen=True, slots=True)
class JsonlHistorySourceHandle:
    """Run-local history ref: one canonical JSON record per line."""

    records: tuple[str, ...]
    utf8_bytes: int
    sha256: str
    line_count: int

    def iter_records(self) -> Iterable[str]:
        return self.records


RefSourceHandle = ZeroCopySourceHandle | JsonlHistorySourceHandle


@dataclass(frozen=True, slots=True)
class RefCatalogEntry:
    manifest: RefManifest
    source: RefSourceHandle
    preview_text: str | None = None


@dataclass(frozen=True, slots=True)
class InvalidRefTextClassificationError(ValueError):
    def __str__(self) -> str:
        return "Eligible ref text classification requires complete measurement"


@dataclass(frozen=True, slots=True)
class RefTextClassification:
    eligible: bool
    utf8_bytes: int | None
    sha256: str | None
    line_count: int | None
    token_count: int | None
    encoder_failed: bool

    def __post_init__(self) -> None:
        if self.eligible and (
            self.utf8_bytes is None
            or self.sha256 is None
            or self.line_count is None
        ):
            raise InvalidRefTextClassificationError


@dataclass(frozen=True, slots=True)
class RefProjectionPlan:
    catalog: tuple[RefCatalogEntry, ...]
    manifests: tuple[RefManifest, ...]
    reader_schema: MappingProxyType | None = None


@dataclass(frozen=True, slots=True)
class RefProjectionError(RuntimeError):
    stage: str

    def __str__(self) -> str:
        return f"Externalized ref {self.stage} failed before provider forward"


@dataclass(frozen=True, slots=True)
class RefExecStage:
    command: str
    ref: str | None = None
    count: int | None = None
    flags: frozenset[str] = frozenset()
    pattern: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    list_kind: RefKind | None = None
    byte_count: int | None = None
    byte_start: int | None = None


@dataclass(frozen=True, slots=True)
class RefExecByteRange:
    requested_start: int
    requested_end: int | None
    actual_start: int
    actual_end: int | None
    marked: bool = False
    actual_empty: bool = False


@dataclass(frozen=True, slots=True)
class RefExecComponent:
    kind: Literal["display_prefix", "text", "synthetic_lf"]
    text: str
    start: int
    end: int
    utf8_bytes: int
    source_byte_start: int | None = None
    source_char_start: int | None = None


@dataclass(frozen=True, slots=True)
class RefExecComponentView:
    components: tuple[RefExecComponent, ...]


@dataclass(frozen=True, slots=True)
class RefExecLine:
    text: str
    number: int
    byte_start: int
    char_start: int
    has_newline: bool
    match_start: int | None = None
    match_end: int | None = None
    display_prefix: str = ""
    byte_range: RefExecByteRange | None = None
    metadata_only: bool = False
    atomic_match: bool = False
    component_view: RefExecComponentView | None = None


@dataclass(frozen=True, slots=True)
class RefExecError(RuntimeError):
    message: str

    def __str__(self) -> str:
        return self.message
def parse_ref(value: str) -> ParsedRef | None:
    tool_match = re.fullmatch(r"tool:([0-9a-f]{64})", value)
    if tool_match is not None:
        return ParsedRef(kind="tool", value=tool_match.group(1))
    history_match = re.fullmatch(r"history:([0-9a-f]{64})", value)
    if history_match is not None:
        return ParsedRef(kind="history", value=history_match.group(1))
    return None


def _split_ref_exec_pipeline(command: str) -> tuple[str, ...]:
    stages: list[str] = []
    buffer: list[str] = []
    single_quoted = False
    double_quoted = False
    escaped = False
    for character in command:
        if escaped:
            buffer.append(character)
            escaped = False
            continue
        if character == "\\" and not single_quoted:
            buffer.append(character)
            escaped = True
            continue
        if character == "'" and not double_quoted:
            single_quoted = not single_quoted
            buffer.append(character)
            continue
        if character == '"' and not single_quoted:
            double_quoted = not double_quoted
            buffer.append(character)
            continue
        if character == "|" and not single_quoted and not double_quoted:
            stage = "".join(buffer).strip()
            if not stage:
                raise RefExecError("Error: empty pipeline stage")
            stages.append(stage)
            buffer = []
            continue
        buffer.append(character)
    if escaped or single_quoted or double_quoted:
        raise RefExecError("Error: malformed quote or escape in command")
    stage = "".join(buffer).strip()
    if not stage:
        raise RefExecError("Error: empty command or pipeline stage")
    stages.append(stage)
    return tuple(stages)


def _parse_ref_exec_count(
    tokens: list[str],
    *,
    command: str,
) -> tuple[int | None, int | None, int | None, list[str]]:
    remaining = list(tokens)
    line_count: int | None = 10
    byte_count: int | None = None
    byte_start: int | None = None
    if remaining and remaining[0] == "-n":
        if len(remaining) < 2 or re.fullmatch(r"[0-9]+", remaining[1]) is None:
            raise RefExecError(
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        line_count = int(remaining[1])
        remaining = remaining[2:]
    elif remaining and remaining[0] == "-c":
        valid_fixed = len(remaining) >= 2 and re.fullmatch(
            r"[0-9]+", remaining[1]
        ) is not None
        valid_start = (
            command == "tail"
            and len(remaining) >= 2
            and re.fullmatch(r"\+[1-9][0-9]*", remaining[1]) is not None
        )
        if not valid_fixed and not valid_start:
            raise RefExecError(
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        line_count = None
        if valid_start:
            byte_start = int(remaining[1][1:])
        else:
            byte_count = int(remaining[1])
        remaining = remaining[2:]
    elif remaining and re.fullmatch(r"-[0-9]+", remaining[0]) is not None:
        line_count = int(remaining[0][1:])
        remaining = remaining[1:]
    return line_count, byte_count, byte_start, remaining


def _parse_ref_exec_grep(tokens: list[str], *, source: bool) -> RefExecStage:
    flags: set[str] = set()
    remaining = list(tokens)
    while remaining and remaining[0].startswith("-") and remaining[0] != "-":
        token = remaining.pop(0)
        if token == "--":
            break
        combined = token[1:]
        if not combined or any(flag not in "Einco" for flag in combined):
            raise RefExecError(
                "Error: usage: grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        flags.update(combined)
    expected = 2 if source else 1
    if len(remaining) != expected:
        raise RefExecError(
            "Error: usage: grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    ref = remaining[1] if source else None
    if ref is not None and parse_ref(ref) is None:
        raise RefExecError(
            "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    return RefExecStage(
        command="grep",
        ref=ref,
        flags=frozenset(flags),
        pattern=remaining[0],
    )


def _parse_ref_exec_sed(tokens: list[str], *, source: bool) -> RefExecStage:
    expected = 3 if source else 2
    if len(tokens) != expected or tokens[0] != "-n":
        raise RefExecError(
            "Error: usage: sed -n Np|M,Np|M,$p [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    selection = tokens[1]
    match = re.fullmatch(r"([1-9][0-9]*)(?:,([1-9][0-9]*|\$))?p", selection)
    if match is None:
        raise RefExecError(
            "Error: usage: sed -n Np|M,Np|M,$p [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    start = int(match.group(1))
    end_value = match.group(2)
    end = start if end_value is None else (None if end_value == "$" else int(end_value))
    if end is not None and start > end:
        raise RefExecError("Error: sed range start exceeds end")
    ref = tokens[2] if source else None
    if ref is not None and parse_ref(ref) is None:
        raise RefExecError(
            "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    return RefExecStage(command="sed", ref=ref, start_line=start, end_line=end)


def _parse_ref_exec_stage(stage: str, *, source: bool) -> RefExecStage:
    try:
        tokens = shlex.split(stage, posix=True)
    except ValueError as exc:
        raise RefExecError("Error: malformed quote in command") from exc
    if not tokens:
        raise RefExecError("Error: empty pipeline stage")
    command = tokens[0]
    arguments = tokens[1:]
    if command not in REF_EXEC_COMMANDS:
        available = ", ".join(REF_EXEC_COMMANDS)
        raise RefExecError(
            f"Error: unknown command. Available: {available}. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    if not source and command not in {"grep", "head", "sed", "tail", "wc"}:
        raise RefExecError("Error: command is not a valid piped consumer")
    if command == "ls":
        if len(arguments) > 1 or (arguments and arguments[0] not in {"history", "tool"}):
            raise RefExecError(
                "Error: usage: ls [history|tool]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        kind: RefKind | None = arguments[0] if arguments else None
        return RefExecStage(command=command, list_kind=kind)
    if command == "grep":
        return _parse_ref_exec_grep(arguments, source=source)
    if command == "sed":
        return _parse_ref_exec_sed(arguments, source=source)
    if command in {"head", "tail"}:
        count, byte_count, byte_start, remaining = _parse_ref_exec_count(
            arguments, command=command
        )
        expected = 1 if source else 0
        if len(remaining) != expected:
            raise RefExecError(
                f"Error: usage: {command} [-n N|-N|-c N{'|-c +N' if command == 'tail' else ''}] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        ref = remaining[0] if source else None
        if ref is not None and parse_ref(ref) is None:
            raise RefExecError(
                "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        return RefExecStage(
            command=command,
            ref=ref,
            count=count,
            byte_count=byte_count,
            byte_start=byte_start,
        )
    if command == "wc":
        if not arguments or arguments[0] not in {"-l", "-w", "-c"}:
            raise RefExecError(
                "Error: usage: wc [-l|-w|-c] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        expected = 2 if source else 1
        if len(arguments) != expected:
            raise RefExecError(
                "Error: usage: wc [-l|-w|-c] [REF]. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        ref = arguments[1] if source else None
        if ref is not None and parse_ref(ref) is None:
            raise RefExecError(
                "Error: invalid externalized ref. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        return RefExecStage(command=command, ref=ref, flags=frozenset({arguments[0][1:]}))
    if len(arguments) != 1 or parse_ref(arguments[0]) is None:
        raise RefExecError(
            f"Error: usage: {command} REF. Expected REF: tool:<64 hex> or history:<64 hex>"
        )
    return RefExecStage(command=command, ref=arguments[0])


def _parse_ref_exec_command(command: str) -> tuple[RefExecStage, ...]:
    if not command.strip():
        raise RefExecError(REF_EXEC_USAGE_ERROR)
    try:
        encoded_command = command.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RefExecError(REF_EXEC_USAGE_ERROR) from exc
    if len(encoded_command) > REF_EXEC_COMMAND_MAX_BYTES:
        raise RefExecError("Error: command exceeds the 1,024 UTF-8 byte parser limit")
    return tuple(
        _parse_ref_exec_stage(stage, source=index == 0)
        for index, stage in enumerate(_split_ref_exec_pipeline(command))
    )


def _check_ref_exec_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise RefExecError("Error: reader command was cancelled")


def _encode_ref_text_checked(text: str, cancelled: threading.Event) -> bytes:
    _check_ref_exec_cancelled(cancelled)
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise RefExecError(
            "Error: externalized ref source is not valid UTF-8"
        ) from exc


def _measure_ref_text_checked(text: str, cancelled: threading.Event) -> tuple[int, str]:
    digest = hashlib.sha256()
    utf8_bytes = 0
    for offset in range(0, len(text), REF_TEXT_HASH_CHUNK_CHARS):
        encoded = _encode_ref_text_checked(
            text[offset : offset + REF_TEXT_HASH_CHUNK_CHARS], cancelled
        )
        utf8_bytes += len(encoded)
        digest.update(encoded)
    return utf8_bytes, digest.hexdigest()


def _measure_ref_text_parts_checked(
    parts: tuple[str, ...],
    cancelled: threading.Event,
) -> tuple[int, str]:
    digest = hashlib.sha256()
    utf8_bytes = 0
    for part in parts:
        for offset in range(0, len(part), REF_TEXT_HASH_CHUNK_CHARS):
            encoded = _encode_ref_text_checked(
                part[offset : offset + REF_TEXT_HASH_CHUNK_CHARS], cancelled
            )
            utf8_bytes += len(encoded)
            digest.update(encoded)
    return utf8_bytes, digest.hexdigest()


def _ref_exec_utf8_range_bytes(
    text: str,
    start: int,
    end: int,
    cancelled: threading.Event,
) -> int:
    utf8_bytes = 0
    for offset in range(start, end, REF_TEXT_HASH_CHUNK_CHARS):
        chunk_end = min(end, offset + REF_TEXT_HASH_CHUNK_CHARS)
        utf8_bytes += len(_encode_ref_text_checked(text[offset:chunk_end], cancelled))
    return utf8_bytes


def _ref_exec_utf8_prefix_index(
    text: str,
    max_bytes: int,
    cancelled: threading.Event,
) -> tuple[int, int]:
    offset = 0
    retained_bytes = 0
    while offset < len(text) and retained_bytes < max_bytes:
        chunk_end = min(len(text), offset + REF_TEXT_HASH_CHUNK_CHARS)
        chunk = text[offset:chunk_end]
        chunk_bytes = len(_encode_ref_text_checked(chunk, cancelled))
        if retained_bytes + chunk_bytes <= max_bytes:
            retained_bytes += chunk_bytes
            offset = chunk_end
            continue
        low = offset
        high = chunk_end
        while low < high:
            midpoint = (low + high + 1) // 2
            candidate_bytes = _ref_exec_utf8_range_bytes(text, offset, midpoint, cancelled)
            if retained_bytes + candidate_bytes <= max_bytes:
                low = midpoint
            else:
                high = midpoint - 1
        retained_bytes += _ref_exec_utf8_range_bytes(text, offset, low, cancelled)
        offset = low
        break
    return offset, retained_bytes


def _ref_exec_utf8_prefix(
    text: str,
    max_bytes: int,
    cancelled: threading.Event,
) -> str:
    offset, _ = _ref_exec_utf8_prefix_index(text, max_bytes, cancelled)
    return text[:offset]


def _iter_ref_text_lines(text: str, cancelled: threading.Event) -> Iterable[RefExecLine]:
    offset = 0
    line_number = 1
    byte_start = 0
    while offset < len(text):
        _check_ref_exec_cancelled(cancelled)
        newline = text.find("\n", offset)
        end = len(text) if newline < 0 else newline
        line = text[offset:end]
        yield RefExecLine(
            text=line,
            number=line_number,
            byte_start=byte_start,
            char_start=offset,
            has_newline=newline >= 0,
        )
        encoded_bytes, _ = _measure_ref_text_checked(line, cancelled)
        byte_start += encoded_bytes + (1 if newline >= 0 else 0)
        offset = end + (1 if newline >= 0 else 0)
        line_number += 1


def _measure_ref_source_checked(
    source: RefSourceHandle,
    cancelled: threading.Event,
) -> tuple[int, str]:
    if isinstance(source, ZeroCopySourceHandle):
        return _measure_ref_text_checked(source.text, cancelled)
    digest = hashlib.sha256()
    utf8_bytes = 0
    emitted = 0
    for record in source.iter_records():
        _check_ref_exec_cancelled(cancelled)
        if emitted > 0:
            digest.update(b"\n")
            utf8_bytes += 1
        for offset in range(0, len(record), REF_TEXT_HASH_CHUNK_CHARS):
            encoded = _encode_ref_text_checked(
                record[offset : offset + REF_TEXT_HASH_CHUNK_CHARS], cancelled
            )
            digest.update(encoded)
            utf8_bytes += len(encoded)
        emitted += 1
    if emitted != source.line_count:
        raise RefExecError("Error: externalized ref integrity verification failed")
    return utf8_bytes, digest.hexdigest()


def _iter_ref_source_lines(
    source: RefSourceHandle,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    if isinstance(source, ZeroCopySourceHandle):
        yield from _iter_ref_text_lines(source.text, cancelled)
        return
    byte_start = 0
    char_start = 0
    for line_number, record in enumerate(source.iter_records(), 1):
        _check_ref_exec_cancelled(cancelled)
        has_newline = line_number < source.line_count
        yield RefExecLine(
            text=record,
            number=line_number,
            byte_start=byte_start,
            char_start=char_start,
            has_newline=has_newline,
        )
        record_bytes, _ = _measure_ref_text_checked(record, cancelled)
        byte_start += record_bytes + int(has_newline)
        char_start += len(record) + int(has_newline)


def _iter_verified_ref_source_lines(
    entry: RefCatalogEntry,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    digest = hashlib.sha256()
    utf8_bytes = 0
    iterator = iter(_iter_ref_source_lines(entry.source, cancelled))

    def update(line: RefExecLine) -> None:
        nonlocal utf8_bytes
        for offset in range(0, len(line.text), REF_TEXT_HASH_CHUNK_CHARS):
            encoded = _encode_ref_text_checked(
                line.text[offset : offset + REF_TEXT_HASH_CHUNK_CHARS], cancelled
            )
            digest.update(encoded)
            utf8_bytes += len(encoded)
        if line.has_newline:
            digest.update(b"\n")
            utf8_bytes += 1

    try:
        current = next(iterator)
    except StopIteration:
        current = None
    if current is not None:
        for following in iterator:
            update(current)
            yield current
            current = following
        update(current)
    if utf8_bytes != entry.manifest.utf8_bytes or digest.hexdigest() != entry.manifest.sha256:
        raise RefExecError("Error: externalized ref integrity verification failed")
    if current is not None:
        yield current


def _ref_exec_head(lines: Iterable[RefExecLine], count: int, cancelled: threading.Event) -> Iterable[RefExecLine]:
    selected = 0
    iterator = iter(lines)
    while selected < count:
        try:
            line = next(iterator)
        except StopIteration:
            return
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            yield line
            continue
        yield line
        selected += 1


def _ref_exec_source_line_count(source: RefSourceHandle) -> int:
    if isinstance(source, ZeroCopySourceHandle):
        text = source.text
        return text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    return source.line_count


def _ref_exec_tail(lines: Iterable[RefExecLine], count: int, cancelled: threading.Event) -> Iterable[RefExecLine]:
    retained: deque[tuple[RefExecLine, int]] = deque()
    retained_bytes = 0
    seen = 0
    metadata: RefExecLine | None = None
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            metadata = line
            continue
        seen += 1
        line_bytes = _ref_exec_presented_line_bytes(line, cancelled)
        retained.append((line, line_bytes))
        retained_bytes += line_bytes
        while (
            len(retained) > count
            or retained_bytes > REF_EXEC_TAIL_MAX_BYTES
            or len(retained) > REF_EXEC_TAIL_MAX_LINES
        ):
            _dropped, dropped_bytes = retained.popleft()
            retained_bytes -= dropped_bytes
    if len(retained) < min(count, seen):
        raise RefExecError(
            "Error: tail line window exceeds 8 MiB or 65,536 lines; reduce -n or filter first"
        )
    for line, _ in retained:
        _check_ref_exec_cancelled(cancelled)
        yield line
    if metadata is not None and count > 0:
        yield metadata


def _ref_exec_component(
    kind: Literal["display_prefix", "text", "synthetic_lf"],
    text: str,
    cancelled: threading.Event,
    *,
    source_start: tuple[int, int] | None = None,
) -> RefExecComponent:
    utf8_bytes, _ = _measure_ref_text_checked(text, cancelled)
    return RefExecComponent(
        kind=kind,
        text=text,
        start=0,
        end=len(text),
        utf8_bytes=utf8_bytes,
        source_byte_start=source_start[0] if source_start is not None else None,
        source_char_start=source_start[1] if source_start is not None else None,
    )


def _ref_exec_line_component_view(
    line: RefExecLine,
    cancelled: threading.Event,
) -> RefExecComponentView:
    if line.component_view is not None:
        return line.component_view
    components: list[RefExecComponent] = []
    if line.display_prefix:
        components.append(
            _ref_exec_component("display_prefix", line.display_prefix, cancelled)
        )
    text_component = _ref_exec_component(
        "text",
        line.text,
        cancelled,
        source_start=(line.byte_start, line.char_start),
    )
    if line.text:
        components.append(text_component)
    if line.has_newline:
        components.append(
            _ref_exec_component(
                "synthetic_lf",
                "\n",
                cancelled,
                source_start=(
                    line.byte_start + text_component.utf8_bytes,
                    line.char_start + len(line.text),
                ),
            )
        )
    return RefExecComponentView(tuple(components))


def _ref_exec_materialize_components(
    components: Iterable[RefExecComponent],
    *,
    include_synthetic_lf: bool = True,
) -> str:
    return "".join(
        component.text[component.start : component.end]
        for component in components
        if include_synthetic_lf or component.kind != "synthetic_lf"
    )


def _ref_exec_materialize_line(
    line: RefExecLine,
    cancelled: threading.Event,
    *,
    include_synthetic_lf: bool = True,
) -> str:
    if line.component_view is None:
        return line.display_prefix + line.text + (
            "\n" if include_synthetic_lf and line.has_newline else ""
        )
    _check_ref_exec_cancelled(cancelled)
    return _ref_exec_materialize_components(
        line.component_view.components,
        include_synthetic_lf=include_synthetic_lf,
    )


def _ref_exec_component_prefix_index(
    component: RefExecComponent,
    max_bytes: int,
    cancelled: threading.Event,
) -> tuple[int, int]:
    offset = component.start
    retained_bytes = 0
    while offset < component.end and retained_bytes < max_bytes:
        chunk_end = min(component.end, offset + REF_TEXT_HASH_CHUNK_CHARS)
        chunk_bytes = _ref_exec_utf8_range_bytes(
            component.text, offset, chunk_end, cancelled
        )
        if retained_bytes + chunk_bytes <= max_bytes:
            retained_bytes += chunk_bytes
            offset = chunk_end
            continue
        low = offset
        high = chunk_end
        while low < high:
            midpoint = (low + high + 1) // 2
            candidate_bytes = _ref_exec_utf8_range_bytes(
                component.text, offset, midpoint, cancelled
            )
            if retained_bytes + candidate_bytes <= max_bytes:
                low = midpoint
            else:
                high = midpoint - 1
        retained_bytes += _ref_exec_utf8_range_bytes(
            component.text, offset, low, cancelled
        )
        offset = low
        break
    return offset, retained_bytes


def _ref_exec_presented_line_bytes(
    line: RefExecLine,
    cancelled: threading.Event,
) -> int:
    if line.component_view is not None:
        return sum(component.utf8_bytes for component in line.component_view.components)
    encoded_bytes, _ = _measure_ref_text_parts_checked(
        (line.display_prefix, line.text), cancelled
    )
    return encoded_bytes + int(line.has_newline)


def _ref_exec_source_byte_bounds(
    line: RefExecLine,
) -> tuple[int, int] | None:
    if line.component_view is None:
        return None
    backed = tuple(
        component
        for component in line.component_view.components
        if component.source_byte_start is not None
    )
    if not backed:
        return None
    return (
        backed[0].source_byte_start + 1,
        backed[-1].source_byte_start + backed[-1].utf8_bytes,
    )


def _ref_exec_slice_presented_line(
    line: RefExecLine,
    *,
    stream_start: int,
    selected_start: int,
    selected_end: int,
    cancelled: threading.Event,
) -> tuple[RefExecLine | None, int, int, bool]:
    actual_start: int | None = None
    actual_end = stream_start
    snapped = False
    component_start = stream_start
    selected_components: list[RefExecComponent] = []
    for component in _ref_exec_line_component_view(line, cancelled).components:
        component_end = component_start + component.utf8_bytes
        overlap_start = max(selected_start, component_start)
        overlap_end = min(selected_end, component_end)
        if overlap_start < overlap_end:
            local_start = overlap_start - component_start
            local_end = overlap_end - component_start
            start_index, start_floor = _ref_exec_component_prefix_index(
                component, local_start, cancelled
            )
            if start_floor < local_start:
                start_index += 1
                snapped = True
            actual_local_start = _ref_exec_utf8_range_bytes(
                component.text, component.start, start_index, cancelled
            )
            end_index, actual_local_end = _ref_exec_component_prefix_index(
                component, local_end, cancelled
            )
            if actual_local_end < local_end:
                snapped = True
            if start_index < end_index:
                part_start = component_start + actual_local_start
                if actual_start is None:
                    actual_start = part_start
                actual_end = component_start + actual_local_end
                selected_components.append(
                    replace(
                        component,
                        start=start_index,
                        end=end_index,
                        utf8_bytes=actual_local_end - actual_local_start,
                        source_byte_start=(
                            component.source_byte_start + actual_local_start
                            if component.source_byte_start is not None
                            else None
                        ),
                        source_char_start=(
                            component.source_char_start
                            + start_index
                            - component.start
                            if component.source_char_start is not None
                            else None
                        ),
                    )
                )
        component_start = component_end
    if actual_start is None:
        if not snapped:
            return None, selected_end, selected_end, False
        return (
            replace(
                line,
                text="",
                byte_start=selected_end,
                has_newline=False,
                match_start=None,
                match_end=None,
                display_prefix="",
                byte_range=None,
                metadata_only=True,
                component_view=RefExecComponentView(()),
            ),
            selected_end,
            selected_end,
            True,
        )
    return (
        replace(
            line,
            text="",
            byte_start=next(
                (
                    component.source_byte_start
                    for component in selected_components
                    if component.kind == "text"
                    and component.source_byte_start is not None
                ),
                line.byte_start,
            ),
            char_start=next(
                (
                    component.source_char_start
                    for component in selected_components
                    if component.kind == "text"
                    and component.source_char_start is not None
                ),
                line.char_start,
            ),
            has_newline=False,
            match_start=None,
            match_end=None,
            display_prefix="",
            byte_range=None,
            component_view=RefExecComponentView(tuple(selected_components)),
        ),
        actual_start,
        actual_end,
        snapped,
    )


def _ref_exec_head_bytes(
    lines: Iterable[RefExecLine],
    count: int,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    if count == 0:
        return
    stream_start = 0
    pending: RefExecLine | None = None
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        line_end = stream_start + _ref_exec_presented_line_bytes(line, cancelled)
        selected, actual_start, actual_end, snapped = _ref_exec_slice_presented_line(
            line,
            stream_start=stream_start,
            selected_start=0,
            selected_end=min(count, line_end),
            cancelled=cancelled,
        )
        if selected is not None:
            source_bounds = (
                _ref_exec_source_byte_bounds(selected)
                if line.component_view is not None
                else None
            )
            range_start = (
                source_bounds[0] if source_bounds is not None else actual_start + 1
            )
            range_end = source_bounds[1] if source_bounds is not None else actual_end
            ranged = replace(
                selected,
                atomic_match=False,
                byte_range=RefExecByteRange(
                    requested_start=1,
                    requested_end=count,
                    actual_start=range_start,
                    actual_end=range_end,
                    marked=snapped or source_bounds is not None,
                    actual_empty=snapped and actual_start == actual_end,
                ),
            )
            if (
                ranged.metadata_only
                and pending is not None
                and pending.byte_range is not None
            ):
                pending = replace(
                    pending,
                    byte_range=replace(
                        pending.byte_range,
                        marked=pending.byte_range.marked or snapped,
                    ),
                )
            else:
                if pending is not None:
                    yield pending
                pending = ranged
        if line_end >= count:
            if pending is not None:
                yield pending
            return
        stream_start = line_end
    if pending is not None:
        yield pending


def _ref_exec_tail_from_bytes(
    lines: Iterable[RefExecLine],
    start: int,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    selected_start = start - 1
    stream_start = 0
    actual_range_start: int | None = None
    snapped_start = False
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            continue
        line_end = stream_start + _ref_exec_presented_line_bytes(line, cancelled)
        if line_end <= selected_start:
            stream_start = line_end
            continue
        selected, actual_start, actual_end, snapped = _ref_exec_slice_presented_line(
            line,
            stream_start=stream_start,
            selected_start=selected_start,
            selected_end=line_end,
            cancelled=cancelled,
        )
        if selected is not None:
            if actual_range_start is None:
                actual_range_start = actual_start + 1
                snapped_start = snapped
            yield replace(
                selected,
                atomic_match=False,
                byte_range=RefExecByteRange(
                    requested_start=start,
                    requested_end=None,
                    actual_start=actual_range_start,
                    actual_end=None,
                    marked=snapped_start,
                    actual_empty=snapped and actual_start == actual_end,
                ),
            )
        stream_start = line_end


@dataclass(frozen=True, slots=True)
class RefExecTailSnapshot:
    retained: bytearray
    write_offset: int
    component_lengths: array
    component_flags: bytearray
    component_source_delta_indexes: array
    component_source_deltas: array
    component_char_delta_indexes: array
    component_char_deltas: array
    record_component_ends: array
    record_flags: bytearray
    record_char_delta_indexes: array
    record_char_deltas: array
    record_number_delta_indexes: array
    record_number_deltas: array
    record_match_starts: array | None
    record_match_ends: array | None
    trailing_plain_default: bool
    trailing_plain_change_indexes: array
    trailing_number_delta_indexes: array
    trailing_number_deltas: array
    trailing_char_delta_indexes: array
    trailing_char_deltas: array
    trailing_match_indexes: array
    trailing_match_starts: array
    trailing_match_ends: array
    compact_missing: int
    retained_bytes: int
    snapped_bytes: int
    stream_start: int
    stream_chars: int
    line_count: int
    trailing_empty_lines: int
    snapped_empty: bool
    snapped_number: int
    snapped_char_start: int
    partial_fallback_char_start: int | None
    count: int


def _iter_ref_exec_tail_snapshot(
    snapshot: RefExecTailSnapshot,
) -> Iterable[RefExecLine]:
    effective_bytes = snapshot.retained_bytes - snapshot.snapped_bytes
    oldest_offset = snapshot.write_offset if snapshot.retained_bytes == snapshot.count else 0
    payload_start = (oldest_offset + snapshot.snapped_bytes) % snapshot.count
    first_bytes = min(effective_bytes, snapshot.count - payload_start)
    decoder = codecs.getincrementaldecoder("utf-8")()
    presented = decoder.decode(
        memoryview(snapshot.retained)[payload_start : payload_start + first_bytes],
        final=first_bytes == effective_bytes,
    )
    if first_bytes < effective_bytes:
        presented += decoder.decode(
            memoryview(snapshot.retained)[: effective_bytes - first_bytes],
            final=True,
        )
    requested_start = max(1, snapshot.stream_start - snapshot.count + 1)
    actual_start = (
        snapshot.stream_start
        - snapshot.retained_bytes
        + snapshot.snapped_bytes
        + 1
    )
    byte_range = RefExecByteRange(
        requested_start=requested_start,
        requested_end=snapshot.stream_start,
        actual_start=actual_start,
        actual_end=snapshot.stream_start,
        marked=actual_start != requested_start,
        actual_empty=actual_start > snapshot.stream_start,
    )
    if not presented and (snapshot.snapped_bytes or snapshot.snapped_empty):
        yield RefExecLine(
            text="",
            number=snapshot.snapped_number,
            byte_start=snapshot.stream_start,
            char_start=snapshot.snapped_char_start,
            has_newline=False,
            byte_range=byte_range,
            metadata_only=True,
            component_view=RefExecComponentView(()),
        )
        if snapshot.trailing_empty_lines == 0:
            return
    output_line_count = (
        len(snapshot.record_component_ends) + snapshot.trailing_empty_lines
    )
    output_number = snapshot.line_count - output_line_count + 1
    presented_char_start = snapshot.stream_chars - len(presented)
    line_byte_start = actual_start - 1
    byte_offset = 0
    char_offset = 0
    component_offset = 0
    source_delta_offset = 0
    char_delta_offset = 0
    number_delta_offset = 0
    record_char_delta_offset = 0
    for record_offset, component_end in enumerate(snapshot.record_component_ends):
        record_flags = snapshot.record_flags[record_offset]
        record_start = bool(record_flags & 0x01)
        record_has_newline = bool(record_flags & 0x02)
        record_plain = bool(record_flags & 0x04)
        record_char_delta = 0
        if (
            record_char_delta_offset < len(snapshot.record_char_delta_indexes)
            and snapshot.record_char_delta_indexes[record_char_delta_offset]
            == record_offset
        ):
            record_char_delta = snapshot.record_char_deltas[
                record_char_delta_offset
            ]
            record_char_delta_offset += 1
        fallback_char_start = presented_char_start + char_offset + record_char_delta
        components: list[RefExecComponent] = []
        while component_offset < component_end:
            component_length = snapshot.component_lengths[component_offset]
            flags = snapshot.component_flags[component_offset]
            kind = flags & 0x03
            component_byte_end = byte_offset + component_length
            component_ring_start = (payload_start + byte_offset) % snapshot.count
            component_first_bytes = min(
                component_length, snapshot.count - component_ring_start
            )
            component_decoder = codecs.getincrementaldecoder("utf-8")()
            component_text = component_decoder.decode(
                memoryview(snapshot.retained)[
                    component_ring_start : component_ring_start
                    + component_first_bytes
                ],
                final=component_first_bytes == component_length,
            )
            if component_first_bytes < component_length:
                component_text += component_decoder.decode(
                    memoryview(snapshot.retained)[
                        : component_length - component_first_bytes
                    ],
                    final=True,
                )
            component_char_end = char_offset + len(component_text)
            source_delta = 0
            if flags & 0x08:
                source_delta = record_char_delta
            elif (
                source_delta_offset < len(snapshot.component_source_delta_indexes)
                and snapshot.component_source_delta_indexes[source_delta_offset]
                == component_offset
            ):
                source_delta = snapshot.component_source_deltas[source_delta_offset]
                source_delta_offset += 1
            source_backed = bool(flags & 0x04)
            source_char_delta = source_delta
            if (
                char_delta_offset < len(snapshot.component_char_delta_indexes)
                and snapshot.component_char_delta_indexes[char_delta_offset]
                == component_offset
            ):
                source_char_delta = snapshot.component_char_deltas[char_delta_offset]
                char_delta_offset += 1
            components.append(
                RefExecComponent(
                    kind=(
                        "display_prefix"
                        if kind == 0
                        else "text" if kind == 1 else "synthetic_lf"
                    ),
                    text=presented,
                    start=char_offset,
                    end=component_char_end,
                    utf8_bytes=component_length,
                    source_byte_start=(
                        actual_start - 1 + byte_offset + source_delta
                        if source_backed
                        else None
                    ),
                    source_char_start=(
                        presented_char_start + char_offset + source_char_delta
                        if source_backed
                        else None
                    ),
                )
            )
            char_offset = component_char_end
            byte_offset = component_byte_end
            component_offset += 1
        number_delta = 0
        if (
            number_delta_offset < len(snapshot.record_number_delta_indexes)
            and snapshot.record_number_delta_indexes[number_delta_offset]
            == record_offset
        ):
            number_delta = snapshot.record_number_deltas[number_delta_offset]
            number_delta_offset += 1
        history_fallback_char_start = fallback_char_start
        if (
            record_offset == 0
            and not record_start
            and snapshot.partial_fallback_char_start is not None
        ):
            history_fallback_char_start = snapshot.partial_fallback_char_start
        reconstructed_char_start = (
            history_fallback_char_start
            if record_start
            else next(
                (
                    component.source_char_start
                    for component in components
                    if component.kind == "text"
                    and component.source_char_start is not None
                ),
                history_fallback_char_start,
            )
        )
        match_start = (
            snapshot.record_match_starts[record_offset]
            if snapshot.record_match_starts is not None
            else snapshot.compact_missing
        )
        match_end = (
            snapshot.record_match_ends[record_offset]
            if snapshot.record_match_ends is not None
            else snapshot.compact_missing
        )
        plain_text = "".join(
            component.text[component.start : component.end]
            for component in components
            if component.kind == "text"
        )
        plain_display_prefix = "".join(
            component.text[component.start : component.end]
            for component in components
            if component.kind == "display_prefix"
        )
        yield RefExecLine(
            text=plain_text if record_plain else "",
            number=output_number + number_delta,
            byte_start=line_byte_start,
            char_start=reconstructed_char_start,
            has_newline=record_start
            and record_has_newline
            and bool(components)
            and components[-1].kind == "synthetic_lf",
            match_start=(
                match_start if match_start != snapshot.compact_missing else None
            ),
            match_end=match_end if match_end != snapshot.compact_missing else None,
            display_prefix=plain_display_prefix if record_plain else "",
            byte_range=byte_range,
            component_view=(
                None if record_plain else RefExecComponentView(tuple(components))
            ),
        )
        output_number += 1
        line_byte_start = actual_start - 1 + byte_offset
    trailing_plain = snapshot.trailing_plain_default
    trailing_plain_change_offset = 0
    trailing_number_delta_offset = 0
    trailing_char_delta_offset = 0
    trailing_match_offset = 0
    for trailing_offset in range(snapshot.trailing_empty_lines):
        if (
            trailing_plain_change_offset
            < len(snapshot.trailing_plain_change_indexes)
            and snapshot.trailing_plain_change_indexes[
                trailing_plain_change_offset
            ]
            == trailing_offset
        ):
            trailing_plain = not trailing_plain
            trailing_plain_change_offset += 1
        number_delta = 0
        if (
            trailing_number_delta_offset
            < len(snapshot.trailing_number_delta_indexes)
            and snapshot.trailing_number_delta_indexes[
                trailing_number_delta_offset
            ]
            == trailing_offset
        ):
            number_delta = snapshot.trailing_number_deltas[
                trailing_number_delta_offset
            ]
            trailing_number_delta_offset += 1
        char_delta = 0
        if (
            trailing_char_delta_offset < len(snapshot.trailing_char_delta_indexes)
            and snapshot.trailing_char_delta_indexes[trailing_char_delta_offset]
            == trailing_offset
        ):
            char_delta = snapshot.trailing_char_deltas[trailing_char_delta_offset]
            trailing_char_delta_offset += 1
        match_start = snapshot.compact_missing
        match_end = snapshot.compact_missing
        if (
            trailing_match_offset < len(snapshot.trailing_match_indexes)
            and snapshot.trailing_match_indexes[trailing_match_offset]
            == trailing_offset
        ):
            match_start = snapshot.trailing_match_starts[trailing_match_offset]
            match_end = snapshot.trailing_match_ends[trailing_match_offset]
            trailing_match_offset += 1
        yield RefExecLine(
            text="",
            number=output_number + number_delta,
            byte_start=snapshot.stream_start,
            char_start=snapshot.stream_chars + char_delta,
            has_newline=False,
            match_start=(
                match_start if match_start != snapshot.compact_missing else None
            ),
            match_end=(
                match_end if match_end != snapshot.compact_missing else None
            ),
            byte_range=byte_range,
            component_view=(None if trailing_plain else RefExecComponentView(())),
        )
        output_number += 1


# allow: SIZE_OK - one-pass circular-buffer state keeps overwrite ordering local.
def _ref_exec_tail_bytes(
    lines: Iterable[RefExecLine],
    count: int,
    cancelled: threading.Event,
) -> Iterable[RefExecLine]:
    if count > REF_EXEC_TAIL_MAX_BYTES:
        raise RefExecError(
            "Error: tail rolling window exceeds 8 MiB; reduce -c or filter first"
    )
    if count == 0:
        return
    retained = bytearray(count)
    retained_bytes = 0
    write_offset = 0
    component_lengths = array("I")
    component_flags = bytearray()
    component_source_delta_indexes = array("I")
    component_source_deltas = array("i")
    component_char_delta_indexes = array("I")
    component_char_deltas = array("i")
    component_base = 0
    record_component_ends = array("I")
    record_flags = bytearray()
    record_char_delta_indexes = array("I")
    record_char_deltas = array("i")
    record_number_delta_indexes = array("I")
    record_number_deltas = array("i")
    record_match_starts: array | None = None
    record_match_ends: array | None = None
    record_base = 0
    trailing_plain_default = False
    trailing_plain_last = False
    trailing_plain_change_indexes = array("Q")
    trailing_number_delta_indexes = array("Q")
    trailing_number_deltas = array("i")
    trailing_char_delta_indexes = array("Q")
    trailing_char_deltas = array("i")
    trailing_match_indexes = array("Q")
    trailing_match_starts = array("i")
    trailing_match_ends = array("i")
    compact_missing = -1
    stream_start = 0
    stream_chars = 0
    retained_char_start = 0
    partial_fallback_char_start: int | None = None
    line_count = 0
    trailing_empty_lines = 0
    snapped_empty = False
    snapped_number = 0
    snapped_char_start = 0

    def append_compact(lane: array, value: int) -> array:
        if lane.typecode == "i" and not -(1 << 31) <= value < 1 << 31:
            lane = array("q", lane)
        lane.append(value)
        return lane

    def drop_metadata(dropped_bytes: int, dropped_offset: int) -> None:
        nonlocal component_base, record_base, partial_fallback_char_start
        previous_component_base = component_base
        remaining = dropped_bytes
        clipped_component = False
        while remaining and component_base < len(component_lengths):
            component_length = component_lengths[component_base]
            if remaining >= component_length:
                remaining -= component_length
                component_base += 1
                continue
            component_lengths[component_base] = component_length - remaining
            remaining = 0
            clipped_component = True
        while (
            record_base < len(record_component_ends)
            and record_component_ends[record_base] <= component_base
        ):
            record_base += 1
        if record_base >= len(record_component_ends):
            return
        record_component_start = (
            record_component_ends[record_base - 1] if record_base else 0
        )
        if record_component_start < component_base or clipped_component:
            if record_flags[record_base] & 0x01:
                record_boundary_bytes = sum(
                    component_lengths[index]
                    for index in range(
                        previous_component_base, record_component_start
                    )
                )
                record_boundary_chars = sum(
                    retained[(dropped_offset + offset) % count] & 0xC0 != 0x80
                    for offset in range(
                        min(record_boundary_bytes, dropped_bytes)
                    )
                )
                char_delta_position = bisect_left(
                    record_char_delta_indexes, record_base
                )
                record_char_delta = (
                    record_char_deltas[char_delta_position]
                    if char_delta_position < len(record_char_delta_indexes)
                    and record_char_delta_indexes[char_delta_position] == record_base
                    else 0
                )
                partial_fallback_char_start = (
                    retained_char_start
                    + record_boundary_chars
                    + record_char_delta
                )
            record_flags[record_base] &= 0x02
            if record_match_starts is not None and record_match_ends is not None:
                record_match_starts[record_base] = compact_missing
                record_match_ends[record_base] = compact_missing

    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            continue
        line_count += 1
        original_view = _ref_exec_line_component_view(line, cancelled)
        line_bytes = sum(component.utf8_bytes for component in original_view.components)
        line_end = stream_start + line_bytes
        line_chars = sum(
            component.end - component.start for component in original_view.components
        )
        selected = line
        selected_start = stream_start
        selected_char_start = stream_chars
        selected_is_snapped_empty = False
        if line_bytes > count:
            selected, selected_start, _, _ = _ref_exec_slice_presented_line(
                line,
                stream_start=stream_start,
                selected_start=line_end - count,
                selected_end=line_end,
                cancelled=cancelled,
            )
            if selected is None:
                stream_start = line_end
                stream_chars += line_chars
                trailing_empty_lines += 1
                continue
            retained_bytes = 0
            write_offset = 0
            component_lengths = array("I")
            component_flags = bytearray()
            component_source_delta_indexes = array("I")
            component_source_deltas = array("i")
            component_char_delta_indexes = array("I")
            component_char_deltas = array("i")
            component_base = 0
            record_component_ends = array("I")
            record_flags = bytearray()
            record_char_delta_indexes = array("I")
            record_char_deltas = array("i")
            record_number_delta_indexes = array("I")
            record_number_deltas = array("i")
            record_match_starts = None
            record_match_ends = None
            record_base = 0
            trailing_empty_lines = 0
            trailing_plain_default = False
            trailing_plain_last = False
            trailing_plain_change_indexes = array("Q")
            trailing_number_delta_indexes = array("Q")
            trailing_number_deltas = array("i")
            trailing_char_delta_indexes = array("Q")
            trailing_char_deltas = array("i")
            trailing_match_indexes = array("Q")
            trailing_match_starts = array("i")
            trailing_match_ends = array("i")
            partial_fallback_char_start = selected.char_start
            snapped_empty = selected.metadata_only
            selected_is_snapped_empty = snapped_empty
            if snapped_empty:
                snapped_number = selected.number
                snapped_char_start = selected.char_start
            selected_chars = sum(
                component.end - component.start
                for component in _ref_exec_line_component_view(
                    selected, cancelled
                ).components
            )
            selected_char_start = stream_chars + line_chars - selected_chars
            retained_char_start = selected_char_start
        selected_view = _ref_exec_line_component_view(selected, cancelled)
        if line_bytes and trailing_empty_lines:
            trailing_empty_lines = 0
            trailing_plain_default = False
            trailing_plain_last = False
            trailing_plain_change_indexes = array("Q")
            trailing_number_delta_indexes = array("Q")
            trailing_number_deltas = array("i")
            trailing_char_delta_indexes = array("Q")
            trailing_char_deltas = array("i")
            trailing_match_indexes = array("Q")
            trailing_match_starts = array("i")
            trailing_match_ends = array("i")
        record_offset = len(record_component_ends)
        record_start = selected_start == stream_start
        record_char_delta = selected.char_start - selected_char_start
        component_stream_start = selected_start
        component_char_start = selected_char_start
        for component in selected_view.components:
            kind = {
                "display_prefix": 0,
                "text": 1,
                "synthetic_lf": 2,
            }[component.kind]
            source_byte_delta = (
                component.source_byte_start - component_stream_start
                if component.source_byte_start is not None
                else 0
            )
            source_char_delta = (
                component.source_char_start - component_char_start
                if component.source_char_start is not None
                else 0
            )
            component_length = 0
            for offset in range(
                component.start, component.end, REF_TEXT_HASH_CHUNK_CHARS
            ):
                chunk_end = min(
                    component.end, offset + REF_TEXT_HASH_CHUNK_CHARS
                )
                encoded = _encode_ref_text_checked(
                    component.text[offset:chunk_end], cancelled
                )
                component_length += len(encoded)
                encoded_offset = 0
                while encoded_offset < len(encoded):
                    available = min(
                        len(encoded) - encoded_offset, count - write_offset
                    )
                    write_end = write_offset + available
                    if retained_bytes == count:
                        drop_metadata(available, write_offset)
                        for retained_offset in range(write_offset, write_end):
                            if retained[retained_offset] & 0xC0 != 0x80:
                                retained_char_start += 1
                    retained[write_offset:write_end] = encoded[
                        encoded_offset : encoded_offset + available
                    ]
                    write_offset = write_end % count
                    encoded_offset += available
                    retained_bytes = min(count, retained_bytes + available)
            if component_length:
                component_lengths.append(component_length)
                flags = kind
                if component.source_byte_start is not None:
                    flags |= 0x04
                    if (
                        source_byte_delta == record_char_delta
                        and source_char_delta == record_char_delta
                    ):
                        flags |= 0x08
                    elif source_byte_delta:
                        component_source_delta_indexes.append(
                            len(component_lengths) - 1
                        )
                        component_source_deltas = append_compact(
                            component_source_deltas, source_byte_delta
                        )
                component_flags.append(flags)
                if source_char_delta != source_byte_delta:
                    component_char_delta_indexes.append(len(component_lengths) - 1)
                    component_char_deltas = append_compact(
                        component_char_deltas, source_char_delta
                    )
            component_stream_start += component.utf8_bytes
            component_char_start += component.end - component.start
        if line_bytes and not selected_is_snapped_empty:
            record_component_ends.append(len(component_lengths))
            flags = int(record_start)
            if record_start and selected.has_newline:
                flags |= 0x02
            if record_start and selected.component_view is None:
                flags |= 0x04
            record_flags.append(flags)
            if record_char_delta:
                record_char_delta_indexes.append(record_offset)
                record_char_deltas = append_compact(
                    record_char_deltas, record_char_delta
                )
            number_delta = selected.number - line_count
            if number_delta:
                record_number_delta_indexes.append(record_offset)
                record_number_deltas = append_compact(
                    record_number_deltas, number_delta
                )
            if selected.match_start is not None or selected.match_end is not None:
                if record_match_starts is None or record_match_ends is None:
                    record_match_starts = array("i", (compact_missing,)) * record_offset
                    record_match_ends = array("i", (compact_missing,)) * record_offset
                record_match_starts = append_compact(
                    record_match_starts,
                    selected.match_start
                    if selected.match_start is not None
                    else compact_missing,
                )
                record_match_ends = append_compact(
                    record_match_ends,
                    selected.match_end
                    if selected.match_end is not None
                    else compact_missing,
                )
            elif record_match_starts is not None and record_match_ends is not None:
                record_match_starts.append(compact_missing)
                record_match_ends.append(compact_missing)
        if retained_bytes:
            physical_oldest_offset = write_offset if retained_bytes == count else 0
            oldest_offset = physical_oldest_offset
            for _ in range(min(4, retained_bytes)):
                if retained[oldest_offset] & 0xC0 != 0x80:
                    break
                oldest_offset = (oldest_offset + 1) % count
            skipped_bytes = (oldest_offset - physical_oldest_offset) % count
            logical_component = component_base
            logical_component_skip = skipped_bytes
            while (
                logical_component < len(component_lengths)
                and logical_component_skip
                >= component_lengths[logical_component]
            ):
                logical_component_skip -= component_lengths[logical_component]
                logical_component += 1
            logical_record = bisect_right(
                record_component_ends,
                logical_component,
                lo=record_base,
            )
            logical_record_start = (
                record_component_ends[logical_record - 1]
                if logical_record
                else 0
            )
            logical_record_intact = (
                logical_record < len(record_flags)
                and bool(record_flags[logical_record] & 0x01)
                and logical_component == logical_record_start
                and logical_component_skip == 0
            )
            if logical_record_intact:
                partial_fallback_char_start = None
            elif (
                logical_record < len(record_component_ends)
                and logical_component < len(component_flags)
            ):
                oldest_flags = component_flags[logical_component]
                if oldest_flags & 0x03 == 1 and oldest_flags & 0x04:
                    char_delta_position = bisect_left(
                        record_char_delta_indexes, logical_record
                    )
                    source_char_delta = 0
                    if (
                        oldest_flags & 0x08
                        and char_delta_position < len(record_char_delta_indexes)
                        and record_char_delta_indexes[char_delta_position]
                        == logical_record
                    ):
                        source_char_delta = record_char_deltas[
                            char_delta_position
                        ]
                    source_delta_position = bisect_left(
                        component_source_delta_indexes, logical_component
                    )
                    if (
                        source_delta_position
                        < len(component_source_delta_indexes)
                        and component_source_delta_indexes[source_delta_position]
                        == logical_component
                    ):
                        source_char_delta = component_source_deltas[
                            source_delta_position
                        ]
                    char_delta_position = bisect_left(
                        component_char_delta_indexes, logical_component
                    )
                    if (
                        char_delta_position < len(component_char_delta_indexes)
                        and component_char_delta_indexes[char_delta_position]
                        == logical_component
                    ):
                        source_char_delta = component_char_deltas[
                            char_delta_position
                        ]
                    partial_fallback_char_start = (
                        retained_char_start + source_char_delta
                    )
        if line_bytes == 0:
            trailing_plain = selected.component_view is None
            if trailing_empty_lines == 0:
                trailing_plain_default = trailing_plain
            elif trailing_plain != trailing_plain_last:
                trailing_plain_change_indexes.append(trailing_empty_lines)
            trailing_plain_last = trailing_plain
            trailing_number_delta = selected.number - line_count
            if trailing_number_delta:
                trailing_number_delta_indexes.append(trailing_empty_lines)
                trailing_number_deltas = append_compact(
                    trailing_number_deltas, trailing_number_delta
                )
            trailing_char_delta = selected.char_start - stream_chars
            if trailing_char_delta:
                trailing_char_delta_indexes.append(trailing_empty_lines)
                trailing_char_deltas = append_compact(
                    trailing_char_deltas, trailing_char_delta
                )
            if selected.match_start is not None or selected.match_end is not None:
                trailing_match_indexes.append(trailing_empty_lines)
                trailing_match_starts = append_compact(
                    trailing_match_starts,
                    selected.match_start
                    if selected.match_start is not None
                    else compact_missing,
                )
                trailing_match_ends = append_compact(
                    trailing_match_ends,
                    selected.match_end
                    if selected.match_end is not None
                    else compact_missing,
                )
            trailing_empty_lines += 1
        stream_start = line_end
        stream_chars += line_chars
        if record_base >= 65_536:
            del record_component_ends[:record_base]
            del record_flags[:record_base]
            char_record_prefix = bisect_left(
                record_char_delta_indexes, record_base
            )
            del record_char_delta_indexes[:char_record_prefix]
            del record_char_deltas[:char_record_prefix]
            for index in range(len(record_char_delta_indexes)):
                record_char_delta_indexes[index] -= record_base
            if record_match_starts is not None and record_match_ends is not None:
                del record_match_starts[:record_base]
                del record_match_ends[:record_base]
            number_prefix = 0
            while (
                number_prefix < len(record_number_delta_indexes)
                and record_number_delta_indexes[number_prefix] < record_base
            ):
                number_prefix += 1
            del record_number_delta_indexes[:number_prefix]
            del record_number_deltas[:number_prefix]
            for index in range(len(record_number_delta_indexes)):
                record_number_delta_indexes[index] -= record_base
            record_base = 0
        if component_base >= 131_072:
            if record_base:
                del record_component_ends[:record_base]
                del record_flags[:record_base]
                char_record_prefix = bisect_left(
                    record_char_delta_indexes, record_base
                )
                del record_char_delta_indexes[:char_record_prefix]
                del record_char_deltas[:char_record_prefix]
                for index in range(len(record_char_delta_indexes)):
                    record_char_delta_indexes[index] -= record_base
                if (
                    record_match_starts is not None
                    and record_match_ends is not None
                ):
                    del record_match_starts[:record_base]
                    del record_match_ends[:record_base]
                number_prefix = bisect_left(
                    record_number_delta_indexes, record_base
                )
                del record_number_delta_indexes[:number_prefix]
                del record_number_deltas[:number_prefix]
                for index in range(len(record_number_delta_indexes)):
                    record_number_delta_indexes[index] -= record_base
                record_base = 0
            del component_lengths[:component_base]
            del component_flags[:component_base]
            source_prefix = 0
            while (
                source_prefix < len(component_source_delta_indexes)
                and component_source_delta_indexes[source_prefix] < component_base
            ):
                source_prefix += 1
            del component_source_delta_indexes[:source_prefix]
            del component_source_deltas[:source_prefix]
            for index in range(len(component_source_delta_indexes)):
                component_source_delta_indexes[index] -= component_base
            char_prefix = 0
            while (
                char_prefix < len(component_char_delta_indexes)
                and component_char_delta_indexes[char_prefix] < component_base
            ):
                char_prefix += 1
            del component_char_delta_indexes[:char_prefix]
            del component_char_deltas[:char_prefix]
            for index in range(len(component_char_delta_indexes)):
                component_char_delta_indexes[index] -= component_base
            for index in range(len(record_component_ends)):
                record_component_ends[index] -= component_base
            component_base = 0
    if retained_bytes == 0 and trailing_empty_lines == 0 and not snapped_empty:
        return
    snapped_bytes = 0
    oldest_offset = write_offset if retained_bytes == count else 0
    while (
        snapped_bytes < retained_bytes
        and retained[(oldest_offset + snapped_bytes) % count] & 0xC0 == 0x80
    ):
        snapped_bytes += 1
    if snapped_bytes:
        drop_metadata(snapped_bytes, oldest_offset)
    if record_base:
        del record_component_ends[:record_base]
        del record_flags[:record_base]
        char_record_prefix = bisect_left(record_char_delta_indexes, record_base)
        del record_char_delta_indexes[:char_record_prefix]
        del record_char_deltas[:char_record_prefix]
        for index in range(len(record_char_delta_indexes)):
            record_char_delta_indexes[index] -= record_base
        if record_match_starts is not None and record_match_ends is not None:
            del record_match_starts[:record_base]
            del record_match_ends[:record_base]
        number_prefix = 0
        while (
            number_prefix < len(record_number_delta_indexes)
            and record_number_delta_indexes[number_prefix] < record_base
        ):
            number_prefix += 1
        del record_number_delta_indexes[:number_prefix]
        del record_number_deltas[:number_prefix]
        for index in range(len(record_number_delta_indexes)):
            record_number_delta_indexes[index] -= record_base
        record_base = 0
    if component_base:
        del component_lengths[:component_base]
        del component_flags[:component_base]
        source_prefix = 0
        while (
            source_prefix < len(component_source_delta_indexes)
            and component_source_delta_indexes[source_prefix] < component_base
        ):
            source_prefix += 1
        del component_source_delta_indexes[:source_prefix]
        del component_source_deltas[:source_prefix]
        for index in range(len(component_source_delta_indexes)):
            component_source_delta_indexes[index] -= component_base
        char_prefix = 0
        while (
            char_prefix < len(component_char_delta_indexes)
            and component_char_delta_indexes[char_prefix] < component_base
        ):
            char_prefix += 1
        del component_char_delta_indexes[:char_prefix]
        del component_char_deltas[:char_prefix]
        for index in range(len(component_char_delta_indexes)):
            component_char_delta_indexes[index] -= component_base
        for index in range(len(record_component_ends)):
            record_component_ends[index] -= component_base
    yield from _iter_ref_exec_tail_snapshot(
        RefExecTailSnapshot(
            retained=retained,
            write_offset=write_offset,
            component_lengths=component_lengths,
            component_flags=component_flags,
            component_source_delta_indexes=component_source_delta_indexes,
            component_source_deltas=component_source_deltas,
            component_char_delta_indexes=component_char_delta_indexes,
            component_char_deltas=component_char_deltas,
            record_component_ends=record_component_ends,
            record_flags=record_flags,
            record_char_delta_indexes=record_char_delta_indexes,
            record_char_deltas=record_char_deltas,
            record_number_delta_indexes=record_number_delta_indexes,
            record_number_deltas=record_number_deltas,
            record_match_starts=record_match_starts,
            record_match_ends=record_match_ends,
            trailing_plain_default=trailing_plain_default,
            trailing_plain_change_indexes=trailing_plain_change_indexes,
            trailing_number_delta_indexes=trailing_number_delta_indexes,
            trailing_number_deltas=trailing_number_deltas,
            trailing_char_delta_indexes=trailing_char_delta_indexes,
            trailing_char_deltas=trailing_char_deltas,
            trailing_match_indexes=trailing_match_indexes,
            trailing_match_starts=trailing_match_starts,
            trailing_match_ends=trailing_match_ends,
            compact_missing=compact_missing,
            retained_bytes=retained_bytes,
            snapped_bytes=snapped_bytes,
            stream_start=stream_start,
            stream_chars=stream_chars,
            line_count=line_count,
            trailing_empty_lines=trailing_empty_lines,
            snapped_empty=snapped_empty,
            snapped_number=snapped_number,
            snapped_char_start=snapped_char_start,
            partial_fallback_char_start=partial_fallback_char_start,
            count=count,
        )
    )


def _ref_exec_sed(lines: Iterable[RefExecLine], start: int, end: int | None, cancelled: threading.Event) -> Iterable[RefExecLine]:
    input_number = 0
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            yield line
            continue
        input_number += 1
        if input_number < start:
            continue
        if end is not None and input_number > end:
            return
        yield line


def _ref_exec_regex_requested(pattern: str, flags: frozenset[str]) -> bool:
    return "E" in flags or any(marker in pattern for marker in ("|", ".*", ".+", ".?", r"\d", r"\w", r"\s")) or bool(re.search(r"\[.+\]", pattern))


def _ref_exec_literal_component_spans(
    line: RefExecLine,
    stage: RefExecStage,
    cancelled: threading.Event,
) -> Iterable[tuple[int, int, RefExecComponent | None, int, int, str | None]]:
    pattern = stage.pattern or ""
    if not pattern:
        component = next(
            (
                candidate
                for candidate in _ref_exec_line_component_view(
                    line, cancelled
                ).components
                if candidate.kind != "synthetic_lf"
            ),
            None,
        )
        if component is None and line.component_view is None:
            component = _ref_exec_component(
                "text",
                line.text,
                cancelled,
                source_start=(line.byte_start, line.char_start),
            )
        position = component.start if component is not None else 0
        yield (0, 0, component, position, position, None)
        return
    components = (
        component
        for component in _ref_exec_line_component_view(line, cancelled).components
        if component.kind != "synthetic_lf"
    )
    compiled_literal = (
        re.compile(re.escape(pattern), re.IGNORECASE)
        if "i" in stage.flags
        else None
    )
    logical_start = 0
    overlap = len(pattern) - 1
    suffix = ""
    next_match_start = 0
    for component in components:
        _check_ref_exec_cancelled(cancelled)
        component_length = component.end - component.start
        if suffix and overlap > 0:
            current_end = min(component.end, component.start + overlap)
            boundary_text = suffix + component.text[component.start:current_end]
            boundary_offset = logical_start - len(suffix)
            boundary = len(suffix)
            search_start = max(0, next_match_start - boundary_offset)
            if compiled_literal is None:
                position = search_start
                while position <= len(boundary_text) - len(pattern):
                    found = boundary_text.find(pattern, position, len(boundary_text))
                    if found < 0:
                        break
                    end = found + len(pattern)
                    if found < boundary < end:
                        global_start = boundary_offset + found
                        global_end = boundary_offset + end
                        yield (
                            global_start,
                            global_end,
                            None,
                            0,
                            0,
                            boundary_text[found:end],
                        )
                        next_match_start = global_end
                        position = max(end, next_match_start - boundary_offset)
                    else:
                        position = found + 1
            else:
                for match in compiled_literal.finditer(
                    boundary_text, search_start, len(boundary_text)
                ):
                    start, end = match.span()
                    if start < boundary < end:
                        global_start = boundary_offset + start
                        global_end = boundary_offset + end
                        yield (
                            global_start,
                            global_end,
                            None,
                            0,
                            0,
                            boundary_text[start:end],
                        )
                        next_match_start = global_end
        if compiled_literal is None:
            position = max(
                component.start,
                component.start + next_match_start - logical_start,
            )
            while position <= component.end - len(pattern):
                found = component.text.find(pattern, position, component.end)
                if found < 0:
                    break
                global_start = logical_start + found - component.start
                global_end = global_start + len(pattern)
                yield (
                    global_start,
                    global_end,
                    component,
                    found,
                    found + len(pattern),
                    None,
                )
                next_match_start = global_end
                position = found + len(pattern)
        else:
            position = max(
                component.start,
                component.start + next_match_start - logical_start,
            )
            for match in compiled_literal.finditer(
                component.text, position, component.end
            ):
                start, end = match.span()
                global_start = logical_start + start - component.start
                global_end = logical_start + end - component.start
                yield (
                    global_start,
                    global_end,
                    component,
                    start,
                    end,
                    None,
                )
                next_match_start = global_end
        if overlap > 0:
            suffix = (
                component.text[component.end - overlap : component.end]
                if component_length >= overlap
                else (
                    suffix + component.text[component.start : component.end]
                )[-overlap:]
            )
        logical_start += component_length


def _ref_exec_source_match(
    line: RefExecLine,
    component: RefExecComponent | None,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    if (
        component is None
        or component.kind != "text"
        or component.source_char_start is None
    ):
        return None
    source_start = component.source_char_start + start - component.start
    return source_start - line.char_start, source_start - line.char_start + end - start


def _ref_exec_logical_source_match(
    line: RefExecLine,
    start: int,
    end: int,
    cancelled: threading.Event,
) -> tuple[RefExecComponent, int, int, tuple[int, int]] | None:
    logical_start = 0
    for component in _ref_exec_line_component_view(line, cancelled).components:
        if component.kind == "synthetic_lf":
            continue
        logical_end = logical_start + component.end - component.start
        if start >= logical_start and end <= logical_end:
            local_start = component.start + start - logical_start
            local_end = component.start + end - logical_start
            source_match = _ref_exec_source_match(
                line, component, local_start, local_end
            )
            if source_match is None:
                return None
            return component, local_start, local_end, source_match
        logical_start = logical_end
    return None


def _ref_exec_line_source_end(
    line: RefExecLine,
    cancelled: threading.Event,
) -> int:
    source_bounds = _ref_exec_source_byte_bounds(line)
    if source_bounds is not None:
        return source_bounds[1]
    line_bytes, _ = _measure_ref_text_checked(line.text, cancelled)
    return line.byte_start + line_bytes + int(line.has_newline)


def _ref_exec_grep(lines: Iterable[RefExecLine], stage: RefExecStage, cancelled: threading.Event) -> Iterable[RefExecLine]:
    pattern = stage.pattern or ""
    use_regex = _ref_exec_regex_requested(pattern, stage.flags)
    only_matching = "o" in stage.flags and "c" not in stage.flags
    compiled = None
    if use_regex:
        if _REGEX is None:
            raise RefExecError("Error: regex support is unavailable")
        normalized = pattern.replace(r"\|", "|")
        try:
            compiled = _REGEX.compile(normalized, _REGEX.IGNORECASE if "i" in stage.flags else 0)
        except (getattr(_REGEX, "error", RuntimeError), RuntimeError) as exc:
            raise RefExecError(f"Error: invalid regex: {exc}") from exc

    def selected() -> Iterable[RefExecLine]:
        count = 0
        budget_start: float | None = None
        last_line = 0
        last_byte = 0
        input_number = 0
        for line in lines:
            _check_ref_exec_cancelled(cancelled)
            if line.metadata_only:
                continue
            input_number += 1
            selected_line = False
            source_match: tuple[int, int] | None = None
            presented_text = None
            if only_matching:
                if compiled is not None:
                    presented_text = _ref_exec_materialize_line(
                        line, cancelled, include_synthetic_lf=False
                    )
                    if budget_start is None:
                        budget_start = time.monotonic()
                    remaining = REF_EXEC_REGEX_BUDGET_SECONDS - (
                        time.monotonic() - budget_start
                    )
                    if remaining <= 0:
                        raise RefExecError(
                            f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                        )
                    regex_matches = iter(
                        compiled.finditer(
                            presented_text,
                            timeout=min(REF_EXEC_REGEX_BUDGET_SECONDS, remaining),
                        )
                    )
                    matches = (
                        (match.span(), match.group(0), None, 0, 0)
                        for match in regex_matches
                    )
                else:
                    matches = (
                        (
                            (start, end),
                            crossing_fragment or pattern,
                            component,
                            local_start,
                            local_end,
                        )
                        for start, end, component, local_start, local_end, crossing_fragment in _ref_exec_literal_component_spans(
                            line,
                            stage,
                            cancelled=cancelled,
                        )
                    )
                measured_text: str | None = None
                measured_origin = 0
                measured_end = 0
                measured_source_byte_start = 0
                measured_bytes = 0
                while True:
                    _check_ref_exec_cancelled(cancelled)
                    if compiled is not None:
                        remaining = REF_EXEC_REGEX_BUDGET_SECONDS - (
                            time.monotonic() - budget_start
                        )
                        if remaining <= 0:
                            raise RefExecError(
                                f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                            )
                    try:
                        match_data = next(matches)
                    except StopIteration:
                        break
                    except TimeoutError as exc:
                        raise RefExecError(
                            f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                        ) from exc
                    (match_start, match_end), fragment, component, local_start, local_end = match_data
                    if match_end == match_start:
                        continue
                    source_match = _ref_exec_source_match(
                        line, component, local_start, local_end
                    )
                    if compiled is not None:
                        logical_source_match = _ref_exec_logical_source_match(
                            line,
                            match_start,
                            match_end,
                            cancelled,
                        )
                        if logical_source_match is not None:
                            component, local_start, local_end, source_match = (
                                logical_source_match
                            )
                    source_backed = source_match is not None
                    source_match_start = source_match[0] if source_match is not None else 0
                    source_prefix_bytes = 0
                    if source_backed:
                        measurement_text = (
                            component.text if component is not None else line.text
                        )
                        measurement_origin = (
                            component.start if component is not None else 0
                        )
                        measurement_start = (
                            local_start
                            if component is not None
                            else source_match_start
                        )
                        measurement_end = (
                            local_end
                            if component is not None
                            else source_match_start + match_end - match_start
                        )
                        measurement_source_byte_start = (
                            component.source_byte_start
                            if component is not None
                            and component.source_byte_start is not None
                            else line.byte_start
                        )
                        if (
                            measured_text is measurement_text
                            and measured_origin == measurement_origin
                            and measured_source_byte_start
                            == measurement_source_byte_start
                            and measured_end <= measurement_start
                        ):
                            source_prefix_bytes = measured_bytes + (
                                _ref_exec_utf8_range_bytes(
                                    measurement_text,
                                    measured_end,
                                    measurement_start,
                                    cancelled,
                                )
                            )
                        else:
                            source_prefix_bytes = _ref_exec_utf8_range_bytes(
                                measurement_text,
                                measurement_origin,
                                measurement_start,
                                cancelled,
                            )
                        measured_bytes = source_prefix_bytes + (
                            _ref_exec_utf8_range_bytes(
                                measurement_text,
                                measurement_start,
                                measurement_end,
                                cancelled,
                            )
                        )
                        measured_text = measurement_text
                        measured_origin = measurement_origin
                        measured_end = measurement_end
                        measured_source_byte_start = measurement_source_byte_start
                    else:
                        measured_text = None
                    yield replace(
                        line,
                        text=(
                            fragment
                            if compiled is not None
                            else (
                                component.text[local_start:local_end]
                                if component is not None
                                else fragment
                            )
                        ),
                        byte_start=(
                            component.source_byte_start + source_prefix_bytes
                            if source_backed
                            and component is not None
                            and component.source_byte_start is not None
                            else line.byte_start + source_prefix_bytes
                        ),
                        char_start=(
                            line.char_start + source_match_start
                            if source_backed
                            else line.char_start
                        ),
                        match_start=None,
                        match_end=None,
                        display_prefix=(
                            f"{input_number}:" if "n" in stage.flags else ""
                        ),
                        has_newline=True,
                        byte_range=None,
                        metadata_only=False,
                        atomic_match=True,
                        component_view=None,
                    )
                last_line = line.number
                last_byte = _ref_exec_line_source_end(line, cancelled)
                continue
            if compiled is not None:
                presented_text = _ref_exec_materialize_line(
                    line, cancelled, include_synthetic_lf=False
                )
                if budget_start is None:
                    budget_start = time.monotonic()
                remaining = REF_EXEC_REGEX_BUDGET_SECONDS - (time.monotonic() - budget_start)
                if remaining <= 0:
                    raise RefExecError(
                        f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                    )
                try:
                    # regex requires one contiguous string for cross-boundary matches.
                    # Keep that unavoidable temporary to one current logical line and
                    # only when a prior stage contributed presentation text.
                    match = compiled.search(
                        presented_text,
                        timeout=min(REF_EXEC_REGEX_BUDGET_SECONDS, remaining),
                    )
                except TimeoutError as exc:
                    raise RefExecError(
                        f"Error: regex timeout; last_completed_byte={last_byte} last_completed_line={last_line}; narrow with sed -n or literal grep"
                    ) from exc
                if match is not None:
                    selected_line = True
                    match_start, match_end = match.span()
                    logical_source_match = _ref_exec_logical_source_match(
                        line,
                        match_start,
                        match_end,
                        cancelled,
                    )
                    if logical_source_match is not None:
                        _, _, _, source_match = logical_source_match
            else:
                literal_match = next(
                    iter(
                        _ref_exec_literal_component_spans(
                            line,
                            stage,
                            cancelled=cancelled,
                        )
                    ),
                    None,
                )
                selected_line = literal_match is not None
                if literal_match is not None:
                    _, _, component, local_start, local_end, _ = literal_match
                    source_match = _ref_exec_source_match(
                        line, component, local_start, local_end
                    )
            last_line = line.number
            last_byte = _ref_exec_line_source_end(line, cancelled)
            if not selected_line:
                continue
            count += 1
            if "c" not in stage.flags:
                display_prefix = (
                    f"{input_number}:" if "n" in stage.flags else ""
                )
                component_view = line.component_view
                if component_view is not None and display_prefix:
                    component_view = RefExecComponentView(
                        (
                            _ref_exec_component(
                                "display_prefix", display_prefix, cancelled
                            ),
                            *component_view.components,
                        )
                    )
                    display_prefix = ""
                elif component_view is None:
                    display_prefix += line.display_prefix
                yield replace(
                    line,
                    match_start=source_match[0] if source_match is not None else None,
                    match_end=source_match[1] if source_match is not None else None,
                    display_prefix=display_prefix,
                    component_view=component_view,
                )
        if "c" in stage.flags:
            yield RefExecLine(str(count), 1, 0, 0, False)

    return selected()


def _count_ref_words_checked(
    text: str,
    cancelled: threading.Event,
    *,
    prefix: str = "",
) -> int:
    count = 0
    previous_ended_in_word = False
    for part in (prefix, text):
        for offset in range(0, len(part), REF_TEXT_HASH_CHUNK_CHARS):
            _check_ref_exec_cancelled(cancelled)
            chunk = part[offset : offset + REF_TEXT_HASH_CHUNK_CHARS]
            chunk_count = sum(1 for _ in re.finditer(r"\S+", chunk))
            if previous_ended_in_word and chunk and not chunk[0].isspace():
                chunk_count -= 1
            count += chunk_count
            previous_ended_in_word = bool(chunk) and not chunk[-1].isspace()
    return count


def _ref_exec_wc(lines: Iterable[RefExecLine], flag: str, cancelled: threading.Event) -> Iterable[RefExecLine]:
    line_count = 0
    word_count = 0
    byte_count = 0
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.metadata_only:
            continue
        line_count += 1
        if line.component_view is None:
            word_count += _count_ref_words_checked(
                line.text, cancelled, prefix=line.display_prefix
            )
            encoded_bytes, _ = _measure_ref_text_parts_checked(
                (line.display_prefix, line.text), cancelled
            )
            byte_count += encoded_bytes + int(line.has_newline)
            continue
        previous_ended_in_word = False
        for component in line.component_view.components:
            byte_count += component.utf8_bytes
            for offset in range(
                component.start, component.end, REF_TEXT_HASH_CHUNK_CHARS
            ):
                _check_ref_exec_cancelled(cancelled)
                chunk_end = min(
                    component.end, offset + REF_TEXT_HASH_CHUNK_CHARS
                )
                chunk = component.text[offset:chunk_end]
                chunk_count = sum(1 for _ in re.finditer(r"\S+", chunk))
                if previous_ended_in_word and chunk and not chunk[0].isspace():
                    chunk_count -= 1
                word_count += chunk_count
                previous_ended_in_word = bool(chunk) and not chunk[-1].isspace()
    value = {"l": line_count, "w": word_count, "c": byte_count}[flag]
    yield RefExecLine(str(value), 1, 0, 0, False)


def _ref_exec_stat(entry: RefCatalogEntry, cancelled: threading.Event) -> Iterable[RefExecLine]:
    line_count = 0
    word_count = 0
    character_count = 0
    for line in _iter_ref_source_lines(entry.source, cancelled):
        line_count += 1
        word_count += _count_ref_words_checked(line.text, cancelled)
        character_count += len(line.text) + int(line.has_newline)
    manifest = entry.manifest
    text = (
        f"ref={manifest.ref} kind={manifest.ref.split(':', 1)[0]} utf8_bytes={manifest.utf8_bytes} "
        f"lines={line_count} words={word_count} chars={character_count} sha256={manifest.sha256}"
    )
    yield RefExecLine(text, 1, 0, 0, False)


def _ref_exec_response_fits(
    text: str,
    *,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> bool:
    encoded_bytes, _ = _measure_ref_text_checked(text, cancelled)
    if encoded_bytes > REF_EXEC_RESPONSE_MAX_BYTES:
        return False
    if encoder is None:
        return encoded_bytes < threshold_tokens
    token_count = _ref_exec_encode_text_token_count(encoder, text)
    if token_count is None:
        return encoded_bytes < threshold_tokens
    return token_count < threshold_tokens


def _ref_exec_truncated_marker(next_command: str) -> str:
    payload = json.dumps(
        {"next": next_command},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        "\n<agent_ref_truncated>"
        f"{payload}"
        "</agent_ref_truncated>"
    )


def _ref_exec_truncated_prefix(
    text: str,
    *,
    marker: str,
    continuation_ref: str | None = None,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    marker_bytes, _ = _measure_ref_text_checked(marker, cancelled)
    bounded_text = _ref_exec_utf8_prefix(
        text,
        max(0, REF_EXEC_RESPONSE_MAX_BYTES - marker_bytes),
        cancelled,
    )
    low = 0
    high = len(bounded_text)
    while low < high:
        midpoint = (low + high + 1) // 2
        prefix = bounded_text[:midpoint]
        prefix_bytes, _ = _measure_ref_text_checked(prefix, cancelled)
        candidate_marker = (
            marker
            if continuation_ref is None
            else _ref_exec_truncated_marker(
                f"tail -c +{prefix_bytes + 1} {continuation_ref}"
            )
        )
        candidate = prefix + candidate_marker
        if _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        ):
            low = midpoint
        else:
            high = midpoint - 1
    prefix = bounded_text[:low]
    prefix_bytes, _ = _measure_ref_text_checked(prefix, cancelled)
    return prefix + (
        marker
        if continuation_ref is None
        else _ref_exec_truncated_marker(
            f"tail -c +{prefix_bytes + 1} {continuation_ref}"
        )
    )


def _ref_exec_component_view_prefix(
    view: RefExecComponentView,
    max_bytes: int,
    cancelled: threading.Event,
) -> str:
    parts: list[str] = []
    remaining = max_bytes
    for component in view.components:
        if remaining <= 0:
            break
        if component.utf8_bytes <= remaining:
            parts.append(component.text[component.start : component.end])
            remaining -= component.utf8_bytes
            continue
        end, retained = _ref_exec_component_prefix_index(
            component, remaining, cancelled
        )
        parts.append(component.text[component.start:end])
        remaining -= retained
        break
    return "".join(parts)


def _ref_exec_grep_excerpt(
    line: RefExecLine,
    *,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    match_start = line.match_start or 0
    match_end = line.match_end if line.match_end is not None else match_start
    if line.component_view is None:
        source_text = line.text
        source_start = 0
        source_end = len(source_text)
        display_prefix = line.display_prefix
    else:
        text_component = next(
            (
                component
                for component in line.component_view.components
                if component.kind == "text"
            ),
            None,
        )
        if text_component is None:
            raise RefExecError("Error: grep excerpt has no source-backed text")
        source_text = text_component.text
        source_start = text_component.start
        source_end = text_component.end
        match_start += source_start
        match_end += source_start
        display_prefix = _ref_exec_materialize_components(
            (
                component
                for component in line.component_view.components
                if component.kind == "display_prefix"
            )
        )
    line_bytes = _ref_exec_utf8_range_bytes(
        source_text, source_start, source_end, cancelled
    )
    prefix_bytes = _ref_exec_utf8_range_bytes(
        source_text, source_start, match_start, cancelled
    )
    match_bytes = _ref_exec_utf8_range_bytes(
        source_text, match_start, match_end, cancelled
    )
    display_prefix_bytes, _ = _measure_ref_text_checked(display_prefix, cancelled)
    absolute_byte_start = line.byte_start + prefix_bytes
    absolute_char_start = line.char_start + match_start - source_start
    left = max(source_start, match_start - 8_192)
    right = min(source_end, max(match_end, match_start + 1) + 8_192)
    while True:
        omitted_prefix = _ref_exec_utf8_range_bytes(
            source_text, source_start, left, cancelled
        )
        visible_end_bytes = _ref_exec_utf8_range_bytes(
            source_text, source_start, right, cancelled
        )
        omitted_suffix = line_bytes - visible_end_bytes
        marker_payload = json.dumps(
            {
                "line": line.number,
                "match_byte_start": absolute_byte_start,
                "match_byte_end": absolute_byte_start + match_bytes,
                "match_char_start": absolute_char_start,
                "match_char_end": line.char_start + match_end - source_start,
                "omitted_prefix_bytes": omitted_prefix,
                "omitted_suffix_bytes": omitted_suffix,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        marker = f"\n<agent_ref_excerpt>{marker_payload}</agent_ref_excerpt>"
        visible_bytes = visible_end_bytes - omitted_prefix
        marker_bytes, _ = _measure_ref_text_checked(marker, cancelled)
        if display_prefix_bytes + visible_bytes + marker_bytes > REF_EXEC_RESPONSE_MAX_BYTES:
            if left >= match_start and right <= match_end:
                raise RefExecError("Error: response budget cannot contain the complete grep match")
            left = min(match_start, left + max(1, (match_start - left) // 2))
            right = max(match_end, right - max(1, (right - match_end) // 2))
            continue
        candidate = display_prefix + source_text[left:right] + marker
        if _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        ):
            return candidate
        if left >= match_start and right <= match_end:
            raise RefExecError("Error: response budget cannot contain the complete grep match")
        left = min(match_start, left + max(1, (match_start - left) // 2))
        right = max(match_end, right - max(1, (right - match_end) // 2))


def _ref_exec_format_byte_range(start: int, end: int | None) -> str:
    return f"{start}-{'*' if end is None else end}"


def _ref_exec_byte_range_marker(
    byte_range: RefExecByteRange,
    *,
    continuation: bool = False,
    next_command: str | None = None,
) -> str:
    metadata = {
        "actual": (
            None
            if byte_range.actual_empty
            or byte_range.actual_end is not None
            and byte_range.actual_end < byte_range.actual_start
            else _ref_exec_format_byte_range(
                byte_range.actual_start, byte_range.actual_end
            )
        ),
        "requested": _ref_exec_format_byte_range(
            byte_range.requested_start, byte_range.requested_end
        ),
    }
    if continuation:
        metadata["next"] = next_command or "wc|grep|head|tail|sed"
    payload = json.dumps(
        metadata,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"\n<agent_ref_range>{payload}</agent_ref_range>"


def _ref_exec_truncated_byte_response(
    text: str,
    *,
    byte_range: RefExecByteRange,
    continuation_ref: str | None = None,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    bounded_text = _ref_exec_utf8_prefix(text, REF_EXEC_RESPONSE_MAX_BYTES, cancelled)
    low = 0
    high = len(bounded_text)
    result = ""
    while low <= high:
        midpoint = (low + high) // 2
        prefix = bounded_text[:midpoint]
        prefix_bytes, _ = _measure_ref_text_checked(prefix, cancelled)
        actual_end = byte_range.actual_start + prefix_bytes - 1
        marker_range = replace(
            byte_range,
            actual_end=actual_end,
            marked=True,
            actual_empty=prefix_bytes == 0,
        )
        next_command = None
        if continuation_ref is not None:
            next_command = f"tail -c +{actual_end + 1} {continuation_ref}"
            if byte_range.requested_end is not None:
                remaining = max(0, byte_range.requested_end - actual_end)
                next_command += f" | head -c {remaining}"
        candidate = prefix + _ref_exec_byte_range_marker(
            marker_range,
            continuation=True,
            next_command=next_command,
        )
        if _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        ):
            result = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return result


def _collect_ref_exec_response(
    lines: Iterable[RefExecLine],
    *,
    continuation_ref: str | None,
    final_grep: bool,
    preserve_source_newlines: bool,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    response = ""
    response_bytes = 0
    emitted_output = False
    selected_byte_range: RefExecByteRange | None = None
    atomic_boundaries = [0]
    for line in lines:
        _check_ref_exec_cancelled(cancelled)
        if line.byte_range is not None:
            current_range = line.byte_range
            if selected_byte_range is None or (
                selected_byte_range.requested_start,
                selected_byte_range.requested_end,
            ) != (current_range.requested_start, current_range.requested_end):
                selected_byte_range = current_range
            else:
                existing_has_actual = not selected_byte_range.actual_empty
                current_has_actual = not current_range.actual_empty
                selected_byte_range = replace(
                    current_range,
                    actual_start=(
                        selected_byte_range.actual_start
                        if existing_has_actual
                        else current_range.actual_start
                    ),
                    actual_end=(
                        current_range.actual_end
                        if current_has_actual
                        else selected_byte_range.actual_end
                    ),
                    marked=selected_byte_range.marked or current_range.marked,
                    actual_empty=not (existing_has_actual or current_has_actual),
                )
            if line.metadata_only:
                continue
        separator = "\n" if emitted_output and not preserve_source_newlines else ""
        if line.component_view is None:
            line_ending = "\n" if preserve_source_newlines and line.has_newline else ""
            line_bytes, _ = _measure_ref_text_parts_checked(
                (separator, line.display_prefix, line.text, line_ending), cancelled
            )
        else:
            line_ending = ""
            line_bytes = len(separator.encode("utf-8")) + sum(
                component.utf8_bytes
                for component in line.component_view.components
                if preserve_source_newlines or component.kind != "synthetic_lf"
            )
        complete_bytes = response_bytes + line_bytes
        candidate: str | None = None
        if complete_bytes <= REF_EXEC_RESPONSE_MAX_BYTES:
            if line.component_view is None:
                presented_line = line.display_prefix + line.text
            else:
                presented_line = _ref_exec_materialize_line(
                    line,
                    cancelled,
                    include_synthetic_lf=preserve_source_newlines,
                )
            candidate = response + separator + presented_line + line_ending
            if selected_byte_range is not None and not final_grep:
                response = candidate
                response_bytes = complete_bytes
                emitted_output = True
                if line.atomic_match:
                    atomic_boundaries.append(len(response))
                continue
            if _ref_exec_response_fits(
                candidate,
                threshold_tokens=threshold_tokens,
                encoder=encoder,
                cancelled=cancelled,
            ):
                response = candidate
                response_bytes = complete_bytes
                emitted_output = True
                if line.atomic_match:
                    atomic_boundaries.append(len(response))
                continue
        if line.atomic_match:
            if not emitted_output:
                return "Error: complete grep match exceeds the response budget; page source bytes with tail -c +N REF"
            marker = _ref_exec_truncated_marker("wc|grep|head|tail|sed")
            for boundary in reversed(atomic_boundaries):
                atomic_response = response[:boundary] + marker
                if _ref_exec_response_fits(
                    atomic_response,
                    threshold_tokens=threshold_tokens,
                    encoder=encoder,
                    cancelled=cancelled,
                ):
                    return atomic_response
            return marker
        if final_grep and line.match_start is not None and not emitted_output:
            return _ref_exec_grep_excerpt(
                line,
                threshold_tokens=threshold_tokens,
                encoder=encoder,
                cancelled=cancelled,
            )
        marker = _ref_exec_truncated_marker("wc|grep|head|tail|sed")
        if candidate is None:
            remaining_bytes = max(
                0,
                REF_EXEC_RESPONSE_MAX_BYTES - response_bytes - len(separator),
            )
            if line.component_view is None:
                display_prefix = _ref_exec_utf8_prefix(
                    line.display_prefix, remaining_bytes, cancelled
                )
                display_prefix_bytes, _ = _measure_ref_text_checked(
                    display_prefix, cancelled
                )
                source_prefix = (
                    _ref_exec_utf8_prefix(
                        line.text,
                        remaining_bytes - display_prefix_bytes,
                        cancelled,
                    )
                    if display_prefix == line.display_prefix
                    else ""
                )
                line_prefix = display_prefix + source_prefix
            else:
                line_prefix = _ref_exec_component_view_prefix(
                    line.component_view, remaining_bytes, cancelled
                )
            candidate = response + separator + line_prefix
        if selected_byte_range is not None:
            return _ref_exec_truncated_byte_response(
                candidate,
                byte_range=selected_byte_range,
                continuation_ref=continuation_ref,
                threshold_tokens=threshold_tokens,
                encoder=encoder,
                cancelled=cancelled,
            )
        return _ref_exec_truncated_prefix(
            candidate,
            marker=marker,
            continuation_ref=continuation_ref,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )
    if selected_byte_range is not None and selected_byte_range.marked:
        candidate = response + _ref_exec_byte_range_marker(selected_byte_range)
        if _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        ):
            return candidate
        return _ref_exec_truncated_byte_response(
            response,
            byte_range=selected_byte_range,
            continuation_ref=continuation_ref,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )
    if selected_byte_range is not None and not _ref_exec_response_fits(
        response,
        threshold_tokens=threshold_tokens,
        encoder=encoder,
        cancelled=cancelled,
    ):
        return _ref_exec_truncated_byte_response(
            response,
            byte_range=selected_byte_range,
            continuation_ref=continuation_ref,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )
    return response


def _execute_ref_reader_sync(
    stages: tuple[RefExecStage, ...],
    catalog: tuple[RefCatalogEntry, ...],
    *,
    threshold_tokens: int,
    encoder: Any,
    cancelled: threading.Event,
) -> str:
    first = stages[0]
    verified_source: Iterable[RefExecLine] | None = None
    if first.command == "ls":
        lines: Iterable[RefExecLine] = (
            RefExecLine(entry.manifest.ref, index, 0, 0, False)
            for index, entry in enumerate(catalog, 1)
            if first.list_kind is None
            or entry.manifest.ref.startswith(f"{first.list_kind}:")
        )
    else:
        requested_hash = parse_ref(first.ref or "") if first.ref else None
        entry = next(
            (
                candidate
                for candidate in catalog
                if requested_hash is not None
                and (candidate_hash := parse_ref(candidate.manifest.ref)) is not None
                and candidate_hash.value == requested_hash.value
            ),
            None,
        )
        if entry is None:
            raise RefExecError(
                "Error: externalized ref is not available in this binding. Expected REF: tool:<64 hex> or history:<64 hex>"
            )
        first_is_byte_stage = first.byte_count is not None or first.byte_start is not None
        if first.command in {"grep", "wc"} or first_is_byte_stage:
            verified_source = _iter_verified_ref_source_lines(entry, cancelled)
            lines = verified_source
        else:
            measured_bytes, measured_hash = _measure_ref_source_checked(entry.source, cancelled)
            if measured_bytes != entry.manifest.utf8_bytes or measured_hash != entry.manifest.sha256:
                raise RefExecError("Error: externalized ref integrity verification failed")
            lines = _ref_exec_stat(entry, cancelled) if first.command == "stat" else _iter_ref_source_lines(entry.source, cancelled)
        if first.command == "head":
            lines = (
                _ref_exec_head_bytes(lines, first.byte_count, cancelled)
                if first.byte_count is not None
                else _ref_exec_head(lines, first.count or 0, cancelled)
            )
        elif first.command == "tail":
            if first.byte_start is not None:
                lines = _ref_exec_tail_from_bytes(lines, first.byte_start, cancelled)
            elif first.byte_count is not None:
                lines = _ref_exec_tail_bytes(lines, first.byte_count, cancelled)
            else:
                count = first.count or 0
                if count == 0:
                    lines = iter(())
                else:
                    total = _ref_exec_source_line_count(entry.source)
                    lines = _ref_exec_sed(
                        lines, max(1, total - count + 1), None, cancelled
                    )
        elif first.command == "sed":
            lines = _ref_exec_sed(lines, first.start_line or 1, first.end_line, cancelled)
        elif first.command == "grep":
            lines = _ref_exec_grep(lines, first, cancelled)
        elif first.command == "wc":
            lines = _ref_exec_wc(lines, next(iter(first.flags)), cancelled)
    for stage in stages[1:]:
        if stage.command == "head":
            lines = (
                _ref_exec_head_bytes(lines, stage.byte_count, cancelled)
                if stage.byte_count is not None
                else _ref_exec_head(lines, stage.count or 0, cancelled)
            )
        elif stage.command == "tail":
            if stage.byte_start is not None:
                lines = _ref_exec_tail_from_bytes(lines, stage.byte_start, cancelled)
            elif stage.byte_count is not None:
                lines = _ref_exec_tail_bytes(lines, stage.byte_count, cancelled)
            else:
                lines = _ref_exec_tail(lines, stage.count or 0, cancelled)
        elif stage.command == "sed":
            lines = _ref_exec_sed(lines, stage.start_line or 1, stage.end_line, cancelled)
        elif stage.command == "grep":
            lines = _ref_exec_grep(lines, stage, cancelled)
        elif stage.command == "wc":
            lines = _ref_exec_wc(lines, next(iter(stage.flags)), cancelled)
    final = stages[-1]
    bounded_source_range = (
        len(stages) == 2
        and first.command == "tail"
        and first.byte_start is not None
        and final.command == "head"
        and final.byte_count is not None
    )
    if bounded_source_range:
        # The preview's recovery command is tail -c +N REF | head -c M.
        # head's input-relative range must retain the original source window
        # so every subsequent page stops before the already-visible tail.
        def source_ranges(selected: Iterable[RefExecLine]) -> Iterable[RefExecLine]:
            source_start = None
            for line in selected:
                byte_range = line.byte_range
                if byte_range is not None:
                    if source_start is None and not byte_range.actual_empty:
                        source_start = byte_range.actual_start
                    line = replace(
                        line,
                        byte_range=replace(
                            byte_range,
                            requested_start=first.byte_start,
                            requested_end=(source_start or first.byte_start) + final.byte_count - 1,
                        ),
                    )
                yield line

        lines = source_ranges(lines)
    continuation_ref = (
        first.ref
        if bounded_source_range
        or len(stages) == 1
        and (first.command == "cat" or first.command == "tail" and first.byte_start is not None)
        else None
    )
    response = _collect_ref_exec_response(
        lines,
        continuation_ref=continuation_ref,
        final_grep=final.command == "grep" and "c" not in final.flags,
        preserve_source_newlines=(
            len(stages) == 1 and first.command in {"cat", "head"}
        )
        or final.byte_count is not None
        or final.byte_start is not None,
        threshold_tokens=threshold_tokens,
        encoder=encoder,
        cancelled=cancelled,
    )
    if verified_source is not None:
        for _ in verified_source:
            _check_ref_exec_cancelled(cancelled)
    return response
RefTextMeasurement: TypeAlias = tuple[int, int, str]


def _measure_ref_text(text: str) -> RefTextMeasurement | None:
    digest = hashlib.sha256()
    utf8_bytes = 0
    line_count = 0
    for offset in range(0, len(text), REF_TEXT_HASH_CHUNK_CHARS):
        chunk = text[offset : offset + REF_TEXT_HASH_CHUNK_CHARS]
        try:
            encoded = chunk.encode("utf-8")
        except UnicodeEncodeError:
            return None
        utf8_bytes += len(encoded)
        line_count += chunk.count("\n")
        digest.update(encoded)
    if text and not text.endswith("\n"):
        line_count += 1
    return utf8_bytes, line_count, digest.hexdigest()


def _classify_ref_text_sync(
    text: str,
    *,
    threshold_tokens: int,
    encoder: Any = None,
    request: Any = None,
) -> RefTextClassification:
    utf8_bytes = 0
    line_count = 0
    for offset in range(0, len(text), REF_TEXT_HASH_CHUNK_CHARS):
        chunk = text[offset : offset + REF_TEXT_HASH_CHUNK_CHARS]
        try:
            utf8_bytes += len(chunk.encode("utf-8"))
        except UnicodeEncodeError:
            return RefTextClassification(
                eligible=False,
                utf8_bytes=None,
                sha256=None,
                line_count=None,
                token_count=None,
                encoder_failed=False,
            )
        line_count += chunk.count("\n")
    if text and not text.endswith("\n"):
        line_count += 1
    token_count = None
    encoder_failed = False
    eligible = utf8_bytes > REF_EXEC_CLASSIFY_EXACT_ENCODE_MAX_BYTES
    resolved_encoder = encoder
    if not eligible:
        if resolved_encoder is None:
            resolved_encoder, _ = _ref_exec_get_tiktoken_encoder(request)
        if resolved_encoder is None:
            encoder_failed = True
        else:
            token_count = _ref_exec_encode_text_token_count(resolved_encoder, text)
            encoder_failed = token_count is None
            eligible = token_count is not None and token_count >= threshold_tokens
    if not eligible:
        return RefTextClassification(
            eligible=False,
            utf8_bytes=utf8_bytes,
            sha256=None,
            line_count=line_count,
            token_count=token_count,
            encoder_failed=encoder_failed,
        )
    measurement = _measure_ref_text(text)
    if measurement is None:
        return RefTextClassification(
            eligible=False,
            utf8_bytes=None,
            sha256=None,
            line_count=None,
            token_count=None,
            encoder_failed=False,
        )
    utf8_bytes, line_count, text_hash = measurement
    return RefTextClassification(
        eligible=True,
        utf8_bytes=utf8_bytes,
        sha256=text_hash,
        line_count=line_count,
        token_count=token_count,
        encoder_failed=False,
    )


async def classify_ref_text(
    text: str,
    *,
    threshold_tokens: int,
    encoder: Any = None,
    request: Any = None,
) -> RefTextClassification:
    return await asyncio.to_thread(
        _classify_ref_text_sync,
        text,
        threshold_tokens=threshold_tokens,
        encoder=encoder,
        request=request,
    )


def _native_tool_names_by_call_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    candidate_names: dict[str, list[str | None]] = {}
    for assistant_message in messages:
        if assistant_message.get("role") != "assistant":
            continue
        tool_calls = assistant_message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            function = tool_call.get("function") if isinstance(tool_call, dict) else None
            call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            name = function.get("name") if isinstance(function, dict) else None
            if isinstance(call_id, str) and call_id:
                candidate_names.setdefault(call_id, []).append(
                    name if isinstance(name, str) and name else None
                )
    names_by_call_id: dict[str, str] = {}
    for call_id, names in candidate_names.items():
        if REF_EXEC_TOOL_NAME in names:
            names_by_call_id[call_id] = REF_EXEC_TOOL_NAME
        elif len(valid_names := {name for name in names if name is not None}) == 1:
            names_by_call_id[call_id] = next(iter(valid_names))
        else:
            names_by_call_id[call_id] = "unknown"
    return names_by_call_id

def _render_tool_ref_preview_sync(
    text: str,
    ref: str,
    utf8_bytes: int,
    *,
    threshold_tokens: int,
    encoder: Any,
) -> str | None:
    """Keep a near-limit head/tail preview with byte-exact middle recovery.

    Match Core's d40225da3b03 contract: the marker counts against both caps,
    and a preview the token counter cannot measure is judged by the same
    byte upper bound the reader uses. Only bounded edge fragments are
    copied or tokenized, even when the source is a single huge line.
    """
    if utf8_bytes < 2:
        return None
    cancelled = threading.Event()
    marker_overhead = len(
        _ref_exec_truncated_marker(
            f"tail -c +{utf8_bytes + 1} {ref} | head -c {utf8_bytes}"
        )
    ) + 1
    maximum = min(utf8_bytes - 1, REF_EXEC_RESPONSE_MAX_BYTES - marker_overhead)
    if maximum < 0:
        return None
    # UTF-8 uses at least one byte per character. These edge windows suffice
    # for every candidate without encoding or indexing the complete source.
    head_bytes = text[:maximum].encode("utf-8")
    tail_bytes = text[-maximum:].encode("utf-8") if maximum else b""

    def preview_at(prefix_end: int, suffix_start: int) -> str | None:
        if not 0 <= prefix_end <= suffix_start <= len(text):
            return None
        prefix = text[:prefix_end]
        suffix = text[suffix_start:]
        prefix_size = len(prefix.encode("utf-8"))
        omitted = utf8_bytes - prefix_size - len(suffix.encode("utf-8"))
        if omitted <= 0:
            return None
        command = f"tail -c +{prefix_size + 1} {ref} | head -c {omitted}"
        return prefix + _ref_exec_truncated_marker(command) + "\n" + suffix

    def fits(candidate: str | None) -> bool:
        if candidate is None:
            return False
        return _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )

    best = None
    prefix_end, suffix_start = 0, len(text)
    low, high = 0, maximum
    while low <= high:
        retained = (low + high) // 2
        prefix = head_bytes[: retained // 2].decode("utf-8", errors="ignore")
        suffix_size = retained - len(prefix.encode("utf-8"))
        suffix = tail_bytes[-suffix_size:].decode("utf-8", errors="ignore") if suffix_size else ""
        ends = (len(prefix), len(text) - len(suffix))
        candidate = preview_at(*ends)
        if fits(candidate):
            best, (prefix_end, suffix_start) = candidate, ends
            low = retained + 1
        else:
            high = retained - 1
    if best is None:
        return None
    # As in Core, polish each edge after the byte search. The cap bounds
    # work for token counters whose result is not monotone in text length.
    for _ in range(8):
        grew = False
        for ends in ((prefix_end + 1, suffix_start), (prefix_end, suffix_start - 1)):
            candidate = preview_at(*ends)
            if fits(candidate):
                best, (prefix_end, suffix_start) = candidate, ends
                grew = True
        if not grew:
            break
    return best

def _cached_tool_ref_preview_sync(
    text: str,
    ref: str,
    utf8_bytes: int,
    *,
    threshold_tokens: int,
    encoder: Any,
    request: Any,
    cache: dict[tuple[str, int, int], tuple[Any, str]],
) -> str | None:
    if encoder is None:
        encoder, _ = _ref_exec_get_tiktoken_encoder(request)
    key = (ref, threshold_tokens, id(encoder))
    cached = cache.get(key)
    if cached is not None:
        return cached[1]
    preview = _render_tool_ref_preview_sync(
        text, ref, utf8_bytes, threshold_tokens=threshold_tokens, encoder=encoder
    )
    if preview is not None:
        # Retain the encoder to prevent ID reuse within this run. The
        # cache holds bounded renders, not raw sources or reader bindings.
        cache[key] = (encoder, preview)
    return preview


async def project_native_tool_texts(
    messages: list[dict[str, Any]],
    *,
    threshold_tokens: int,
    encoder: Any = None,
    request: Any = None,
    preview_cache: dict[tuple[str, int, int], tuple[Any, str]] | None = None,
    classification_cache: dict[str, RefTextClassification] | None = None,
) -> RefProjectionPlan:
    """Classify oversized native tool texts and build their ref projection plan."""
    catalog_by_hash: dict[str, RefCatalogEntry] = {}
    if preview_cache is None:
        preview_cache = {}
    names_by_call_id = _native_tool_names_by_call_id(messages)

    eligible_results: list[tuple[str, str, RefTextMeasurement]] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        tool_call_id = message.get("tool_call_id")
        paired_name = (
            names_by_call_id.get(tool_call_id, "unknown")
            if isinstance(tool_call_id, str)
            else "unknown"
        )
        classification = None
        if classification_cache is not None:
            classification = classification_cache.get(content)
        if classification is None:
            classification = await classify_ref_text(
                content,
                threshold_tokens=threshold_tokens,
                encoder=encoder,
                request=request,
            )
            if classification_cache is not None:
                classification_cache[content] = classification
        if not classification.eligible:
            continue
        utf8_bytes = classification.utf8_bytes
        line_count = classification.line_count
        text_hash = classification.sha256
        if utf8_bytes is None or line_count is None or text_hash is None:
            raise InvalidRefTextClassificationError
        measurement: RefTextMeasurement = (utf8_bytes, line_count, text_hash)
        eligible_results.append((content, paired_name, measurement))

    for content, paired_name, measurement in eligible_results:
        utf8_bytes, line_count, text_hash = measurement
        ref = f"tool:{text_hash}"
        preview = await asyncio.to_thread(
            _cached_tool_ref_preview_sync,
            content,
            ref,
            utf8_bytes,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            request=request,
            cache=preview_cache,
        )
        if preview is None:
            # Fail closed: eligible text must never reach the provider raw.
            raise RefProjectionError(stage="tool preview rendering")
        manifest = RefManifest(
            ref=ref,
            utf8_bytes=utf8_bytes,
            sha256=text_hash,
        )
        source = ZeroCopySourceHandle(text=content)
        catalog_by_hash.setdefault(
            text_hash,
            RefCatalogEntry(
                manifest=manifest,
                source=source,
                preview_text=preview,
            ),
        )
    catalog = tuple(catalog_by_hash.values())
    return RefProjectionPlan(
        catalog=catalog,
        manifests=tuple(entry.manifest for entry in catalog),
        reader_schema=REF_EXEC_TOOL_SPEC if catalog else None,
    )

def _apply_ref_projection_plan_sync(
    messages: list[dict[str, Any]],
    plan: RefProjectionPlan,
) -> list[dict[str, Any]]:
    projected = copy.deepcopy(messages)
    source_previews: dict[str, str] = {}
    for entry in plan.catalog:
        parsed = parse_ref(entry.manifest.ref)
        if (
            parsed is not None
            and parsed.kind == "tool"
            and isinstance(entry.source, ZeroCopySourceHandle)
            and entry.preview_text is not None
        ):
            source_previews.setdefault(entry.source.text, entry.preview_text)
    for message in projected:
        if message.get("role") != "tool":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        preview = source_previews.get(content)
        if preview is None:
            continue
        message["content"] = preview
    return projected


async def apply_ref_projection_plan(
    messages: list[dict[str, Any]],
    plan: RefProjectionPlan,
) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_apply_ref_projection_plan_sync, messages, plan)


def _ref_exec_encode_text_token_count(encoder: Any, text: str) -> int | None:
    try:
        return len(encoder.encode(text, disallowed_special=()))
    except TypeError:
        try:
            return len(encoder.encode(text))
        except Exception:
            return None
    except Exception:
        return None


def _ref_exec_tiktoken_encoding_names(request: Any = None) -> list[str]:
    names: list[str] = []
    config = getattr(getattr(getattr(request, "app", None), "state", None), "config", None)
    configured = getattr(config, "TIKTOKEN_ENCODING_NAME", None)
    if configured:
        names.append(str(configured))
    try:
        from open_webui import config as open_webui_config

        fallback = getattr(open_webui_config, "TIKTOKEN_ENCODING_NAME", None)
        if fallback:
            names.append(str(fallback))
    except Exception:
        pass
    names.append("cl100k_base")
    return list(dict.fromkeys(names))


def _ref_exec_get_tiktoken_encoder(request: Any = None) -> tuple[Any | None, str | None]:
    try:
        import tiktoken
    except Exception:
        return None, None

    for encoding_name in _ref_exec_tiktoken_encoding_names(request):
        try:
            return tiktoken.get_encoding(encoding_name), encoding_name
        except Exception:
            continue
    return None, None


REF_EXEC_TOOL_SPEC = MappingProxyType(
    {
        "type": "function",
        "function": MappingProxyType(
            {
                "name": REF_EXEC_TOOL_NAME,
                "description": (
                    "Read externalized content in this sub-agent run. Oversized tool results include a bounded head/tail preview and a tool:<64 hex> ref; compacted history uses history:<64 hex>. "
                    "A <agent_ref_truncated> marker embeds a next command to read the omitted span; follow continuation commands across pages when needed. Commands: ls [tool|history]; stat REF; "
                    "wc -l|-w|-c REF; cat REF; head [-n N|-N|-c N] REF; tail [-n N|-N|-c N|-c +N] REF; sed -n 'M,Np' REF; grep [-E] [-i] [-n] [-c] [-o] [--] PATTERN REF "
                    "(patterns match literally unless regex syntax is auto-detected; -E forces regex). REF is the complete token including its tool:/history: prefix, exactly as written. Pipelines are supported; "
                    "only grep/head/tail/sed/wc consume piped input, e.g. grep -n PATTERN tool:<hash> | head -20. Start with stat, then prefer grep/sed/head over cat for large refs."
                ),
                "parameters": MappingProxyType(
                    {
                        "type": "object",
                        "properties": MappingProxyType(
                            {
                                "command": MappingProxyType(
                                    {
                                        "type": "string",
                                        "description": "One command line, max 1,024 UTF-8 bytes, e.g. sed -n '1,120p' tool:<64 hex>",
                                    }
                                )
                            }
                        ),
                        "required": ("command",),
                        "additionalProperties": False,
                    }
                ),
            }
        ),
    }
)


def _mutable_ref_schema_value(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {key: _mutable_ref_schema_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_ref_schema_value(item) for item in value]
    return value


def ref_exec_tool_spec_payload() -> dict[str, Any]:
    payload = _mutable_ref_schema_value(REF_EXEC_TOOL_SPEC)
    if not isinstance(payload, dict):
        raise RefProjectionError(stage="reader schema rendering")
    return payload


def _truncation_only_marker(omitted_bytes: int) -> str:
    payload = json.dumps(
        {"omitted": omitted_bytes},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"\n<{'agent_ref_truncated'}>{payload}</{'agent_ref_truncated'}>"


def render_truncate_preview_sync(
    text: str,
    utf8_bytes: int,
    *,
    threshold_tokens: int,
    encoder: Any,
) -> str | None:
    """Irreversible middle truncation using the ref-exec preview renderer."""
    if utf8_bytes < 2:
        return None
    cancelled = threading.Event()
    marker_overhead = len(_truncation_only_marker(utf8_bytes)) + 1
    maximum = min(utf8_bytes - 1, REF_EXEC_RESPONSE_MAX_BYTES - marker_overhead)
    if maximum < 0:
        return None
    head_bytes = text[:maximum].encode("utf-8")
    tail_bytes = text[-maximum:].encode("utf-8") if maximum else b""

    def preview_at(prefix_end: int, suffix_start: int) -> str | None:
        if not 0 <= prefix_end <= suffix_start <= len(text):
            return None
        prefix = text[:prefix_end]
        suffix = text[suffix_start:]
        prefix_size = len(prefix.encode("utf-8"))
        omitted = utf8_bytes - prefix_size - len(suffix.encode("utf-8"))
        if omitted <= 0:
            return None
        return prefix + _truncation_only_marker(omitted) + "\n" + suffix

    def fits(candidate: str | None) -> bool:
        if candidate is None:
            return False
        return _ref_exec_response_fits(
            candidate,
            threshold_tokens=threshold_tokens,
            encoder=encoder,
            cancelled=cancelled,
        )

    best = None
    prefix_end, suffix_start = 0, len(text)
    low, high = 0, maximum
    while low <= high:
        retained = (low + high) // 2
        prefix = head_bytes[: retained // 2].decode("utf-8", errors="ignore")
        suffix_size = retained - len(prefix.encode("utf-8"))
        suffix = tail_bytes[-suffix_size:].decode("utf-8", errors="ignore") if suffix_size else ""
        ends = (len(prefix), len(text) - len(suffix))
        candidate = preview_at(*ends)
        if fits(candidate):
            best, (prefix_end, suffix_start) = candidate, ends
            low = retained + 1
        else:
            high = retained - 1
    if best is None:
        return None
    for _ in range(8):
        grew = False
        for ends in ((prefix_end + 1, suffix_start), (prefix_end, suffix_start - 1)):
            candidate = preview_at(*ends)
            if fits(candidate):
                best, (prefix_end, suffix_start) = candidate, ends
                grew = True
        if not grew:
            break
    return best


class RefRunStore:
    """Run-local catalog of externalized refs (tool texts + folded history)."""

    def __init__(self) -> None:
        self._entries: dict[str, RefCatalogEntry] = {}
        self._history_line_counts: dict[str, int] = {}
        self.preview_cache: dict[tuple[str, int, int], tuple[Any, str]] = {}

    def catalog_tuple(self) -> tuple[RefCatalogEntry, ...]:
        return tuple(self._entries.values())

    def intern_tool_text(self, text: str, entry: RefCatalogEntry) -> str:
        parsed = parse_ref(entry.manifest.ref)
        key = parsed.value if parsed is not None else entry.manifest.ref
        self._entries.setdefault(key, entry)
        return entry.manifest.ref

    def add_history_records(self, records: tuple[str, ...]) -> str:
        digest = hashlib.sha256()
        utf8_bytes = 0
        for index, record in enumerate(records):
            encoded = record.encode("utf-8")
            if index > 0:
                digest.update(b"\n")
                utf8_bytes += 1
            digest.update(encoded)
            utf8_bytes += len(encoded)
        text_hash = digest.hexdigest()
        ref = f"history:{text_hash}"
        self._history_line_counts[text_hash] = len(records)
        if text_hash not in self._entries:
            self._entries[text_hash] = RefCatalogEntry(
                manifest=RefManifest(ref=ref, utf8_bytes=utf8_bytes, sha256=text_hash),
                source=JsonlHistorySourceHandle(
                    records=records,
                    utf8_bytes=utf8_bytes,
                    sha256=text_hash,
                    line_count=len(records),
                ),
            )
        return ref

    def history_manifest_payload(self, ref: str) -> dict[str, Any] | None:
        requested = parse_ref(ref)
        if requested is None:
            return None
        entry = self._entries.get(requested.value)
        if entry is None or entry.manifest.utf8_bytes is None:
            return None
        return {
            "bytes": entry.manifest.utf8_bytes,
            "kind": "history",
            "lines": self._history_line_counts.get(requested.value),
            "ref": ref,
            "version": 1,
        }


def build_ref_reader(
    store: RefRunStore,
    *,
    threshold_tokens: int,
    encoder: Any = None,
) -> Callable[[str], Awaitable[str]]:
    """Bind a reader tool callable to a run-local ref store."""

    async def reader(command: str = "") -> str:
        """Inspect one run-local externalized ref with bounded virtual reader commands.

        Follow the next command in truncation markers to read omitted spans across pages.

        :param command: Use ls, stat, wc, head, tail, sed -n, grep, or cat and optional bounded pipelines.
        """
        try:
            if not isinstance(command, str):
                return REF_EXEC_USAGE_ERROR
            stages = _parse_ref_exec_command(command)
            catalog = store.catalog_tuple()
            cancelled = threading.Event()
            resolved_encoder = encoder
            if resolved_encoder is None and stages[-1].command != "wc":
                # tiktoken.get_encoding may download BPE files with no
                # timeout, so resolution must not run on the event loop;
                # it also must not run inside the sync worker, because a
                # cancelled call would then join-wait the download.
                resolved_encoder, _ = await asyncio.to_thread(
                    _ref_exec_get_tiktoken_encoder
                )
            try:
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        _execute_ref_reader_sync,
                        stages,
                        catalog,
                        threshold_tokens=threshold_tokens,
                        encoder=resolved_encoder,
                        cancelled=cancelled,
                    )
                )
                return await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled.set()
                try:
                    await asyncio.shield(worker)
                except Exception:
                    pass
                raise
        except RefExecError as exc:
            return str(exc)
        except Exception:
            LOG.exception("Unexpected externalized ref reader failure")
            return "Error: externalized ref reader is unavailable"

    reader.__name__ = REF_EXEC_TOOL_NAME
    return reader
