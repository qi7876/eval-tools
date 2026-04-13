"""Helpers for aggregating OpenAI-compatible chat completion streams."""

from __future__ import annotations

from typing import Any


def _text_from_content_part(part: Any) -> str:
    if part is None:
        return ""
    if isinstance(part, str):
        return part
    if isinstance(part, list):
        return "".join(_text_from_content_part(item) for item in part)
    if isinstance(part, dict):
        text = part.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            value = text.get("value")
            if isinstance(value, str):
                return value
        value = part.get("value")
        if isinstance(value, str):
            return value
        return ""
    text_attr = getattr(part, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(text_attr, dict):
        value = text_attr.get("value")
        if isinstance(value, str):
            return value
    value_attr = getattr(part, "value", None)
    if isinstance(value_attr, str):
        return value_attr
    return ""


def collect_chat_completion_stream_texts(stream: Any) -> list[str]:
    texts_by_index: dict[int, list[str]] = {}
    for chunk in stream:
        for fallback_index, choice in enumerate(getattr(chunk, "choices", []) or []):
            choice_index = getattr(choice, "index", fallback_index)
            delta = getattr(choice, "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content is None:
                continue
            text = _text_from_content_part(content)
            if not text:
                continue
            texts_by_index.setdefault(int(choice_index), []).append(text)
    if not texts_by_index:
        return [""]
    return ["".join(texts_by_index[index]) for index in sorted(texts_by_index)]
