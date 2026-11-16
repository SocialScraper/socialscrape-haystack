"""Turn API objects into Haystack Documents: the object's text is the content, the rest is metadata."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from haystack import Document

# Which fields of an object are its text, by the object's type. Anything else goes to meta.
_TEXT_FIELDS: dict[str, tuple[str, ...]] = {
    "post": ("title", "text"),
    "comment": ("text",),
    "review": ("text",),
    "transcript": ("text",),
    "profile": ("name", "headline", "bio"),
    "community": ("name", "title", "description"),
    "collection": ("title", "description"),
    "ad": ("title", "body"),
    "product": ("title", "description"),
    "repo": ("full_name", "description"),
    "search_result": ("title", "snippet"),
    "link_page": ("name", "bio"),
    "event": ("title", "description"),
}
_FALLBACK_TEXT = ("title", "name", "text", "description")
_SCALARS = (str, int, float, bool)


def _meta(obj: Mapping[str, Any], text_fields: tuple[str, ...]) -> dict[str, Any]:
    """Flat meta a document store accepts: the object's scalar fields, its counts and who made it."""
    meta: dict[str, Any] = {}
    for key, value in obj.items():
        if key in text_fields or key == "extra":
            continue
        if key == "source" and isinstance(value, str):
            meta[f"{obj.get('object') or 'item'}_source"] = value  # a transcript's `source` is captions or ai
        elif isinstance(value, _SCALARS):
            meta[key] = value
        elif key == "stats" and isinstance(value, Mapping):
            meta.update({name: count for name, count in value.items() if isinstance(count, _SCALARS)})
        elif key in ("author", "owner", "advertiser", "seller", "host") and isinstance(value, Mapping):
            who = value.get("handle") or value.get("name")
            if who:
                meta[key] = who
            if isinstance(value.get("url"), str):
                meta[f"{key}_url"] = value["url"]
    if "url" in meta:
        meta["source"] = meta["url"]
    return meta


def to_document(obj: Any, endpoint_id: str) -> Document:
    """One API object as a Document. Meta has `endpoint`, `platform` and, when it has one, `url` and `source`."""
    if not isinstance(obj, Mapping):
        return Document(content=json.dumps(obj, ensure_ascii=False), meta={"endpoint": endpoint_id})
    kind = obj.get("object")
    fields = _TEXT_FIELDS.get(kind, _FALLBACK_TEXT) if isinstance(kind, str) else _FALLBACK_TEXT
    parts = [obj[name].strip() for name in fields if isinstance(obj.get(name), str) and obj[name].strip()]
    if parts:
        content, used = "\n\n".join(parts), fields
    else:  # nothing that reads as text (a credit balance, a contribution calendar): keep the whole object
        content, used = json.dumps(obj, ensure_ascii=False), ()
    meta = _meta(obj, used)
    meta["endpoint"] = endpoint_id
    ident = obj.get("id") or obj.get("video_id")
    doc_id = f"{obj.get('platform')}:{kind}:{ident}" if ident and obj.get("platform") and kind else None
    if doc_id:
        return Document(id=doc_id, content=content, meta=meta)
    return Document(content=content, meta=meta)


def items_of(data: Any) -> tuple[list[Any], str | None]:
    """The items of an answer and the cursor for the next page: a list page has them, any other object is one item."""
    if isinstance(data, Mapping) and data.get("object") == "list":
        cursor = data.get("cursor") if data.get("has_more") is not False else None
        return list(data.get("items") or []), cursor or None
    return [data], None
