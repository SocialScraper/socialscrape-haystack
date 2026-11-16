"""The endpoint catalogue this package ships, and the routing the tools share.

`endpoints.json` is generated from the SocialScrape API reference: every data
endpoint that is live, with its parameters. Nothing here makes a request, so
looking an endpoint up or checking parameters is free.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any
from urllib.parse import urlsplit

# The jobs the cross-platform tools do, and the endpoint that does each one per
# platform. Every id is checked against endpoints.json by the tests.
ROLES: dict[str, dict[str, str]] = {
    "profile": {
        "apple_music": "apple_music.artist",
        "bluesky": "bluesky.profile",
        "cosmos": "cosmos.profile",
        "facebook": "facebook.profile",
        "github": "github.user",
        "instagram": "instagram.profile",
        "kwai": "kwai.profile",
        "linkedin": "linkedin.profile",
        "snapchat": "snapchat.profile",
        "soundcloud": "soundcloud.artist",
        "spotify": "spotify.artist",
        "threads": "threads.profile",
        "tiktok": "tiktok.profile",
        "truthsocial": "truthsocial.profile",
        "twitch": "twitch.profile",
        "twitter": "twitter.profile",
        "youtube": "youtube.channel",
    },
    "posts": {
        "bluesky": "bluesky.profile.posts",
        "cosmos": "cosmos.profile.posts",
        "facebook": "facebook.profile.posts",
        "instagram": "instagram.profile.posts",
        "kwai": "kwai.profile.posts",
        "linkedin": "linkedin.company.posts",
        "rumble": "rumble.channel.videos",
        "telegram": "telegram.channel.posts",
        "threads": "threads.profile.posts",
        "tiktok": "tiktok.profile.videos",
        "truthsocial": "truthsocial.profile.posts",
        "twitch": "twitch.user.videos",
        "twitter": "twitter.profile.tweets",
        "youtube": "youtube.channel.videos",
    },
    "post": {
        "bluesky": "bluesky.post",
        "cosmos": "cosmos.post",
        "facebook": "facebook.post",
        "instagram": "instagram.post",
        "kick": "kick.clip",
        "kwai": "kwai.post",
        "linkedin": "linkedin.post",
        "pinterest": "pinterest.pin",
        "reddit": "reddit.post",
        "rumble": "rumble.video",
        "snapchat": "snapchat.spotlight",
        "telegram": "telegram.post",
        "threads": "threads.post",
        "tiktok": "tiktok.video",
        "truthsocial": "truthsocial.post",
        "twitch": "twitch.clip",
        "twitter": "twitter.tweet",
        "youtube": "youtube.video",
    },
    "comments": {
        "facebook": "facebook.post.comments",
        "instagram": "instagram.post.comments",
        "reddit": "reddit.post.comments",
        "rumble": "rumble.video.comments",
        "snapchat": "snapchat.spotlight.comments",
        "tiktok": "tiktok.video.comments",
        "youtube": "youtube.video.comments",
    },
    "search": {
        "apple_music": "apple_music.search",
        "cosmos": "cosmos.search",
        "facebook": "facebook.search.videos",
        "google": "google.search",
        "instagram": "instagram.search.reels",
        "linkedin": "linkedin.search.posts",
        "pinterest": "pinterest.search",
        "reddit": "reddit.search",
        "rumble": "rumble.search",
        "spotify": "spotify.search",
        "threads": "threads.search",
        "tiktok": "tiktok.search.keyword",
        "twitter": "twitter.search.posts",
        "youtube": "youtube.search",
    },
}

TRANSCRIPT = "transcript"
IDENTIFIERS = ("handle", "url", "id")


@dataclass(frozen=True)
class Endpoint:
    """One live endpoint: what it is called, what it takes, what it costs."""

    id: str
    platform: str
    title: str
    path: str
    summary: str
    description: str
    response: str
    paginated: bool
    credits: str
    params: tuple[Mapping[str, Any], ...]
    one_of: tuple[tuple[str, ...], ...]
    example: Mapping[str, Any]
    common: bool  # takes trim, raw and max_age_days (every data endpoint does; the account ones do not)

    @property
    def names(self) -> tuple[str, ...]:
        """Parameter names of its own, without the ones every endpoint takes."""
        return tuple(p["name"] for p in self.params)

    @property
    def required(self) -> tuple[str, ...]:
        return tuple(p["name"] for p in self.params if p["required"])

    @property
    def hosts(self) -> tuple[str, ...]:
        """Hosts its `url` parameter accepts, when it has one."""
        for p in self.params:
            if p["name"] == "url":
                return tuple(p.get("hosts") or ())
        return ()

    def signature(self) -> str:
        """`handle | id (one needed), sort, cursor`: the parameters in one line."""
        grouped = {name for group in self.one_of for name in group}
        bits = [" | ".join(group) + " (one needed)" for group in self.one_of]
        bits += [name + " (required)" for name in self.required if name not in grouped]
        bits += [name for name in self.names if name not in grouped and name not in self.required]
        return ", ".join(bits) or "none"


@lru_cache(maxsize=1)
def _load() -> tuple[dict[str, Endpoint], tuple[Mapping[str, Any], ...]]:
    text = resources.files(__package__).joinpath("endpoints.json").read_text(encoding="utf-8")
    doc = json.loads(text)
    endpoints = {
        row["id"]: Endpoint(
            id=row["id"], platform=row["platform"], title=row["title"], path=row["path"], summary=row["summary"],
            description=row["description"], response=row["response"], paginated=row["paginated"],
            credits=row["credits"], params=tuple(row["params"]),
            one_of=tuple(tuple(group) for group in row["one_of"]), example=row["example"],
            common=row["common"],
        )
        for row in doc["endpoints"]
    }
    return endpoints, tuple(doc["common"])


def endpoints() -> dict[str, Endpoint]:
    """Every endpoint in the catalogue, by id, in the order of the API reference."""
    return _load()[0]


def common_params() -> tuple[Mapping[str, Any], ...]:
    """The parameters every data endpoint takes: trim, raw, max_age_days."""
    return _load()[1]


def platforms() -> dict[str, str]:
    """Platform key to display name, in the order of the API reference."""
    out: dict[str, str] = {}
    for ep in endpoints().values():
        out.setdefault(ep.platform, ep.title)
    return out


def get_endpoint(endpoint_id: str) -> Endpoint:
    """The endpoint with this id, or a ValueError that says where to look."""
    found = endpoints().get(endpoint_id)
    if found is None:
        near = [i for i in endpoints() if i.startswith(endpoint_id.split(".")[0] + ".")][:12]
        hint = f" Ids that start the same way: {', '.join(near)}." if near else ""
        raise ValueError(f"unknown endpoint id {endpoint_id!r}.{hint} List them with socialscrape_list_endpoints.")
    return found


def check_params(ep: Endpoint, params: Mapping[str, Any]) -> dict[str, Any]:
    """The parameters to send, with None dropped, or a ValueError naming what is wrong.

    Checked before any request is made: an unknown name, a missing required
    parameter, or none of a group where one is needed.
    """
    sent = {k: v for k, v in params.items() if v is not None}
    allowed = set(ep.names) | ({p["name"] for p in common_params()} if ep.common else set())
    unknown = sorted(set(sent) - allowed)
    if unknown:
        raise ValueError(f"{ep.id} does not take {', '.join(unknown)}. It takes: {ep.signature()}.")
    missing = [name for name in ep.required if name not in sent]
    if missing:
        raise ValueError(f"{ep.id} needs {', '.join(missing)}. It takes: {ep.signature()}.")
    for group in ep.one_of:
        if not any(name in sent for name in group):
            raise ValueError(f"{ep.id} needs one of {', '.join(group)}. It takes: {ep.signature()}.")
    return sent


def _host_matches(host: str, allowed: tuple[str, ...]) -> bool:
    return any(host == h or host.endswith("." + h) for h in allowed)


def platform_of_url(role: str, url: str) -> str | None:
    """Which platform of a role a URL belongs to, read from its host."""
    host = (urlsplit(url if "//" in url else "https://" + url).hostname or "").lower()
    if not host:
        return None
    for platform, endpoint_id in ROLES[role].items():
        if _host_matches(host, endpoints()[endpoint_id].hosts):
            return platform
    return None


def route(role: str, platform: str | None, *, handle: str | None = None, url: str | None = None,
          id: str | None = None) -> tuple[Endpoint, dict[str, Any]]:
    """The endpoint that does `role` on a platform, and the one identifier to send it.

    Without a platform the URL's host decides. Endpoints differ in what they
    look a thing up by, so an identifier the endpoint does not take is an
    error that says which ones it does.
    """
    table = ROLES[role]
    if not platform:
        if not url:
            raise ValueError(f"pass a platform ({', '.join(sorted(table))}), or a url it can be read from")
        platform = platform_of_url(role, url)
        if platform is None:
            raise ValueError(f"no platform recognised in {url!r}; pass platform, one of {', '.join(sorted(table))}")
    key = platform.strip().lower().replace("-", "_")
    key = {"x": "twitter"}.get(key, key)
    if key not in table:
        raise ValueError(f"no {role} lookup for {platform!r}; platforms: {', '.join(sorted(table))}")
    ep = endpoints()[table[key]]
    given = {name: value for name, value in (("handle", handle), ("url", url), ("id", id)) if value}
    for name in IDENTIFIERS:
        if name in given and name in ep.names:
            return ep, {name: given[name]}
    takes = " or ".join(name for name in IDENTIFIERS if name in ep.names)
    got = ", ".join(given) or "nothing"
    raise ValueError(f"{ep.id} looks things up by {takes}; got {got}")
