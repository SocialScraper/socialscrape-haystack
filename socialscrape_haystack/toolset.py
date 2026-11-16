"""SocialScrapeToolset: the SocialScrape jobs agents do most, as Haystack Tools.

One tool per common job (a profile, a profile's posts, one post, its comments, a video transcript, a search),
each across platforms, plus two for everything else: `socialscrape_list_endpoints` shows what exists and
`socialscrape_call_endpoint` runs any endpoint by its id. The tools mirror the LangChain toolkit's.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools import Tool, Toolset
from haystack.utils import Secret, deserialize_secrets_inplace
from socialscrape import APIError, SocialScrapeConnectionError, SocialScrapeTimeoutError

from ._catalog import (
    IDENTIFIERS,
    ROLES,
    TRANSCRIPT,
    Endpoint,
    common_params,
    endpoints,
    get_endpoint,
    platforms,
    route,
)
from ._client import Session

# Haystack added Tool(async_function=...) after 2.14; older releases run the sync function in a thread instead.
_TOOL_TAKES_ASYNC = "async_function" in inspect.signature(Tool.__init__).parameters

_FAILURES = (ValueError, APIError, SocialScrapeConnectionError, SocialScrapeTimeoutError)

TOOL_NAMES = (
    "socialscrape_get_profile",
    "socialscrape_list_posts",
    "socialscrape_get_post",
    "socialscrape_list_comments",
    "socialscrape_get_transcript",
    "socialscrape_search",
    "socialscrape_list_endpoints",
    "socialscrape_call_endpoint",
)


def _titles(role: str) -> str:
    return ", ".join(endpoints()[endpoint_id].title for endpoint_id in ROLES[role].values())


def _lookups(role: str) -> str:
    rows = []
    for platform, endpoint_id in ROLES[role].items():
        names = [name for name in IDENTIFIERS if name in endpoints()[endpoint_id].names]
        rows.append(f"{platform}: {' or '.join(names)}")
    return "; ".join(rows)


def _line(ep: Endpoint) -> str:
    return f"{ep.id}: {ep.summary}. Takes: {ep.signature()}. Returns {ep.response}. Price: {ep.credits}."


def _detail(ep: Endpoint) -> str:
    rows = [f"{ep.id}: {ep.summary}", ep.description,
            f"Returns {ep.response}" + (", one page per call: pass `cursor` back for the next page." if ep.paginated
                                        else ".") + f" Price: {ep.credits}.",
            "Parameters:" if ep.params else "No parameters."]
    for p in ep.params:
        facts = [p["type"]]
        if p["required"]:
            facts.append("required")
        if p.get("enum"):
            facts.append("one of " + ", ".join(str(v) for v in p["enum"]))
        if "default" in p:
            facts.append(f"default {p['default']}")
        if "minimum" in p or "maximum" in p:
            facts.append(f"{p.get('minimum', '')} to {p.get('maximum', '')}".strip())
        rows.append(f"- {p['name']} ({'; '.join(facts)}): {p['description']}")
    for group in ep.one_of:
        rows.append(f"One of {', '.join(group)} is needed.")
    if ep.common:
        rows.append("Also accepted: " + ", ".join(p["name"] for p in common_params()) + ".")
    if ep.example:
        rows.append("Example params: " + json.dumps(ep.example, ensure_ascii=False))
    return "\n".join(rows)


def _lookup(platform: str | None = None, endpoint: str | None = None, search: str | None = None) -> str:
    """What exists, from the catalogue. Makes no request."""
    if endpoint:
        return _detail(get_endpoint(endpoint.strip()))
    if platform:
        key = platform.strip().lower().replace("-", "_")
        key = {"x": "twitter"}.get(key, key)
        rows = [ep for ep in endpoints().values() if ep.platform == key]
        if not rows:
            raise ValueError(f"no platform {platform!r}; platforms: {', '.join(platforms())}")
        return "\n".join(_line(ep) for ep in rows)
    if search:
        words = search.lower().split()
        found = [ep for ep in endpoints().values()
                 if all(w in f"{ep.id} {ep.summary} {ep.description}".lower() for w in words)]
        if not found:
            return f"No endpoint mentions {search!r}. Call with no arguments for the full list."
        more = f"\n{len(found) - 30} more: narrow the search." if len(found) > 30 else ""
        return "\n".join(_line(ep) for ep in found[:30]) + more
    by_platform: dict[str, list[str]] = {}
    for ep in endpoints().values():
        by_platform.setdefault(ep.platform, []).append(ep.id)
    head = (f"{len(endpoints())} endpoints, listed by platform key. Pass `platform` for what its endpoints "
            "take, or `endpoint` for one endpoint's parameters.")
    return "\n".join([head] + [f"{key} ({platforms()[key]}): {', '.join(ids)}" for key, ids in by_platform.items()])


def _transcript_help() -> str:
    return endpoints()[TRANSCRIPT].description.split(" Same price")[0]


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def _text(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


_CURSOR = _text("The `cursor` of the previous answer, to get the next page. Leave it out for the first page.")
_HANDLE = _text("Username or handle, with or without the leading @.")
_PROFILE_URL = _text("Full URL of the profile, channel or page.")
_ACCOUNT_ID = _text("The platform's own id of the account, where it has one.")
_POST_ID = _text("The post's id, on platforms that look a post up by id. Needs platform.")


def _platform(role: str, description: str = "Platform key.") -> dict[str, Any]:
    return {"type": "string", "enum": sorted(ROLES[role]), "description": description}


_FROM_LINK = "Platform key. Leave it out when you pass a url: the platform is read from the link."


@dataclass
class SocialScrapeToolset(Toolset):
    """Eight SocialScrape tools for a Haystack Agent, sharing one API client.

    Tools: `socialscrape_get_profile`, `socialscrape_list_posts`, `socialscrape_get_post`,
    `socialscrape_list_comments`, `socialscrape_get_transcript`, `socialscrape_search`,
    `socialscrape_list_endpoints` and `socialscrape_call_endpoint`.

    A tool answers with a JSON string: `data` (the endpoint's object, or a list page with `items`, `cursor` and
    `has_more`), `credits_charged` and `credits_remaining`. A request the API refuses is free and comes back as
    the tool's answer (`error: not_found: ...`), so the agent can read it and try something else. A missing API
    key raises when the first tool is called.

    Usage example:
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from socialscrape_haystack import SocialScrapeToolset

    agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=SocialScrapeToolset())
    result = agent.run(messages=[ChatMessage.from_user("How many followers does @nasa have on TikTok?")])
    ```
    """

    api_key: Secret = field(default_factory=lambda: Secret.from_env_var("SOCIALSCRAPE_API_KEY"))
    """SocialScrape API key. Read from `SOCIALSCRAPE_API_KEY` by default."""
    base_url: str | None = None
    """API base URL override."""
    max_age_days: float | None = None
    """Accept a cached result up to this many days old (0 to 30), which costs 0 credits."""
    trim: bool = True
    """Ask for the smaller response, without null fields and the `extra` object."""
    client: Any = field(default=None, repr=False, compare=False)
    """A `socialscrape.SocialScrape` you configured yourself, used instead of making one."""
    async_client: Any = field(default=None, repr=False, compare=False)
    """A `socialscrape.AsyncSocialScrape` for async calls, used instead of making one per call."""
    _session: Session | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.tools = self._build()
        super().__post_init__()

    def _get_session(self) -> Session:
        if self._session is None:
            self._session = Session(api_key=self.api_key.resolve_value() if self.client is None else None,
                                    base_url=self.base_url, client=self.client, async_client=self.async_client)
        return self._session

    def _query(self, endpoint_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Add the toolset's own trim and max_age_days; a value the caller passed wins."""
        if get_endpoint(endpoint_id).common:
            params = {"trim": True if self.trim else None, "max_age_days": self.max_age_days, **params}
        return params

    def _answer(self, resp: Any, data: Any = None) -> str:
        out: dict[str, Any] = {"data": resp.data if data is None else data, "credits_charged": resp.credits_charged,
                               "credits_remaining": resp.credits_remaining}
        if resp.cached:
            out["cached"] = True
        return json.dumps(out, ensure_ascii=False)

    def _runner(self, plan: Callable[..., tuple[str, dict[str, Any]]],
                shape: Callable[[Any, dict[str, Any]], Any] | None = None) -> tuple[Callable[..., str],
                                                                                      Callable[..., Any]]:
        """A sync and an async function for one tool. Failures come back as text the agent can act on."""

        def fail(e: Exception) -> str:
            return f"error: {e}"

        def run(**kwargs: Any) -> str:
            session = self._get_session()  # a missing key raises here instead of being answered as text
            try:
                endpoint_id, params = plan(**kwargs)
                resp = session.call(endpoint_id, self._query(endpoint_id, params))
            except _FAILURES as e:
                return fail(e)
            return self._answer(resp, shape(resp.data, kwargs) if shape else None)

        async def arun(**kwargs: Any) -> str:
            session = self._get_session()
            try:
                endpoint_id, params = plan(**kwargs)
                resp = await session.acall(endpoint_id, self._query(endpoint_id, params))
            except _FAILURES as e:
                return fail(e)
            return self._answer(resp, shape(resp.data, kwargs) if shape else None)

        return run, arun

    def _tool(self, name: str, description: str, parameters: dict[str, Any], plan: Callable[..., Any],
              shape: Callable[[Any, dict[str, Any]], Any] | None = None) -> Tool:
        run, arun = self._runner(plan, shape)
        extra: dict[str, Any] = {"async_function": arun} if _TOOL_TAKES_ASYNC else {}
        return Tool(name=name, description=description, parameters=parameters, function=run, **extra)

    def _build(self) -> list[Tool]:
        def profile(platform: str, handle: str | None = None, url: str | None = None, id: str | None = None):
            ep, ident = route("profile", platform, handle=handle, url=url, id=id)
            return ep.id, ident

        def posts(platform: str, handle: str | None = None, url: str | None = None, id: str | None = None,
                  cursor: str | None = None):
            ep, ident = route("posts", platform, handle=handle, url=url, id=id)
            return ep.id, {**ident, "cursor": cursor}

        def post(url: str | None = None, platform: str | None = None, id: str | None = None):
            ep, ident = route("post", platform, url=url, id=id)
            return ep.id, ident

        def comments(url: str | None = None, platform: str | None = None, id: str | None = None,
                     cursor: str | None = None):
            ep, ident = route("comments", platform, url=url, id=id)
            return ep.id, {**ident, "cursor": cursor}

        def transcript(url: str, language: str | None = None, allow_long_transcription: bool = False,
                       include_segments: bool = False):
            return TRANSCRIPT, {"url": url, "language": language,
                                "allow_long_transcription": True if allow_long_transcription else None}

        def no_segments(data: Any, kwargs: dict[str, Any]) -> Any:
            if isinstance(data, dict) and not kwargs.get("include_segments"):
                return {k: v for k, v in data.items() if k != "segments"}
            return data

        def search(platform: str, query: str, cursor: str | None = None):
            table = ROLES["search"]
            key = platform.strip().lower().replace("-", "_")
            key = {"x": "twitter"}.get(key, key)
            if key not in table:
                raise ValueError(f"no search for {platform!r}; platforms: {', '.join(sorted(table))}")
            ep = endpoints()[table[key]]
            return ep.id, {"query": query, "cursor": cursor if "cursor" in ep.names else None}

        def call(endpoint: str, params: dict[str, Any] | None = None):
            return get_endpoint(endpoint.strip()).id, dict(params or {})

        def list_endpoints(platform: str | None = None, endpoint: str | None = None, search: str | None = None) -> str:
            try:
                return _lookup(platform, endpoint, search)
            except ValueError as e:
                return f"error: {e}"

        tools = [
            self._tool(
                "socialscrape_get_profile",
                "Get a profile from a social platform: name, bio, links, verification and follower counts. "
                f"Platforms: {_titles('profile')}. Pass the platform and one identifier. What each platform is "
                f"looked up by: {_lookups('profile')}. Costs 1 credit; a lookup that fails is free.",
                _schema({"platform": _platform("profile"), "handle": _HANDLE, "url": _PROFILE_URL,
                         "id": _ACCOUNT_ID}, ["platform"]),
                profile),
            self._tool(
                "socialscrape_list_posts",
                "List the latest posts or videos of a profile, channel or page, newest first, one page per call: "
                f"text, time, stats and media links per post. Platforms: {_titles('posts')}. Pass the platform and "
                f"one identifier. What each platform is looked up by: {_lookups('posts')}. When the answer has "
                "`has_more` true, pass its `cursor` back for the next page. Costs 1 credit per page; a failed "
                "call is free.",
                _schema({"platform": _platform("posts"), "handle": _HANDLE, "url": _PROFILE_URL,
                         "id": _ACCOUNT_ID, "cursor": _CURSOR}, ["platform"]),
                posts),
            self._tool(
                "socialscrape_get_post",
                "Get one post, video, tweet, pin or clip: text, author, time, stats and media links. "
                f"Platforms: {_titles('post')}. Pass its url; the platform is read from the link. Costs 1 credit; "
                "a lookup that fails is free.",
                _schema({"url": _text("Full URL of the post, video, tweet, pin or clip."),
                         "platform": _platform("post", _FROM_LINK), "id": _POST_ID}),
                post),
            self._tool(
                "socialscrape_list_comments",
                "List the comments on a post or video, one page per call: text, author, time and likes per "
                f"comment. Platforms: {_titles('comments')}. Pass the post's url; the platform is read from the "
                "link. When the answer has `has_more` true, pass its `cursor` back for the next page. Costs 1 "
                "credit per page; a failed call is free.",
                _schema({"url": _text("Full URL of the post or video."),
                         "platform": _platform("comments", _FROM_LINK), "id": _POST_ID, "cursor": _CURSOR}),
                comments),
            self._tool(
                "socialscrape_get_transcript",
                "Get the transcript of a video from its URL: the full text, its language and where it came from "
                f"(captions or AI transcription of the audio). {_transcript_help()} "
                f"Price: {endpoints()[TRANSCRIPT].credits}. A failed call is free.",
                _schema({
                    "url": _text("Full URL of the video."),
                    "language": _text("Preferred caption language as a BCP 47 code, for example en or pt-BR."),
                    "allow_long_transcription": {
                        "type": "boolean",
                        "description": "Allow AI transcription of audio longer than 2 minutes, at 5 credits per "
                                       "started minute of the whole video. Without it a longer video with no "
                                       "captions answers an error that names the cost, which is free."},
                    "include_segments": {
                        "type": "boolean",
                        "description": "Also return the timed segments (start, end, text). The full text is "
                                       "always returned."},
                }, ["url"]),
                transcript, no_segments),
            self._tool(
                "socialscrape_search",
                "Search one platform by keyword, one page per call. Posts and videos on most platforms; tracks, "
                f"albums and artists on Spotify and Apple Music; web results on Google. Platforms: "
                f"{_titles('search')}. When the answer has `has_more` true, pass its `cursor` back for the next "
                "page. Costs 1 credit per page; a failed call is free. For a narrower search (users, hashtags, "
                "filters) find the endpoint with socialscrape_list_endpoints and run it with "
                "socialscrape_call_endpoint.",
                _schema({"platform": _platform("search"), "query": _text("Search terms."), "cursor": _CURSOR},
                        ["platform", "query"]),
                search),
            Tool(
                name="socialscrape_list_endpoints",
                description=(
                    f"Look up what SocialScrape can fetch: {len(endpoints())} endpoints (profiles, posts, "
                    "comments, search, ad libraries, music, shops, transcripts, account usage). Free and "
                    "instant: it makes no request. With no arguments it lists every platform with its endpoint "
                    "ids; with `platform` that platform's endpoints and what each takes; with `endpoint` one "
                    "endpoint's full description and parameters; with `search` the endpoints that mention those "
                    "words. Use the id with socialscrape_call_endpoint."),
                parameters=_schema({
                    "platform": _text("Platform key, for example tiktok: lists its endpoints with what each takes."),
                    "endpoint": _text("Endpoint id, for example tiktok.profile.videos: its full description and "
                                      "every parameter."),
                    "search": _text("Words to look for in endpoint ids and descriptions, for example "
                                    "'ad library' or 'followers'."),
                }),
                function=list_endpoints),
            self._tool(
                "socialscrape_call_endpoint",
                f"Run any SocialScrape endpoint by its id ({len(endpoints())} of them, for example "
                "youtube.channel.videos or facebook.ad_library.search), with its parameters by name. Find the id "
                "and the parameters with socialscrape_list_endpoints first. Parameters are checked before the "
                "request, so a wrong name is answered at once and costs nothing. Most endpoints cost 1 credit "
                "per call; a failed call is free.",
                _schema({
                    "endpoint": _text("Endpoint id from socialscrape_list_endpoints, for example "
                                      "instagram.profile.reels."),
                    "params": {"type": "object",
                               "description": "The endpoint's parameters by name, for example {\"handle\": \"nasa\"}."},
                }, ["endpoint"]),
                call),
        ]
        return tools

    def to_dict(self) -> dict[str, Any]:
        """Serialize the toolset. The API key is stored as a reference to its environment variable."""
        return {
            "type": generate_qualified_class_name(type(self)),
            "data": {"api_key": self.api_key.to_dict(), "base_url": self.base_url,
                     "max_age_days": self.max_age_days, "trim": self.trim},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SocialScrapeToolset:
        """Deserialize the toolset."""
        inner = dict(data["data"])
        deserialize_secrets_inplace(inner, keys=["api_key"])
        return cls(**inner)

    def add(self, tool: Tool) -> None:
        """Add another tool next to the SocialScrape ones."""
        super().add(tool)
