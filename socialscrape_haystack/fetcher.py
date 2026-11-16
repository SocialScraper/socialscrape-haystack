"""SocialScrapeFetcher: run any SocialScrape endpoint and get Haystack Documents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from socialscrape import APIError, SocialScrapeConnectionError, SocialScrapeTimeoutError

from ._catalog import get_endpoint
from ._client import Session
from ._documents import items_of, to_document

FAILURES = (ValueError, APIError, SocialScrapeConnectionError, SocialScrapeTimeoutError)


@component
class SocialScrapeFetcher:
    """Runs one SocialScrape endpoint and returns the result as Documents.

    A list endpoint (a profile's posts, a video's comments, a search) gives one
    Document per item and is followed across pages up to `max_pages`. Any
    other endpoint (a profile, one post, a transcript) gives one Document.
    The item's text is the content: a post's title and text, a comment's
    text, a transcript's text, a profile's name and bio. Its other fields go
    to `meta`, flat, with `endpoint`, `platform` and the item's `url`
    (also as `source`).

    A request the API refuses raises, unless `raise_on_failure` is False: then
    nothing is returned for it and the message is in `errors`. A request that
    fails costs no credits.

    Usage example:
    ```python
    from socialscrape_haystack import SocialScrapeFetcher

    fetcher = SocialScrapeFetcher()
    result = fetcher.run(endpoint="youtube.video.comments",
                         params={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
    result["documents"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SOCIALSCRAPE_API_KEY"),  # noqa: B008
        endpoint: str | None = None,
        params: dict[str, Any] | None = None,
        max_pages: int = 1,
        max_items: int | None = None,
        max_age_days: float | None = None,
        raise_on_failure: bool = True,
        base_url: str | None = None,
    ) -> None:
        """
        Create a SocialScrapeFetcher.

        :param api_key: SocialScrape API key. Read from the `SOCIALSCRAPE_API_KEY` environment variable by default.
        :param endpoint: Default endpoint id, for example `tiktok.profile.videos`. The ids are in the API
            reference and in `socialscrape_haystack.endpoints()`. Can be given at run time instead.
        :param params: Default parameters of the endpoint by name, for example `{"handle": "nasa"}`.
            Parameters passed to `run` are merged over these.
        :param max_pages: Most pages to read from a list endpoint. Each page is one request and is charged like one.
        :param max_items: Stop after this many Documents.
        :param max_age_days: Accept a cached result up to this many days old (0 to 30), which costs 0 credits.
        :param raise_on_failure: Raise when a request fails. When False the failure is returned in `errors`.
        :param base_url: API base URL override.
        """
        if max_pages < 1:
            raise ValueError("max_pages must be at least 1")
        if endpoint is not None:
            get_endpoint(endpoint)
        self.api_key = api_key
        self.endpoint = endpoint
        self.params = dict(params or {})
        self.max_pages = max_pages
        self.max_items = max_items
        self.max_age_days = max_age_days
        self.raise_on_failure = raise_on_failure
        self.base_url = base_url
        self._session: Session | None = None

    def _get_session(self) -> Session:
        if self._session is None:
            self._session = Session(api_key=self.api_key.resolve_value(), base_url=self.base_url)
        return self._session

    def _plan(self, endpoint: str | None, params: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
        endpoint_id = endpoint or self.endpoint
        if not endpoint_id:
            raise ValueError("no endpoint: pass endpoint to run() or to the constructor, for example 'tiktok.profile'")
        ep = get_endpoint(endpoint_id)
        merged = {**self.params, **(params or {})}
        if self.max_age_days is not None and ep.common:
            merged.setdefault("max_age_days", self.max_age_days)
        return ep.id, merged

    def _full(self, docs: list[Document]) -> bool:
        return self.max_items is not None and len(docs) >= self.max_items

    @component.output_types(documents=list[Document], errors=list[dict[str, str]])
    def run(self, endpoint: str | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Run the endpoint and return its result as Documents.

        :param endpoint: Endpoint id. Falls back to the one given at construction.
        :param params: Endpoint parameters. Merged over the ones given at construction.
        :returns: `documents`, one per item, and `errors`, a list of `{"endpoint", "error"}` that is empty unless
            `raise_on_failure` is False and a request failed.
        """
        documents: list[Document] = []
        errors: list[dict[str, str]] = []
        endpoint_id, query = self._plan(endpoint, params)
        session = self._get_session()
        try:
            for _ in range(self.max_pages):
                data = session.call(endpoint_id, query).data
                items, cursor = items_of(data)
                for item in items:
                    if not self._full(documents):
                        documents.append(to_document(item, endpoint_id))
                if not cursor or self._full(documents) or "cursor" not in get_endpoint(endpoint_id).names:
                    break
                query = {**query, "cursor": cursor}
        except FAILURES as e:
            if self.raise_on_failure:
                raise
            errors.append({"endpoint": endpoint_id, "error": str(e)})
        return {"documents": documents, "errors": errors}

    @component.output_types(documents=list[Document], errors=list[dict[str, str]])
    async def run_async(self, endpoint: str | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Async version of `run`."""
        documents: list[Document] = []
        errors: list[dict[str, str]] = []
        endpoint_id, query = self._plan(endpoint, params)
        session = self._get_session()
        try:
            for _ in range(self.max_pages):
                data = (await session.acall(endpoint_id, query)).data
                items, cursor = items_of(data)
                for item in items:
                    if not self._full(documents):
                        documents.append(to_document(item, endpoint_id))
                if not cursor or self._full(documents) or "cursor" not in get_endpoint(endpoint_id).names:
                    break
                query = {**query, "cursor": cursor}
        except FAILURES as e:
            if self.raise_on_failure:
                raise
            errors.append({"endpoint": endpoint_id, "error": str(e)})
        return {"documents": documents, "errors": errors}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component. The API key is stored as a reference to its environment variable."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            endpoint=self.endpoint,
            params=self.params,
            max_pages=self.max_pages,
            max_items=self.max_items,
            max_age_days=self.max_age_days,
            raise_on_failure=self.raise_on_failure,
            base_url=self.base_url,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SocialScrapeFetcher:
        """Deserialize the component."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)
