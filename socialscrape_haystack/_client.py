"""How the components and the toolset reach the API: one place that owns the SocialScrape client."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import metadata
from typing import Any

from socialscrape import AsyncSocialScrape, Response, SocialScrape
from socialscrape import __version__ as _sdk_version

from ._catalog import check_params, get_endpoint

try:
    _version = metadata.version("socialscrape-haystack")
except metadata.PackageNotFoundError:  # a source checkout that was never installed
    _version = "0"

USER_AGENT = f"socialscrape-haystack/{_version} socialscrape-python/{_sdk_version}"


class Session:
    """Makes the SocialScrape client on first use and keeps it for later calls.

    Pass `client` (and `async_client`) to use ones you configured yourself, for
    example with your own timeouts or `httpx` client. Otherwise the key is
    `api_key`, or SOCIALSCRAPE_API_KEY when it is left out.
    """

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None, client: SocialScrape | None = None,
                 async_client: AsyncSocialScrape | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client = client
        self._async_client = async_client

    def _options(self) -> dict[str, Any]:
        return {"api_key": self._api_key, "base_url": self._base_url, "default_headers": {"user-agent": USER_AGENT}}

    def client(self) -> SocialScrape:
        if self._client is None:
            self._client = SocialScrape(**self._options())
        return self._client

    def call(self, endpoint_id: str, params: Mapping[str, Any]) -> Response[Any]:
        """Run one endpoint. Parameters are checked first, so a bad call costs no request."""
        ep = get_endpoint(endpoint_id)
        return self.client().request("GET", ep.path, query=check_params(ep, params))

    async def acall(self, endpoint_id: str, params: Mapping[str, Any]) -> Response[Any]:
        ep = get_endpoint(endpoint_id)
        query = check_params(ep, params)
        if self._async_client is not None:
            return await self._async_client.request("GET", ep.path, query=query)
        # A client of our own lives for one call: it cannot outlast the event loop it was made in.
        async with AsyncSocialScrape(**self._options()) as client:
            return await client.request("GET", ep.path, query=query)
