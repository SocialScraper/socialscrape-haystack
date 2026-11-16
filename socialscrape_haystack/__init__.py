"""Haystack integration for SocialScrape: social media data for pipelines and agents.

`SocialScrapeFetcher` runs any endpoint and returns Documents, `SocialScrapeTranscriptsFetcher` turns video URLs
into transcript Documents, and `SocialScrapeToolset` gives a Haystack Agent tools for profiles, posts,
comments, transcripts and search across platforms.
"""

from importlib import metadata

from ._catalog import Endpoint, endpoints
from .fetcher import SocialScrapeFetcher
from .toolset import SocialScrapeToolset
from .transcripts import SocialScrapeTranscriptsFetcher

# Newer Haystack only loads a pipeline YAML whose classes come from trusted modules (`haystack`,
# `haystack_integrations`). Whoever imports this package trusts it, so say so, or Pipeline.from_dict and
# Pipeline.loads would refuse the components here. Older Haystack has no allowlist.
try:
    from haystack.core.serialization import allow_deserialization_module

    allow_deserialization_module("socialscrape_haystack")
except ImportError:  # pragma: no cover
    pass

try:
    __version__ = metadata.version("socialscrape-haystack")
except metadata.PackageNotFoundError:
    __version__ = ""

__all__ = [
    "Endpoint",
    "SocialScrapeFetcher",
    "SocialScrapeToolset",
    "SocialScrapeTranscriptsFetcher",
    "__version__",
    "endpoints",
]
