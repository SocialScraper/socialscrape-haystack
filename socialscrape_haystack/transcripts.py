"""SocialScrapeTranscriptsFetcher: video URLs in, transcript Documents out."""

from __future__ import annotations

from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

from ._client import Session
from ._documents import to_document
from .fetcher import FAILURES

ENDPOINT = "transcript"


@component
class SocialScrapeTranscriptsFetcher:
    """Fetches the transcript of each video URL and returns one Document per URL.

    Works with the video links the SocialScrape transcript endpoint accepts:
    YouTube, TikTok, Instagram and other platforms. The Document content is
    the transcript text. `meta` has `url`, `source` (the URL), `platform`,
    `video_id`, `language`, `duration_seconds` and `transcript_source`
    (`captions`, or `ai` when the audio was transcribed), plus whatever else
    the API returns for that video, such as a title.

    One bad URL never fails the batch: a URL that fails (private video,
    unsupported link, no captions and no permission for AI transcription) is
    skipped and reported in the `errors` output as `{"url", "error"}`. Set
    `raise_on_failure=True` to raise on the first failure instead. A failed
    request costs no credits.

    Usage example:
    ```python
    from socialscrape_haystack import SocialScrapeTranscriptsFetcher

    fetcher = SocialScrapeTranscriptsFetcher()
    result = fetcher.run(urls=["https://www.youtube.com/watch?v=dQw4w9WgXcQ"])
    result["documents"], result["errors"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SOCIALSCRAPE_API_KEY"),  # noqa: B008
        language: str | None = None,
        allow_long_transcription: bool = False,
        max_age_days: float | None = None,
        raise_on_failure: bool = False,
        base_url: str | None = None,
    ) -> None:
        """
        Create a SocialScrapeTranscriptsFetcher.

        :param api_key: SocialScrape API key. Read from the `SOCIALSCRAPE_API_KEY` environment variable by default.
        :param language: Preferred caption language as a BCP 47 code, for example `en` or `pt-BR`.
        :param allow_long_transcription: Allow AI transcription of audio longer than 2 minutes, which is charged
            per started minute of the whole video. Without it, a long video that has no captions is reported in
            `errors` (the error names the cost, and a failed request is free).
        :param max_age_days: Accept a cached transcript up to this many days old (0 to 30), which costs 0 credits.
        :param raise_on_failure: Raise on the first URL that fails instead of reporting it in `errors`.
        :param base_url: API base URL override.
        """
        self.api_key = api_key
        self.language = language
        self.allow_long_transcription = allow_long_transcription
        self.max_age_days = max_age_days
        self.raise_on_failure = raise_on_failure
        self.base_url = base_url
        self._session: Session | None = None

    def _get_session(self) -> Session:
        if self._session is None:
            self._session = Session(api_key=self.api_key.resolve_value(), base_url=self.base_url)
        return self._session

    def _params(self, url: str) -> dict[str, Any]:
        return {
            "url": url,
            "language": self.language,
            "allow_long_transcription": True if self.allow_long_transcription else None,
            "max_age_days": self.max_age_days,
            "trim": True,
        }

    @staticmethod
    def _document(data: Any, url: str) -> Document:
        doc = to_document(data, ENDPOINT)
        doc.meta.setdefault("url", url)
        doc.meta.setdefault("source", url)
        return doc

    @component.output_types(documents=list[Document], errors=list[dict[str, str]])
    def run(self, urls: list[str]) -> dict[str, Any]:
        """
        Fetch the transcript of each URL.

        :param urls: Video URLs.
        :returns: `documents`, one per URL that worked, in the order of `urls`, and `errors`, one
            `{"url", "error"}` per URL that did not.
        """
        documents: list[Document] = []
        errors: list[dict[str, str]] = []
        session = self._get_session()
        for url in urls:
            try:
                documents.append(self._document(session.call(ENDPOINT, self._params(url)).data, url))
            except FAILURES as e:
                if self.raise_on_failure:
                    raise
                errors.append({"url": url, "error": str(e)})
        return {"documents": documents, "errors": errors}

    @component.output_types(documents=list[Document], errors=list[dict[str, str]])
    async def run_async(self, urls: list[str]) -> dict[str, Any]:
        """Async version of `run`."""
        documents: list[Document] = []
        errors: list[dict[str, str]] = []
        session = self._get_session()
        for url in urls:
            try:
                documents.append(self._document((await session.acall(ENDPOINT, self._params(url))).data, url))
            except FAILURES as e:
                if self.raise_on_failure:
                    raise
                errors.append({"url": url, "error": str(e)})
        return {"documents": documents, "errors": errors}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component. The API key is stored as a reference to its environment variable."""
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            language=self.language,
            allow_long_transcription=self.allow_long_transcription,
            max_age_days=self.max_age_days,
            raise_on_failure=self.raise_on_failure,
            base_url=self.base_url,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SocialScrapeTranscriptsFetcher:
        """Deserialize the component."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)
