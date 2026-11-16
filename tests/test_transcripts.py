import asyncio
import copy

import pytest
from haystack.utils import Secret
from socialscrape import APIError

from socialscrape_haystack import SocialScrapeTranscriptsFetcher

YT = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def make(session, **kwargs):
    fetcher = SocialScrapeTranscriptsFetcher(api_key=Secret.from_token("ss_test_key"), **kwargs)
    fetcher._session = session
    return fetcher


def test_one_document_per_url(api, session):
    api.add("/v1/transcript", "transcript")
    out = make(session, language="en", max_age_days=2).run(urls=[YT])
    assert out["errors"] == []
    (doc,) = out["documents"]
    assert doc.content.startswith("Example transcript text.")
    assert doc.meta["platform"] == "youtube" and doc.meta["language"] == "en"
    assert doc.meta["duration_seconds"] == 213.0 and doc.meta["transcript_source"] == "captions"
    assert doc.meta["url"] == YT and doc.meta["source"] == YT and doc.meta["endpoint"] == "transcript"
    assert "segments" not in doc.meta
    assert api.params()["url"] == YT and api.params()["language"] == "en"
    assert "allow_long_transcription" not in api.params()


def test_allow_long_transcription_is_sent(api, session):
    api.add("/v1/transcript", "transcript")
    make(session, allow_long_transcription=True).run(urls=[YT])
    assert api.params()["allow_long_transcription"] == "true"


def test_one_failure_does_not_fail_the_batch(api, session):
    api.add("/v1/transcript", "transcript")
    api.fail("/v1/transcript", "not_found", "No transcript for that video.", 404)
    api.add("/v1/transcript", "transcript")
    # queue order: ok, fail, ok (the last queued answer repeats, so queue three)
    urls = [YT, "https://www.tiktok.com/@nasa/video/1", "https://www.youtube.com/watch?v=abc"]
    out = make(session).run(urls=urls)
    assert len(out["documents"]) + len(out["errors"]) == 3
    assert out["errors"] and out["errors"][0]["url"] in urls and "not_found" in out["errors"][0]["error"]


def test_raise_on_failure(api, session):
    api.fail("/v1/transcript", "not_found", "No transcript for that video.", 404)
    with pytest.raises(APIError):
        make(session, raise_on_failure=True).run(urls=[YT])


def test_empty_urls(session):
    assert make(session).run(urls=[]) == {"documents": [], "errors": []}


def test_async_batch(api, async_session):
    api.add("/v1/transcript", "transcript")
    fetcher = SocialScrapeTranscriptsFetcher(api_key=Secret.from_token("ss_test_key"))
    fetcher._session = async_session
    out = asyncio.run(fetcher.run_async(urls=[YT]))
    assert len(out["documents"]) == 1


def test_serialization_round_trip(monkeypatch):
    fetcher = SocialScrapeTranscriptsFetcher(language="pt-BR", allow_long_transcription=True, max_age_days=5,
                                            raise_on_failure=True, base_url="https://example.test")
    data = fetcher.to_dict()
    original = copy.deepcopy(data)
    assert data["type"] == "socialscrape_haystack.transcripts.SocialScrapeTranscriptsFetcher"
    monkeypatch.setenv("SOCIALSCRAPE_API_KEY", "ss_test_key")
    back = SocialScrapeTranscriptsFetcher.from_dict(data)
    assert back.to_dict() == original and back.language == "pt-BR" and back.allow_long_transcription


def test_token_secret_is_not_serializable():
    with pytest.raises(ValueError):
        SocialScrapeTranscriptsFetcher(api_key=Secret.from_token("ss_test_key")).to_dict()


def test_pipeline_into_splitter_keeps_meta(api, session, monkeypatch):
    from haystack import Pipeline
    from haystack.components.preprocessors import DocumentSplitter

    monkeypatch.setenv("SOCIALSCRAPE_API_KEY", "ss_test_key")
    api.add("/v1/transcript", "transcript")
    fetcher = SocialScrapeTranscriptsFetcher()
    fetcher._session = session
    pipe = Pipeline()
    pipe.add_component("fetcher", fetcher)
    pipe.add_component("splitter", DocumentSplitter(split_by="word", split_length=5, split_overlap=0))
    pipe.connect("fetcher.documents", "splitter.documents")
    docs = pipe.run({"fetcher": {"urls": [YT]}})["splitter"]["documents"]
    assert len(docs) > 1 and all(d.meta["url"] == YT and d.meta["language"] == "en" for d in docs)
