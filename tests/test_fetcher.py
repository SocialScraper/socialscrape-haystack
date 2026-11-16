import asyncio
import copy

import pytest
from haystack import Document, Pipeline
from haystack.utils import Secret
from socialscrape import APIError

from socialscrape_haystack import SocialScrapeFetcher


def make(session, **kwargs):
    fetcher = SocialScrapeFetcher(api_key=Secret.from_token("ss_test_key"), **kwargs)
    fetcher._session = session
    return fetcher


def test_list_endpoint_gives_one_document_per_item(api, session):
    api.add("/v1/youtube/video/comments", "youtube.video.comments")
    out = make(session).run(endpoint="youtube.video.comments", params={"url": "https://www.youtube.com/watch?v=x"})
    docs = out["documents"]
    assert out["errors"] == [] and len(docs) >= 1
    assert all(isinstance(d, Document) for d in docs)
    first = docs[0]
    assert first.content.startswith("So I just found")
    assert first.meta["endpoint"] == "youtube.video.comments" and first.meta["platform"] == "youtube"
    assert first.meta["source"] == first.meta["url"] and first.meta["author"] == "sample_user"
    assert first.id == "youtube:comment:UgzHlseKEhh-IqOwYux4AaABAg"
    assert api.params() == {"url": "https://www.youtube.com/watch?v=x"}


def test_single_object_gives_one_document(api, session):
    api.add("/v1/tiktok/profile", "tiktok.profile")
    docs = make(session).run(endpoint="tiktok.profile", params={"handle": "nasa"})["documents"]
    assert len(docs) == 1 and docs[0].meta["platform"] == "tiktok" and docs[0].meta["endpoint"] == "tiktok.profile"
    assert "followers" in docs[0].meta


def test_defaults_at_init_and_merge_at_run(api, session):
    api.add("/v1/reddit/search", "reddit.search")
    fetcher = make(session, endpoint="reddit.search", params={"sort": "top"}, max_age_days=3)
    fetcher.run(params={"query": "keyboards"})
    assert api.params() == {"sort": "top", "query": "keyboards", "max_age_days": "3"}


def test_pages_are_followed_up_to_max_pages(api, session, sample):
    page1 = sample("youtube.channel.videos")
    page1["data"].update(cursor="o30", has_more=True)
    last = sample("youtube.channel.videos")
    last["data"].update(cursor=None, has_more=False)
    api.add("/v1/youtube/channel/videos", page1)
    api.add("/v1/youtube/channel/videos", last)
    n = len(page1["data"]["items"])
    docs = make(session, max_pages=5).run(endpoint="youtube.channel.videos", params={"handle": "NASA"})["documents"]
    assert len(docs) == 2 * n and len(api.requests) == 2 and api.params(1)["cursor"] == "o30"


def test_max_pages_default_is_one_and_max_items_stops_early(api, session, sample):
    page = sample("youtube.channel.videos")
    page["data"].update(cursor="o30", has_more=True)
    api.add("/v1/youtube/channel/videos", page)
    assert len(make(session).run(endpoint="youtube.channel.videos", params={"handle": "NASA"})["documents"]) >= 1
    assert len(api.requests) == 1
    docs = make(session, max_pages=3, max_items=1).run(endpoint="youtube.channel.videos",
                                                       params={"handle": "NASA"})["documents"]
    assert len(docs) == 1 and len(api.requests) == 2


def test_no_endpoint_is_an_error(session):
    with pytest.raises(ValueError, match="no endpoint"):
        make(session).run()


def test_unknown_endpoint_fails_at_construction():
    with pytest.raises(ValueError, match="unknown endpoint id"):
        SocialScrapeFetcher(endpoint="tiktok.nope")


def test_bad_params_cost_no_request(api, session):
    with pytest.raises(ValueError, match="does not take"):
        make(session).run(endpoint="tiktok.profile", params={"handle": "nasa", "bogus": 1})
    assert api.requests == []


def test_api_error_raises_by_default_and_is_reported_when_asked(api, session):
    api.fail("/v1/tiktok/profile", "not_found", "No TikTok account with that handle.", 404)
    with pytest.raises(APIError):
        make(session).run(endpoint="tiktok.profile", params={"handle": "nobody"})
    out = make(session, raise_on_failure=False).run(endpoint="tiktok.profile", params={"handle": "nobody"})
    assert out["documents"] == [] and "not_found" in out["errors"][0]["error"]
    assert out["errors"][0]["endpoint"] == "tiktok.profile"


def test_async(api, async_session):
    api.add("/v1/tiktok/profile", "tiktok.profile")
    fetcher = SocialScrapeFetcher(api_key=Secret.from_token("ss_test_key"))
    fetcher._session = async_session
    out = asyncio.run(fetcher.run_async(endpoint="tiktok.profile", params={"handle": "nasa"}))
    assert len(out["documents"]) == 1


def test_serialization_round_trip(monkeypatch):
    fetcher = SocialScrapeFetcher(endpoint="tiktok.profile", params={"handle": "nasa"}, max_pages=2, max_items=5,
                                  max_age_days=7, raise_on_failure=False)
    data = fetcher.to_dict()
    original = copy.deepcopy(data)
    assert data["type"] == "socialscrape_haystack.fetcher.SocialScrapeFetcher"
    assert data["init_parameters"]["api_key"] == {"type": "env_var", "env_vars": ["SOCIALSCRAPE_API_KEY"],
                                                  "strict": True}
    monkeypatch.setenv("SOCIALSCRAPE_API_KEY", "ss_test_key")
    back = SocialScrapeFetcher.from_dict(data)
    assert back.to_dict() == original and back.api_key.resolve_value() == "ss_test_key"


def test_missing_key_raises(session):
    fetcher = SocialScrapeFetcher(endpoint="tiktok.profile", params={"handle": "nasa"})
    with pytest.raises(ValueError):
        fetcher.run()


def test_in_a_pipeline_and_pipeline_serializes(api, session, monkeypatch):
    monkeypatch.setenv("SOCIALSCRAPE_API_KEY", "ss_test_key")
    api.add("/v1/tiktok/profile", "tiktok.profile")
    pipe = Pipeline()
    fetcher = SocialScrapeFetcher()
    fetcher._session = session
    pipe.add_component("fetcher", fetcher)
    out = pipe.run({"fetcher": {"endpoint": "tiktok.profile", "params": {"handle": "nasa"}}})
    assert len(out["fetcher"]["documents"]) == 1
    assert Pipeline.from_dict(pipe.to_dict()).get_component("fetcher").api_key.to_dict()["type"] == "env_var"
