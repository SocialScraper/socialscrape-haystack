import asyncio
import copy
import inspect
import json

import pytest
from haystack.tools import Tool, Toolset
from haystack.tools.errors import ToolInvocationError
from haystack.utils import Secret

from socialscrape_haystack import SocialScrapeToolset
from socialscrape_haystack.toolset import TOOL_NAMES


@pytest.fixture
def toolset(client, async_client):
    return SocialScrapeToolset(api_key=Secret.from_token("ss_test_key"), client=client, async_client=async_client)


def call(toolset, name, **args):
    return next(t for t in toolset if t.name == name).invoke(**args)


def test_eight_tools_with_llm_ready_schemas(toolset):
    assert isinstance(toolset, Toolset) and len(toolset) == 8
    assert tuple(t.name for t in toolset) == TOOL_NAMES
    for tool in toolset:
        assert isinstance(tool, Tool) and len(tool.description) > 60
        assert tool.parameters["type"] == "object"
        assert tool.tool_spec["name"] == tool.name
    profile = next(t for t in toolset if t.name == "socialscrape_get_profile")
    assert "tiktok" in profile.parameters["properties"]["platform"]["enum"]
    assert profile.parameters["required"] == ["platform"]
    for tool in toolset:
        for text in [tool.description] + [p.get("description", "") for p in tool.parameters["properties"].values()]:
            assert "\u2014" not in text


def test_profile(api, toolset):
    api.add("/v1/tiktok/profile", "tiktok.profile")
    out = json.loads(call(toolset, "socialscrape_get_profile", platform="tiktok", handle="nasa"))
    assert out["data"]["object"] == "profile" and out["credits_charged"] == 1
    assert api.params() == {"handle": "nasa", "trim": "true"}


def test_posts_and_cursor(api, toolset):
    api.add("/v1/youtube/channel/videos", "youtube.channel.videos")
    call(toolset, "socialscrape_list_posts", platform="youtube", handle="NASA", cursor="o30")
    assert api.params()["cursor"] == "o30"


def test_post_platform_from_url(api, toolset):
    api.add("/v1/youtube/video", "youtube.video")
    call(toolset, "socialscrape_get_post", url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert api.requests[0].url.path == "/v1/youtube/video"


def test_comments(api, toolset):
    api.add("/v1/youtube/video/comments", "youtube.video.comments")
    out = json.loads(call(toolset, "socialscrape_list_comments", url="https://www.youtube.com/watch?v=x"))
    assert out["data"]["object"] == "list"


def test_transcript_leaves_segments_out_unless_asked(api, toolset):
    api.add("/v1/transcript", "transcript")
    out = json.loads(call(toolset, "socialscrape_get_transcript", url="https://www.youtube.com/watch?v=x"))
    assert "segments" not in out["data"] and out["data"]["source"] == "captions"
    out = json.loads(call(toolset, "socialscrape_get_transcript", url="https://www.youtube.com/watch?v=x",
                          include_segments=True, allow_long_transcription=True, language="en"))
    assert len(out["data"]["segments"]) == 2
    assert api.params()["allow_long_transcription"] == "true" and api.params()["language"] == "en"


def test_search(api, toolset):
    api.add("/v1/reddit/search", "reddit.search")
    out = json.loads(call(toolset, "socialscrape_search", platform="reddit", query="keyboards"))
    assert len(out["data"]["items"]) == 2 and api.params() == {"query": "keyboards", "trim": "true"}
    answer = call(toolset, "socialscrape_search", platform="github", query="x")
    assert answer.startswith("error: no search for") and len(api.requests) == 1


def test_list_endpoints_makes_no_request(api, toolset):
    assert "tiktok.profile.videos" in call(toolset, "socialscrape_list_endpoints", platform="tiktok")
    assert "Parameters:" in call(toolset, "socialscrape_list_endpoints", endpoint="tiktok.profile")
    assert "ad_library" in call(toolset, "socialscrape_list_endpoints", search="ad library")
    assert call(toolset, "socialscrape_list_endpoints", platform="nope").startswith("error:")
    assert "endpoints, listed by platform key" in call(toolset, "socialscrape_list_endpoints")
    assert api.requests == []


def test_call_endpoint(api, toolset):
    api.add("/v1/github/user/repositories", "github.user.repositories")
    out = json.loads(call(toolset, "socialscrape_call_endpoint", endpoint="github.user.repositories",
                          params={"handle": "torvalds"}))
    assert out["data"]["object"] == "list"
    assert call(toolset, "socialscrape_call_endpoint", endpoint="github.user.repositories",
                params={"bogus": 1}).startswith("error: github.user.repositories does not take bogus")
    assert len(api.requests) == 1


def test_api_error_comes_back_as_text(api, toolset):
    api.fail("/v1/tiktok/profile", "not_found", "No TikTok account with that handle.", 404)
    answer = call(toolset, "socialscrape_get_profile", platform="tiktok", handle="nobody")
    assert answer.startswith("error: ") and "not_found" in answer


def test_cached_flag_and_options(api, client, sample):
    body = sample("tiktok.profile")
    body.update(cached=True, credits_charged=0)
    api.add("/v1/tiktok/profile", body)
    tools = SocialScrapeToolset(api_key=Secret.from_token("k"), client=client, max_age_days=5, trim=False)
    out = json.loads(call(tools, "socialscrape_get_profile", platform="tiktok", handle="nasa"))
    assert out["cached"] is True and out["credits_charged"] == 0
    assert "trim" not in api.params() and api.params()["max_age_days"] in ("5", "5.0")


HAS_ASYNC_TOOLS = "async_function" in inspect.signature(Tool.__init__).parameters


@pytest.mark.skipif(not HAS_ASYNC_TOOLS, reason="this Haystack has no Tool(async_function=...)")
def test_async(api, toolset):
    api.add("/v1/tiktok/profile", "tiktok.profile")
    tool = next(t for t in toolset if t.name == "socialscrape_get_profile")
    out = asyncio.run(tool.async_function(platform="tiktok", handle="nasa"))
    assert json.loads(out)["data"]["object"] == "profile"


def test_missing_key_raises(monkeypatch):
    tools = SocialScrapeToolset()
    with pytest.raises(ToolInvocationError, match="SOCIALSCRAPE_API_KEY"):
        call(tools, "socialscrape_get_profile", platform="tiktok", handle="nasa")


def test_serialization_round_trip(monkeypatch):
    toolset = SocialScrapeToolset(max_age_days=3, trim=False, base_url="https://example.test")
    data = toolset.to_dict()
    original = copy.deepcopy(data)
    assert data["type"] == "socialscrape_haystack.toolset.SocialScrapeToolset"
    assert data["data"]["api_key"]["env_vars"] == ["SOCIALSCRAPE_API_KEY"]
    json.dumps(data)
    monkeypatch.setenv("SOCIALSCRAPE_API_KEY", "ss_test_key")
    back = SocialScrapeToolset.from_dict(data)
    assert back.to_dict() == original and len(back) == 8 and back.max_age_days == 3 and back.trim is False


def test_works_inside_an_agent_and_serializes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator

    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=SocialScrapeToolset())
    data = agent.to_dict()
    assert data["init_parameters"]["tools"]["type"] == "socialscrape_haystack.toolset.SocialScrapeToolset"
