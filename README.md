# socialscrape-haystack

[Haystack](https://haystack.deepset.ai) components and agent tools for [SocialScrape](https://socialscrape.dev/?utm_source=github&utm_medium=integration&utm_campaign=socialscrape-haystack), a social media data API: profiles, posts, comments, search results and video transcripts from YouTube, TikTok, Instagram, X, Facebook, LinkedIn, Reddit and more, as JSON in one response shape, with one API key.

```bash
pip install -U socialscrape-haystack
export SOCIALSCRAPE_API_KEY="ss_live_..."   # get a key: https://socialscrape.dev/sign-up
```

Python 3.11 or newer, Haystack 2.14 or newer. The package builds on the official [`socialscrape-sdk`](https://pypi.org/project/socialscrape-sdk/) client, so retries, timeouts and long transcription jobs are handled for you. Full guide: [socialscrape.dev/docs/haystack](https://socialscrape.dev/docs/haystack?utm_source=github&utm_medium=integration&utm_campaign=socialscrape-haystack).

## Components

| Class | What it does |
|---|---|
| `SocialScrapeTranscriptFetcher` | Video URLs in, one transcript `Document` per URL out. A URL that fails goes to `errors`, not an exception. |
| `SocialScrapeFetcher` | Runs any endpoint by id with its parameters and returns `Document`s: one per item of a list, one otherwise. |
| `SocialScrapeToolset` | Eight tools for a Haystack `Agent`: profile, posts, one post, comments, transcript, search, endpoint lookup and call-any-endpoint. |

The API key is a Haystack `Secret`. The default is `Secret.from_env_var("SOCIALSCRAPE_API_KEY")`, and every component serializes with `to_dict` and `from_dict` without writing the key out.

## Transcripts into a pipeline

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import ChatMessage
from socialscrape_haystack import SocialScrapeTranscriptFetcher

template = [ChatMessage.from_user(
    "Summarize the main points of these transcripts.\n"
    "{% for doc in documents %}\n[{{ doc.meta.url }}]\n{{ doc.content }}\n{% endfor %}"
)]

pipe = Pipeline()
pipe.add_component("fetcher", SocialScrapeTranscriptFetcher())
pipe.add_component("splitter", DocumentSplitter(split_by="word", split_length=300, split_overlap=30))
pipe.add_component("prompt", ChatPromptBuilder(template=template, required_variables=["documents"]))
pipe.add_component("llm", OpenAIChatGenerator(model="gpt-4o-mini"))
pipe.connect("fetcher.documents", "splitter.documents")
pipe.connect("splitter.documents", "prompt.documents")
pipe.connect("prompt.prompt", "llm.messages")

result = pipe.run({"fetcher": {"urls": [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.tiktok.com/@nasa/video/7665075736742530317",
]}})
print(result["llm"]["replies"][0].text)
```

For retrieval, put an embedder and a `DocumentWriter` after the splitter instead of the prompt builder. Each chunk keeps the meta of its video (`url`, `platform`, `language`, `duration_seconds`).

A transcript Document has the transcript text as `content` and this `meta`: `url` and `source` (the video URL), `platform`, `video_id`, `language`, `duration_seconds`, `transcript_source` (`captions`, or `ai` when the audio was transcribed) and `endpoint`.

### Errors

`SocialScrapeTranscriptFetcher` does not fail a batch for one bad link. A URL that cannot be fetched (private video, unsupported link, no captions) is left out of `documents` and listed in `errors`:

```python
result = SocialScrapeTranscriptFetcher().run(urls=["https://www.youtube.com/watch?v=dQw4w9WgXcQ", "https://example.com/nope"])
result["errors"]   # [{"url": "https://example.com/nope", "error": "..."}]
```

A failed request costs no credits. Pass `raise_on_failure=True` to raise on the first failure instead. A video with no captions is transcribed from its audio up to 2 minutes; longer audio needs `allow_long_transcription=True`, which is charged per started minute (see Billing).

## Any endpoint as Documents

```python
from socialscrape_haystack import SocialScrapeFetcher

fetcher = SocialScrapeFetcher(max_pages=3)

# One Document per comment, up to three pages
comments = fetcher.run(
    endpoint="youtube.video.comments",
    params={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
)["documents"]

# One Document for a profile
profile = fetcher.run(endpoint="tiktok.profile", params={"handle": "nasa"})["documents"][0]
profile.content   # name, headline and bio
profile.meta      # {'platform': 'tiktok', 'handle': 'nasa', 'followers': ..., 'endpoint': 'tiktok.profile', ...}
```

You can also fix the endpoint and its parameters when you create the component and pass only what changes at run time: `SocialScrapeFetcher(endpoint="reddit.search", params={"sort": "top"}).run(params={"query": "mechanical keyboards"})`.

- A list endpoint gives one Document per item and is followed across pages up to `max_pages` (default 1; each page is one request). `max_items` stops early.
- Any other endpoint (a profile, one post, a transcript) gives one Document.
- The content is the item's text: a post's title and text, a comment's text, a transcript's text, a profile's name and bio. An object with no text of its own is kept whole as JSON.
- The item's other scalar fields, its counts and its author go to `meta`, flat, with `endpoint`, `platform` and the item's `url` (also as `source`).
- By default a refused request raises. With `raise_on_failure=False` it is returned in `errors`.

The endpoint ids and their parameters are in the [API reference](https://socialscrape.dev/docs/api?utm_source=github&utm_medium=integration&utm_campaign=socialscrape-haystack), and in code:

```python
from socialscrape_haystack import endpoints

endpoints()["tiktok.profile.videos"].signature()
```

Parameters are checked against the endpoint before the request is made, so a misspelt name is answered at once and costs nothing.

## Give an Agent the tools

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from socialscrape_haystack import SocialScrapeToolset

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=SocialScrapeToolset(),   # reads SOCIALSCRAPE_API_KEY
    system_prompt="You answer questions about social media accounts and videos. Use the tools; do not guess.",
)

result = agent.run(messages=[ChatMessage.from_user(
    "What did NASA post on YouTube lately, and what are people saying in the comments?"
)])
print(result["last_message"].text)
```

Any chat generator that supports tool calling works. The toolset has eight tools that share one client:

| Tool | What it does |
|---|---|
| `socialscrape_get_profile` | A profile, channel or artist: name, bio, links, verification, follower counts. |
| `socialscrape_list_posts` | The latest posts or videos of a profile, channel or page, one page per call. |
| `socialscrape_get_post` | One post, video, tweet, pin or clip from its URL. |
| `socialscrape_list_comments` | The comments on a post or video, one page per call. |
| `socialscrape_get_transcript` | The transcript of a video from its URL. |
| `socialscrape_search` | Keyword search on one platform, one page per call. |
| `socialscrape_list_endpoints` | What else exists: every endpoint id with its parameters and price. Makes no request, so it is free. |
| `socialscrape_call_endpoint` | Runs any endpoint by id: ad libraries, followers, hashtags, playlists, music, shops and the rest. |

A tool answers with a JSON string: `data` (the object, or a list page with `items`, `cursor` and `has_more`), `credits_charged` and `credits_remaining`. A request the API refuses is free and comes back as the tool's answer (`error: not_found: ...`), so the agent can read it and try something else. Tools that list things return a `cursor` to pass back for the next page.

Toolset options: `api_key`, `max_age_days` (accept a cached result up to this many days old, which costs 0 credits), `trim` (default `True`, a smaller response that saves context), `client` and `async_client` (a `socialscrape.SocialScrape` or `AsyncSocialScrape` you configured yourself) and `base_url`.

## Billing

Most requests cost 1 credit: a profile, a post, a page of a list or of search results, a transcript from captions. AI transcription of a video with no captions costs 1 credit up to 2 minutes of audio; longer audio is transcribed only when you pass `allow_long_transcription`, at 5 credits per started minute of the whole video. A failed request and a cached repeat cost 0. A new account starts with 100 free credits, no card needed. See [pricing](https://socialscrape.dev/pricing?utm_source=github&utm_medium=integration&utm_campaign=socialscrape-haystack).

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests run offline against sample responses from the API reference.

## License

MIT, the same license as [langchain-socialscrape](https://github.com/SocialScraper/langchain-socialscrape).
