import json

import pytest

from socialscrape_haystack import endpoints
from socialscrape_haystack._catalog import ROLES, check_params, get_endpoint, route


def test_every_routed_endpoint_exists():
    for role, table in ROLES.items():
        for platform, endpoint_id in table.items():
            assert endpoint_id in endpoints(), f"{role}/{platform}: {endpoint_id}"


def test_check_params_rejects_before_any_request():
    ep = get_endpoint("tiktok.profile")
    with pytest.raises(ValueError, match="does not take"):
        check_params(ep, {"handle": "nasa", "bogus": 1})
    with pytest.raises(ValueError, match="needs one of"):
        check_params(ep, {})


def test_route_reads_the_platform_from_a_url():
    ep, ident = route("post", None, url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert ep.id == "youtube.video" and ident == {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}


def test_unknown_endpoint_names_neighbours():
    with pytest.raises(ValueError, match="Ids that start the same way"):
        get_endpoint("tiktok.nope")


def test_catalogue_is_json():
    json.dumps([e.id for e in endpoints().values()])
