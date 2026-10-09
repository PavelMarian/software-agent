import sys
from types import SimpleNamespace

from software_bench.harness.services.web_search import DirectWebSearch


def test_direct_search_normalizes_results_and_enforces_domains(monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def text(self, query, **kwargs):
            calls.append((query, kwargs))
            return [
                {"title": "Rejected", "href": "https://example.com/x", "body": "no"},
                {
                    "title": "OpenFOAM guide",
                    "href": "https://doc.openfoam.com/guide",
                    "body": "Boundary-condition reference",
                },
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    results = DirectWebSearch(timeout_seconds=7, fetch_pages=False).search(
        "oblique shock", max_results=2, allowed_domains=("openfoam.com",)
    )

    assert calls[0] == ("init", {"timeout": 7})
    assert "site:openfoam.com" in calls[1][0]
    assert results == [{
        "title": "OpenFOAM guide",
        "url": "https://doc.openfoam.com/guide",
        "snippet": "Boundary-condition reference",
    }]


def test_direct_search_attaches_complete_parsed_page_content(monkeypatch):
    class FakeDDGS:
        def __init__(self, **kwargs):
            pass

        def text(self, query, **kwargs):
            return [{
                "title": "Guide",
                "href": "https://docs.example.test/guide",
                "body": "short search snippet",
            }]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    search = DirectWebSearch()
    monkeypatch.setattr(
        search, "_fetch_page", lambda url, max_chars: ("complete parsed page", False)
    )

    results = search.search("solver syntax", max_results=1)

    assert results[0]["snippet"] == "short search snippet"
    assert results[0]["content"] == "complete parsed page"
    assert results[0]["content_chars"] == 20
    assert results[0]["content_truncated"] is False
