"""Unit tests for Fear & Greed and narrative rotation. Offline."""
import pytest

from tradingview_mcp.core.services import sentiment_index_service as svc


@pytest.fixture(autouse=True)
def _clear_cache():
    svc._cache.clear()
    yield
    svc._cache.clear()


def _patch_get(monkeypatch, payload_by_url_fragment, capture=None):
    def fake(url):
        if capture is not None:
            capture.append(url)
        for fragment, payload in payload_by_url_fragment.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"unexpected url {url}")
    monkeypatch.setattr(svc, "_get", fake)


def _fng(values):
    return {"data": [
        {"value": str(v), "value_classification": svc._classify(v),
         "timestamp": str(1700000000 + i)}
        for i, v in enumerate(values)
    ]}


def _category(name, mcap, change, cid=None):
    return {"name": name, "id": cid or name.lower().replace(" ", "-"),
            "market_cap": mcap, "market_cap_change_24h": change,
            "volume_24h": 1000.0, "top_3_coins_id": ["a", "b", "c"]}


# ── Fear & Greed ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (0, "Extreme Fear"), (24, "Extreme Fear"), (25, "Fear"), (44, "Fear"),
    (45, "Neutral"), (55, "Neutral"), (56, "Greed"), (75, "Greed"),
    (76, "Extreme Greed"), (100, "Extreme Greed"),
])
def test_classification_boundaries(value, expected):
    assert svc._classify(value) == expected


def test_current_reading_is_the_first_row(monkeypatch):
    _patch_get(monkeypatch, {"fng": _fng([32, 40, 55, 69])})
    out = svc.get_fear_greed(4)
    assert out["current"]["value"] == 32
    assert out["current"]["classification"] == "Fear"


def test_direction_of_travel(monkeypatch):
    # Newest first: 32 now, 69 a month ago -> moving toward fear.
    _patch_get(monkeypatch, {"fng": _fng([32, 50, 60, 69])})
    out = svc.get_fear_greed(4)
    assert out["change_over_window"] == 32 - 69
    assert out["direction"] == "toward fear"


def test_direction_toward_greed(monkeypatch):
    _patch_get(monkeypatch, {"fng": _fng([70, 50, 30, 20])})
    assert svc.get_fear_greed(4)["direction"] == "toward greed"


def test_direction_flat_within_tolerance(monkeypatch):
    _patch_get(monkeypatch, {"fng": _fng([50, 51, 49, 52])})
    assert svc.get_fear_greed(4)["direction"] == "flat"


def test_range_and_mean(monkeypatch):
    _patch_get(monkeypatch, {"fng": _fng([30, 50, 70])})
    out = svc.get_fear_greed(3)
    assert out["range"] == {"min": 30, "max": 70}
    assert out["mean"] == 50.0


def test_providers_own_label_is_preserved(monkeypatch):
    # If the provider disagrees with our boundaries, quote the provider —
    # the tool must not contradict the source it cites.
    payload = {"data": [{"value": "50", "value_classification": "Provider Label",
                         "timestamp": "1"}]}
    _patch_get(monkeypatch, {"fng": payload})
    assert svc.get_fear_greed(1)["current"]["classification"] == "Provider Label"


def test_single_reading_has_no_direction(monkeypatch):
    _patch_get(monkeypatch, {"fng": _fng([32])})
    out = svc.get_fear_greed(1)
    assert out["change_over_window"] is None
    assert out["direction"] is None


def test_unparseable_values_are_skipped(monkeypatch):
    payload = {"data": [{"value": "x", "timestamp": "1"},
                        {"value": "40", "timestamp": "2"}]}
    _patch_get(monkeypatch, {"fng": payload})
    assert svc.get_fear_greed(2)["current"]["value"] == 40


def test_empty_payload_is_no_data(monkeypatch):
    _patch_get(monkeypatch, {"fng": {"data": []}})
    assert svc.get_fear_greed()["error"]["code"] == "NO_DATA"


def test_upstream_failure_is_retryable(monkeypatch):
    _patch_get(monkeypatch, {"fng": RuntimeError("dns")})
    out = svc.get_fear_greed()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


@pytest.mark.parametrize("days", [0, 366, -5])
def test_history_days_is_bounded(days):
    assert svc.get_fear_greed(days)["error"]["code"] == "INVALID_PARAMETER"


# ── Narrative rotation ────────────────────────────────────────────────────────

def test_market_cap_floor_removes_dust(monkeypatch):
    # The reason the floor is on by default: unfiltered, a worthless category
    # up 1331% leads the list.
    _patch_get(monkeypatch, {"categories": [
        _category("Dust", 10_000.0, 1331.0),
        _category("Real", 5_000_000_000.0, 3.0),
    ]})
    out = svc.get_narrative_rotation(min_market_cap_usd=100_000_000.0)
    assert [r["name"] for r in out["rows"]] == ["Real"]
    assert out["below_market_cap_floor"] == 1


def test_floor_can_be_disabled(monkeypatch):
    _patch_get(monkeypatch, {"categories": [
        _category("Dust", 10_000.0, 1331.0),
        _category("Real", 5_000_000_000.0, 3.0),
    ]})
    out = svc.get_narrative_rotation(min_market_cap_usd=None)
    assert out["rows"][0]["name"] == "Dust"
    assert out["below_market_cap_floor"] == 0


def test_counts_reconcile_with_categories_seen(monkeypatch):
    _patch_get(monkeypatch, {"categories": [
        _category("Big", 5e9, 2.0),
        _category("Small", 1.0, 5.0),
        {"name": "NoChange", "id": "nc", "market_cap": 5e9,
         "market_cap_change_24h": None},
    ]})
    out = svc.get_narrative_rotation(min_market_cap_usd=1e8)
    assert out["ranked_categories"] == 1
    assert out["below_market_cap_floor"] == 1
    assert out["without_change_data"] == 1
    total = (out["ranked_categories"] + out["below_market_cap_floor"]
             + out["without_change_data"])
    assert total == out["categories_seen"], "every category must be accounted for"


def test_order_gainers_and_losers(monkeypatch):
    _patch_get(monkeypatch, {"categories": [
        _category("Up", 5e9, 10.0), _category("Down", 5e9, -10.0),
    ]})
    assert [r["name"] for r in svc.get_narrative_rotation(order="gainers")["rows"]] \
        == ["Up", "Down"]
    assert [r["name"] for r in svc.get_narrative_rotation(order="losers")["rows"]] \
        == ["Down", "Up"]


def test_advancing_declining_and_median(monkeypatch):
    _patch_get(monkeypatch, {"categories": [
        _category("A", 5e9, 5.0), _category("B", 5e9, -1.0),
        _category("C", 5e9, -3.0),
    ]})
    out = svc.get_narrative_rotation()
    assert out["advancing"] == 1 and out["declining"] == 2
    assert out["median_change_24h_pct"] == -1.0


def test_top_coins_are_included(monkeypatch):
    _patch_get(monkeypatch, {"categories": [_category("A", 5e9, 1.0)]})
    assert svc.get_narrative_rotation()["rows"][0]["top_coins"] == ["a", "b", "c"]


def test_everything_filtered_out_is_no_data(monkeypatch):
    _patch_get(monkeypatch, {"categories": [_category("Dust", 1.0, 5.0)]})
    out = svc.get_narrative_rotation(min_market_cap_usd=1e9)
    assert out["error"]["code"] == "NO_DATA"
    assert out["error"]["categories_seen"] == 1


def test_empty_category_payload(monkeypatch):
    _patch_get(monkeypatch, {"categories": []})
    assert svc.get_narrative_rotation()["error"]["code"] == "NO_DATA"


def test_narrative_upstream_failure(monkeypatch):
    _patch_get(monkeypatch, {"categories": RuntimeError("429")})
    out = svc.get_narrative_rotation()
    assert out["error"]["code"] == "UPSTREAM_ERROR"
    assert out["error"]["retryable"] is True


@pytest.mark.parametrize("kwargs", [
    {"limit": 0}, {"limit": 500}, {"order": "sideways"},
])
def test_narrative_rejects_bad_params(kwargs):
    assert svc.get_narrative_rotation(**kwargs)["error"]["code"] == "INVALID_PARAMETER"


# ── Caching ───────────────────────────────────────────────────────────────────

def test_repeat_reads_are_cached(monkeypatch):
    calls = []

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"data": [{"value": "50", "timestamp": "1"}]}'

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Resp()

    monkeypatch.setattr(svc.urllib.request, "urlopen", fake_urlopen)
    svc.get_fear_greed(1)
    svc.get_fear_greed(1)
    assert len(calls) == 1, "these indices move slowly; a repeat read must cache"
