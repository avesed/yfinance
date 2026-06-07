"""Tests for Yahoo Finance Premium support.

Two layers:

* Pure unit tests (CI-safe, no network, no credentials): synthetic fixtures are
  fed to mocked ``YfData`` methods to verify parsing logic, the cookie-safe
  request contract (``allow_strategy_switch=False`` + research headers), and the
  typed exceptions raised on 401/403.
* Env-gated integration tests: run only when ``YF_TEST_COOKIE_T`` and
  ``YF_TEST_COOKIE_Y`` are present in the environment.
"""
import contextlib
import json as _json
import os
import unittest
from unittest import mock

import pandas as pd

from yfinance import const
from yfinance.data import YfData
from yfinance.scrapers.premium import Premium
from yfinance.config import YfConfig
from yfinance.exceptions import YFNotLoggedInError, YFNotSubscribedError


@contextlib.contextmanager
def hide_exceptions(value=True):
    """Temporarily set ``YfConfig.debug.hide_exceptions`` (restoring after).

    ``NestedConfig`` doesn't support ``delattr`` so ``mock.patch.object`` cannot
    be used; set/restore the underlying value directly instead.
    """
    prev = YfConfig.debug.hide_exceptions
    YfConfig.debug.hide_exceptions = value
    try:
        yield
    finally:
        YfConfig.debug.hide_exceptions = prev


def _make_response(status_code=200, json_body=None, text=""):
    """Build a lightweight stand-in for an HTTP response object."""
    resp = mock.Mock()
    resp.status_code = status_code
    resp.text = text
    if json_body is None:
        resp.json.side_effect = ValueError("no json")
    else:
        resp.json.return_value = json_body
    # raise_for_status raises only on >= 400 (mirrors requests behaviour enough
    # for these tests; premium fetches check status before raise_for_status).
    if status_code >= 400:
        resp.raise_for_status.side_effect = RuntimeError(f"HTTP {status_code}")
    else:
        resp.raise_for_status = mock.Mock()
    return resp


def _premium_with_get(response):
    """Premium whose ``_data.get`` returns ``response``."""
    data = mock.Mock(spec=YfData)
    data.get.return_value = response
    return Premium(data, "AAPL")


def _premium_with_post(response):
    """Premium whose ``_data.post`` returns ``response``."""
    data = mock.Mock(spec=YfData)
    data.post.return_value = response
    return Premium(data, "AAPL")


def _timeseries_fixture(prefix, fields):
    """Build a synthetic premium fundamentals-timeseries payload.

    Args:
        prefix (str): timescale prefix, e.g. "annual" / "quarterly".
        fields (dict): {field_name: {asOfDate: raw_value}} (no prefix on names).

    Returns:
        dict: payload shaped like Yahoo's timeseries response.
    """
    result = []
    for name, values in fields.items():
        timestamps = [int(pd.Timestamp(d).timestamp()) for d in values.keys()]
        entries = [
            {"asOfDate": d, "reportedValue": {"raw": v}}
            for d, v in values.items()
        ]
        result.append({
            "meta": {"symbol": ["TEST"], "type": [prefix + name]},
            "timestamp": timestamps,
            prefix + name: entries,
        })
    return {"timeseries": {"result": result}}


class TestPremiumRequestContract(unittest.TestCase):
    """Every premium fetch must be cookie-safe (no strategy switch) and, for the
    research endpoints, carry the Referer/Origin headers."""

    def test_get_passes_allow_strategy_switch_false(self):
        payload = _timeseries_fixture("annual", {"TotalRevenue": {"2023-12-31": 1.0}})
        premium = _premium_with_get(_make_response(200, payload))
        _ = premium.income_stmt
        _, kwargs = premium._data.get.call_args
        self.assertIs(kwargs.get("allow_strategy_switch"), False)

    def test_dict_endpoint_passes_allow_strategy_switch_false(self):
        premium = _premium_with_get(_make_response(200, {"finance": {"result": []}}))
        _ = premium.technical_events
        _, kwargs = premium._data.get.call_args
        self.assertIs(kwargs.get("allow_strategy_switch"), False)

    def test_overlay_sends_research_headers_and_no_switch(self):
        payload = {"researchReportsOverlay": {"result": [{"id": "x", "body": "<p>hi</p>"}]}}
        premium = _premium_with_get(_make_response(200, payload))
        _ = premium.research_report("3676_Analyst Report_1700000000000")
        _, kwargs = premium._data.get.call_args
        self.assertIs(kwargs.get("allow_strategy_switch"), False)
        self.assertEqual(kwargs.get("headers"), const._PREMIUM_RESEARCH_HEADERS_)
        self.assertEqual(kwargs["headers"].get("Referer"), "https://finance.yahoo.com/research/")

    def test_visualization_post_sends_research_headers_and_no_switch(self):
        payload = {"finance": {"result": [{"documents": [{"columns": [], "rows": []}]}]}}
        premium = _premium_with_post(_make_response(200, payload))
        _ = premium.research_reports
        _, kwargs = premium._data.post.call_args
        self.assertIs(kwargs.get("allow_strategy_switch"), False)
        self.assertEqual(kwargs.get("headers"), const._PREMIUM_RESEARCH_HEADERS_)


class TestPremiumTypedExceptions(unittest.TestCase):
    """401/403 surface as typed exceptions (only when hide_exceptions is off)."""

    def test_401_raises_not_logged_in(self):
        premium = _premium_with_get(_make_response(401, text="not logged in"))
        with hide_exceptions(False), self.assertRaises(YFNotLoggedInError):
            _ = premium.income_stmt

    def test_403_raises_not_subscribed(self):
        premium = _premium_with_get(_make_response(403, text="forbidden"))
        with hide_exceptions(False), self.assertRaises(YFNotSubscribedError):
            _ = premium.technical_events

    def test_402_raises_not_subscribed(self):
        premium = _premium_with_get(_make_response(402, text="payment required"))
        with hide_exceptions(False), self.assertRaises(YFNotSubscribedError):
            _ = premium.fair_value

    def test_401_on_post_raises_not_logged_in(self):
        premium = _premium_with_post(_make_response(401, text="not logged in"))
        with hide_exceptions(False), self.assertRaises(YFNotLoggedInError):
            _ = premium.research_reports

    def test_401_returns_empty_df_when_hidden(self):
        premium = _premium_with_get(_make_response(401, text="not logged in"))
        with hide_exceptions(True):
            df = premium.income_stmt
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_403_returns_none_when_hidden(self):
        premium = _premium_with_get(_make_response(403, text="forbidden"))
        with hide_exceptions(True):
            self.assertIsNone(premium.technical_events)


class TestPremiumFundamentals(unittest.TestCase):
    """Premium fundamentals timeseries -> DataFrame reshaping."""

    def test_income_stmt_reshape(self):
        payload = _timeseries_fixture("annual", {
            "TotalRevenue": {"2022-12-31": 100.0, "2023-12-31": 120.0},
            "NetIncome": {"2022-12-31": 10.0, "2023-12-31": 12.0},
        })
        premium = _premium_with_get(_make_response(200, payload))
        df = premium.income_stmt
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("TotalRevenue", df.index)
        self.assertIn("NetIncome", df.index)
        self.assertEqual(df.columns[0], pd.Timestamp("2023-12-31"))
        self.assertEqual(df.loc["TotalRevenue"].iloc[0], 120.0)
        self.assertEqual(str(df.dtypes.iloc[0]), "float64")

    def test_quarterly_balance_sheet_reshape(self):
        payload = _timeseries_fixture("quarterly", {
            "TotalAssets": {"2023-09-30": 500.0, "2023-12-31": 550.0},
        })
        premium = _premium_with_get(_make_response(200, payload))
        df = premium.quarterly_balance_sheet
        self.assertIn("TotalAssets", df.index)
        self.assertEqual(df.loc["TotalAssets"].iloc[0], 550.0)

    def test_valuation_measures_reshape(self):
        payload = _timeseries_fixture("quarterly", {
            "PeRatio": {"2023-09-30": 25.0, "2023-12-31": 28.0},
            "MarketCap": {"2023-09-30": 1e12, "2023-12-31": 1.1e12},
        })
        premium = _premium_with_get(_make_response(200, payload))
        df = premium.valuation_measures
        self.assertIn("PeRatio", df.index)
        self.assertIn("MarketCap", df.index)

    def test_fetch_failure_returns_empty_df_when_hidden(self):
        data = mock.Mock(spec=YfData)
        data.get.side_effect = RuntimeError("network down")
        premium = Premium(data, "AAPL")
        with hide_exceptions(True):
            df = premium.income_stmt
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


class TestPremiumDictEndpoints(unittest.TestCase):
    """Dict endpoints unwrap the {finance:{result}} envelope + cache."""

    def test_fair_value_unwraps_result(self):
        result = [{"symbol": "AAPL", "fairValue": 123.0}]
        premium = _premium_with_get(_make_response(200, {"finance": {"result": result}}))
        self.assertEqual(premium.fair_value, result)
        _ = premium.fair_value  # cached, no re-fetch
        self.assertEqual(premium._data.get.call_count, 1)

    def test_company_360_unwraps_result(self):
        result = [{"companySnapshot": {}}]
        premium = _premium_with_get(_make_response(200, {"finance": {"result": result}}))
        self.assertEqual(premium.company_360, result)

    def test_technical_insights_unwraps_result(self):
        result = {"instrumentInfo": {}}
        premium = _premium_with_get(_make_response(200, {"finance": {"result": result}}))
        self.assertEqual(premium.technical_insights, result)

    def test_technical_events_no_envelope_unchanged(self):
        # This endpoint has no finance/result envelope -> returned as-is.
        payload = {"total": 3, "technicalEvents": {"shortTerm": {}}, "events": []}
        premium = _premium_with_get(_make_response(200, payload))
        self.assertEqual(premium.technical_events, payload)

    def test_visitor_trends_unwraps_result(self):
        result = {"timestamp": [1, 2], "normalizedValues": [0.5, 0.9]}
        premium = _premium_with_get(_make_response(200, {"finance": {"result": result}}))
        self.assertEqual(premium.visitor_trends, result)

    def test_fair_value_history_unwraps_and_alias(self):
        result = {"currentPrice": 200.0, "peRatio": 30.0}
        premium = _premium_with_get(_make_response(200, {"finance": {"result": result}}))
        self.assertEqual(premium.fair_value_history, result)
        # Alias returns the cached value, no extra fetch.
        self.assertEqual(premium.value_analyzer_drilldown, result)
        self.assertEqual(premium._data.get.call_count, 1)

    def test_dict_failure_returns_none_when_hidden(self):
        data = mock.Mock(spec=YfData)
        data.get.side_effect = RuntimeError("boom")
        premium = Premium(data, "AAPL")
        with hide_exceptions(True):
            self.assertIsNone(premium.fair_value)


class TestPremiumResearchListings(unittest.TestCase):
    """Visualization listings -> DataFrame, overlays -> dict body."""

    @staticmethod
    def _viz_payload(columns, rows):
        return {"finance": {"result": [{"documents": [{"columns": columns, "rows": rows}]}]}}

    def test_research_reports_reshape_and_filter(self):
        # Real shape: the column is "Tickers" (plural) and each cell is a LIST of
        # covered tickers (a single report can cover many companies).
        columns = [{"label": "Report Date"}, {"label": "Tickers"}, {"label": "ID"},
                   {"label": "PDF URL"}]
        rows = [
            ["2024-01-01", ["AAPL"], "3676_Analyst Report_1700000000000", "http://x/a.pdf"],
            ["2024-01-02", ["CDNL", "AAPL", "MSFT"], "5555_Analyst Report_1700000000002", "http://x/c.pdf"],
            ["2024-01-03", ["MSFT"], "9999_Analyst Report_1700000000001", "http://x/b.pdf"],
        ]
        premium = _premium_with_post(_make_response(200, self._viz_payload(columns, rows)))
        df = premium.research_reports
        self.assertIsInstance(df, pd.DataFrame)
        # Keep the two rows whose ticker list includes AAPL (incl. the
        # multi-company report); drop the MSFT-only row.
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["ID"]), ["3676_Analyst Report_1700000000000",
                                          "5555_Analyst Report_1700000000002"])

    def test_trade_ideas_reshape(self):
        columns = [{"label": "Ticker"}, {"label": "ID"}, {"label": "Rating"}]
        rows = [["AAPL", "tc_abc", "Bullish"]]
        premium = _premium_with_post(_make_response(200, self._viz_payload(columns, rows)))
        df = premium.trade_ideas
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["ID"], "tc_abc")

    def test_visualization_empty_returns_empty_df(self):
        premium = _premium_with_post(_make_response(200, self._viz_payload([], [])))
        self.assertTrue(premium.research_reports.empty)

    def test_research_report_body_extracts_result0(self):
        body = {"id": "3676_Analyst Report_1700000000000", "title": "AAPL report",
                "body": "<p>analysis</p>", "pdfUrl": "http://x/a.pdf"}
        payload = {"researchReportsOverlay": {"result": [body]}}
        premium = _premium_with_get(_make_response(200, payload))
        out = premium.research_report("3676_Analyst Report_1700000000000")
        self.assertEqual(out, body)
        self.assertEqual(out["pdfUrl"], "http://x/a.pdf")

    def test_trade_idea_body_extracts_result0(self):
        body = {"eventTypeName": "Double Bottom", "companyName": "Apple Inc.", "ticker": "AAPL"}
        payload = {"tradeIdeasOverlay": {"result": [body]}}
        premium = _premium_with_get(_make_response(200, payload))
        out = premium.trade_idea("tc_abc")
        self.assertEqual(out, body)

    def test_overlay_missing_result_returns_none(self):
        payload = {"researchReportsOverlay": {"result": []}}
        premium = _premium_with_get(_make_response(200, payload))
        self.assertIsNone(premium.research_report("nope"))

    def test_listing_failure_returns_empty_df_when_hidden(self):
        data = mock.Mock(spec=YfData)
        data.post.side_effect = RuntimeError("boom")
        premium = Premium(data, "AAPL")
        with hide_exceptions(True):
            self.assertTrue(premium.research_reports.empty)


class TestPremiumCorporateEvents(unittest.TestCase):
    """Corporate (sigdev) events flattening into one row per event."""

    def test_events_flattened_and_sorted(self):
        payload = {"timeseries": {"result": [
            {"meta": {}, "timestamp": [1], "sigdev_products": [
                {"id": 1, "headline": "New product", "sourceDate": "2023-01-01",
                 "significance": "3", "description": "d1", "parentTopics": "p"},
            ]},
            {"meta": {}, "timestamp": [2], "sigdev_financing": [
                {"id": 2, "headline": "Raised debt", "sourceDate": "2024-05-01",
                 "significance": "2", "description": "d2", "parentTopics": "p"},
            ]},
        ]}}
        premium = _premium_with_get(_make_response(200, payload))
        df = premium.corporate_events
        self.assertEqual(len(df), 2)
        self.assertIn("eventType", df.columns)
        self.assertIn("headline", df.columns)
        self.assertEqual(df.iloc[0]["eventType"], "financing")
        self.assertEqual(set(df["eventType"]), {"products", "financing"})

    def test_no_events_returns_empty(self):
        payload = {"timeseries": {"result": [{"meta": {}, "timestamp": []}]}}
        premium = _premium_with_get(_make_response(200, payload))
        self.assertTrue(premium.corporate_events.empty)

    def test_events_failure_returns_empty_when_hidden(self):
        data = mock.Mock(spec=YfData)
        data.get.side_effect = RuntimeError("boom")
        premium = Premium(data, "AAPL")
        with hide_exceptions(True):
            self.assertTrue(premium.corporate_events.empty)


class TestDataHeadersThreading(unittest.TestCase):
    """YfData.get/post/_make_request must thread the optional headers param."""

    def _yfdata(self):
        d = YfData.__new__(YfData)
        # Minimal stubs so _make_request runs without real network/cookies.
        d._session = mock.Mock()
        d._session.proxies = {}
        d._get_cookie_and_crumb = mock.Mock(return_value=("CRUMB", "basic"))
        return d

    def test_make_request_merges_headers(self):
        d = self._yfdata()
        resp = mock.Mock(status_code=200, url="http://x")
        d._session.get.return_value = resp
        d.get("http://x", headers={"Referer": "http://ref", "Origin": "http://orig"},
              allow_strategy_switch=False)
        _, kwargs = d._session.get.call_args
        self.assertEqual(kwargs["headers"]["Referer"], "http://ref")
        self.assertEqual(kwargs["headers"]["Origin"], "http://orig")

    def test_make_request_merges_headers_with_data_content_type(self):
        d = self._yfdata()
        resp = mock.Mock(status_code=200, url="http://x")
        d._session.post.return_value = resp
        d.post("http://x", data=_json.dumps({"a": 1}), headers={"Referer": "http://ref"})
        _, kwargs = d._session.post.call_args
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["headers"]["Referer"], "http://ref")

    def test_no_headers_omits_headers_key(self):
        d = self._yfdata()
        resp = mock.Mock(status_code=200, url="http://x")
        d._session.get.return_value = resp
        d.get("http://x")
        _, kwargs = d._session.get.call_args
        self.assertNotIn("headers", kwargs)


@unittest.skipUnless(
    os.environ.get("YF_TEST_COOKIE_T") and os.environ.get("YF_TEST_COOKIE_Y"),
    "premium cookies not provided",
)
class TestPremiumIntegration(unittest.TestCase):
    """Live integration tests (require real premium cookies via env vars)."""

    def setUp(self):
        cookie_t = os.environ.get("YF_TEST_COOKIE_T")
        cookie_y = os.environ.get("YF_TEST_COOKIE_Y")
        if not (cookie_t and cookie_y):
            self.skipTest("premium cookies not provided")
        import yfinance as yf
        self.auth = yf.Auth()
        self.auth.set_login_cookies(cookie_t, cookie_y)
        self.yf = yf

    def test_subscription_tier(self):
        tier = self.auth.subscription_tier()
        self.assertIn(tier, ("gold", "silver", "bronze", "premium", "free", None))

    def test_income_stmt(self):
        df = self.yf.Ticker("AAPL").premium.income_stmt
        self.assertIsInstance(df, pd.DataFrame)

    def test_research_reports_and_body(self):
        premium = self.yf.Ticker("AAPL").premium
        df = premium.research_reports
        self.assertIsInstance(df, pd.DataFrame)
        if not df.empty:
            id_col = next((c for c in df.columns if str(c).lower() == "id"), None)
            self.assertIsNotNone(id_col)
            body = premium.research_report(df.iloc[0][id_col])
            self.assertIsInstance(body, (dict, type(None)))

    def test_technical_events(self):
        out = self.yf.Ticker("AAPL").premium.technical_events
        self.assertIsInstance(out, (dict, type(None)))


if __name__ == "__main__":
    unittest.main()
