"""Yahoo Finance Premium data scrapers.

These endpoints require a logged-in session with an active Yahoo Finance
Premium subscription. Authentication is handled transparently by
:class:`~yfinance.data.YfData` (cookie + crumb), so callers only need to set
login cookies once via :meth:`yfinance.Auth.set_login_cookies`.

Use :meth:`yfinance.Auth.subscription_tier` to confirm the account's access
level (``'gold'`` / ``'silver'`` / ``'bronze'`` / ``'free'`` / ``None``) before
relying on these methods returning data. The required tier is noted in each
method's docstring; rather than pre-checking the tier (which costs an extra
request), the endpoints are called directly and a typed
:class:`~yfinance.exceptions.YFNotLoggedInError` /
:class:`~yfinance.exceptions.YFNotSubscribedError` is raised when access is
denied (unless ``YfConfig.debug.hide_exceptions`` is set, in which case an
empty result is returned).

Every premium request is issued with ``allow_strategy_switch=False`` so a
401/403 from an under-privileged session is read directly instead of triggering
:class:`~yfinance.data.YfData`'s cookie-strategy toggle, which would clear the
session cookies and wipe the user's ``T``/``Y`` login cookies.
"""
from __future__ import annotations

import datetime

import pandas as pd

from yfinance import utils, const
from yfinance.config import YfConfig
from yfinance.data import YfData
from yfinance.exceptions import YFNotLoggedInError, YFNotSubscribedError

# Earliest period accepted by Yahoo for premium timeseries (matches the
# yahooquery default; Yahoo clamps to the deepest available history).
_PREMIUM_TS_START = datetime.datetime(1985, 9, 24, tzinfo=datetime.timezone.utc)


class Premium:
    """Accessor for Yahoo Finance Premium data for a single symbol.

    Exposed via ``Ticker.premium``. Each table/dict is lazily fetched on first
    access and cached on the instance.

    Args:
        data (YfData): Shared data client (carries the session cookie + crumb).
        symbol (str): Yahoo Finance symbol, e.g. ``"AAPL"``.
    """

    def __init__(self, data: YfData, symbol: str):
        self._data = data
        self._symbol = symbol

        # Premium fundamentals timeseries caches, keyed by frequency.
        self._income_time_series: dict[str, pd.DataFrame] = {}
        self._balance_sheet_time_series: dict[str, pd.DataFrame] = {}
        self._cash_flow_time_series: dict[str, pd.DataFrame] = {}

        self._valuation_measures: pd.DataFrame | None = None
        self._corporate_events: pd.DataFrame | None = None

        self._fair_value: dict | None = None
        self._company_360: dict | None = None
        self._technical_insights: dict | None = None
        self._technical_events: dict | None = None
        self._visitor_trends: dict | None = None
        self._fair_value_history: dict | None = None

        # Market-level research listings (filtered to this symbol).
        self._research_reports: pd.DataFrame | None = None
        self._trade_ideas: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Premium fundamentals (income / balance-sheet / cash-flow)
    # ------------------------------------------------------------------

    @property
    def income_stmt(self) -> pd.DataFrame:
        """Annual income statement (premium, deeper history)."""
        return self._get_income_time_series("yearly")

    @property
    def quarterly_income_stmt(self) -> pd.DataFrame:
        """Quarterly income statement (premium, deeper history)."""
        return self._get_income_time_series("quarterly")

    @property
    def balance_sheet(self) -> pd.DataFrame:
        """Annual balance sheet (premium, deeper history)."""
        return self._get_balance_sheet_time_series("yearly")

    @property
    def quarterly_balance_sheet(self) -> pd.DataFrame:
        """Quarterly balance sheet (premium, deeper history)."""
        return self._get_balance_sheet_time_series("quarterly")

    @property
    def cash_flow(self) -> pd.DataFrame:
        """Annual cash-flow statement (premium, deeper history)."""
        return self._get_cash_flow_time_series("yearly")

    @property
    def quarterly_cash_flow(self) -> pd.DataFrame:
        """Quarterly cash-flow statement (premium, deeper history)."""
        return self._get_cash_flow_time_series("quarterly")

    @property
    def valuation_measures(self) -> pd.DataFrame:
        """Historical valuation measures (PE, PS, PB, EV/EBITDA, market cap, ...).

        Returns:
            pandas.DataFrame: Measures (rows) by date (columns); empty on failure.
        """
        if self._valuation_measures is None:
            keys = const.fundamentals_premium_keys["valuation"]
            self._valuation_measures = self._fetch_time_series("valuation-measures", "quarterly", keys)
        return self._valuation_measures

    @property
    def corporate_events(self) -> pd.DataFrame:
        """Significant corporate / development events feed.

        Unlike the financial statements, this feed returns rich event records
        (headline, description, significance, source date, ...) rather than a
        numeric timeseries, so it is reshaped into one row per event.

        Returns:
            pandas.DataFrame: One row per event; empty on failure or no data.
        """
        if self._corporate_events is None:
            self._corporate_events = self._fetch_corporate_events()
        return self._corporate_events

    @utils.log_indent_decorator
    def _fetch_corporate_events(self) -> pd.DataFrame:
        """Fetch and flatten the significant-developments (sigdev) event feed.

        Returns:
            pandas.DataFrame: One row per event with an ``eventType`` column,
            sorted newest-first by ``sourceDate``; empty on handled failure.
        """
        keys = const.fundamentals_premium_keys["corporate-events"]
        # The sigdev_* event keys are requested WITHOUT a timescale prefix.
        url = f"{const._FUNDAMENTALS_TIMESERIES_PREMIUM_URL_}/{self._symbol}?symbol={self._symbol}"
        url += "&type=" + ",".join(keys)
        end = pd.Timestamp.now('UTC').ceil("D")
        url += f"&period1={int(_PREMIUM_TS_START.timestamp())}&period2={int(end.timestamp())}"

        try:
            response = self._premium_get(url)
            json_data = response.json()
            data_raw = json_data["timeseries"]["result"]
            rows = []
            for entry in data_raw:
                for key, events in entry.items():
                    if not key.startswith("sigdev") or not isinstance(events, list):
                        continue
                    event_type = key.replace("sigdev_", "")
                    for ev in events:
                        if isinstance(ev, dict):
                            rows.append({"eventType": event_type, **ev})
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            if "sourceDate" in df.columns:
                df = df.sort_values("sourceDate", ascending=False).reset_index(drop=True)
            return df
        except Exception as e:
            if not YfConfig.debug.hide_exceptions:
                raise
            utils.get_yf_logger().error(f"{self._symbol}: Failed to fetch premium corporate events for reason: {e}")
        return pd.DataFrame()

    def _get_income_time_series(self, freq: str) -> pd.DataFrame:
        res = self._income_time_series
        if freq not in res:
            keys = const.fundamentals_keys["financials"]
            res[freq] = self._fetch_time_series("income", freq, keys)
        return res[freq]

    def _get_balance_sheet_time_series(self, freq: str) -> pd.DataFrame:
        res = self._balance_sheet_time_series
        if freq not in res:
            keys = const.fundamentals_keys["balance-sheet"]
            res[freq] = self._fetch_time_series("balance-sheet", freq, keys)
        return res[freq]

    def _get_cash_flow_time_series(self, freq: str) -> pd.DataFrame:
        res = self._cash_flow_time_series
        if freq not in res:
            keys = const.fundamentals_keys["cash-flow"]
            res[freq] = self._fetch_time_series("cash-flow", freq, keys)
        return res[freq]

    @utils.log_indent_decorator
    def _fetch_time_series(self, name: str, timescale: str, keys: list) -> pd.DataFrame:
        """Fetch and reshape a premium fundamentals-timeseries table.

        Mirrors the free fundamentals reshape logic; the only difference is the
        premium URL (which yields a deeper history).

        Args:
            name (str): Logical table name (for log messages only).
            timescale (str): One of ``"yearly"`` / ``"quarterly"`` / ``"trailing"``.
            keys (list): Field names (without the timescale prefix).

        Returns:
            pandas.DataFrame: Reshaped table, or an empty frame on handled failure.
        """
        try:
            statement = self._get_premium_time_series(timescale, keys)
            if statement is not None:
                return statement
        except Exception as e:
            if not YfConfig.debug.hide_exceptions:
                raise
            utils.get_yf_logger().error(f"{self._symbol}: Failed to create premium {name} table for reason: {e}")
        return pd.DataFrame()

    def _get_premium_time_series(self, timescale: str, keys: list) -> pd.DataFrame:
        timescale_translation = {"yearly": "annual", "quarterly": "quarterly", "trailing": "trailing"}
        timescale = timescale_translation[timescale]

        # Construct premium URL (differs from free endpoint only by '/premium/').
        ts_url_base = f"{const._FUNDAMENTALS_TIMESERIES_PREMIUM_URL_}/{self._symbol}?symbol={self._symbol}"
        url = ts_url_base + "&type=" + ",".join([timescale + k for k in keys])
        end = pd.Timestamp.now('UTC').ceil("D")
        url += f"&period1={int(_PREMIUM_TS_START.timestamp())}&period2={int(end.timestamp())}"

        response = self._premium_get(url)
        json_data = response.json()
        data_raw = json_data["timeseries"]["result"]
        for d in data_raw:
            del d["meta"]

        timestamps = set()
        data_unpacked = {}
        for x in data_raw:
            for k in x.keys():
                if k == "timestamp":
                    timestamps.update(x[k])
                else:
                    data_unpacked[k] = x[k]
        timestamps = sorted(list(timestamps))
        dates = pd.to_datetime(timestamps, unit="s")
        df = pd.DataFrame(columns=dates, index=list(data_unpacked.keys()))
        for k, v in data_unpacked.items():
            df.loc[k] = {pd.Timestamp(x["asOfDate"]): x["reportedValue"]["raw"] for x in v}

        df.index = df.index.str.replace("^" + timescale, "", regex=True)

        # Ensure float type, not object
        for d in df.columns:
            df[d] = df[d].astype('float')

        # Reorder rows to match the canonical key order, newest column first.
        df = df.reindex([k for k in keys if k in df.index])
        df = df[sorted(df.columns, reverse=True)]

        if timescale == "trailing" and not df.empty:
            df = df.iloc[:, [0]]

        return df

    # ------------------------------------------------------------------
    # Premium dict endpoints
    # ------------------------------------------------------------------

    @property
    def fair_value(self) -> dict | None:
        """Yahoo Premium fair-value estimate snapshot (Value Analyzer, Silver+ tier).

        See :attr:`fair_value_history` for the historical drilldown series.

        Returns:
            dict | None: Fair-value payload, or ``None`` on handled failure.
        """
        if self._fair_value is None:
            self._fair_value = self._fetch_json(
                const._PREMIUM_FAIR_VALUE_URL_,
                {"symbols": self._symbol, "formatted": "false"},
                "fair_value",
            )
        return self._fair_value

    @property
    def company_360(self) -> dict | None:
        """Yahoo Premium Company 360 snapshot (multiple modules).

        Returns:
            dict | None: Company 360 payload, or ``None`` on handled failure.
        """
        if self._company_360 is None:
            self._company_360 = self._fetch_json(
                const._PREMIUM_COMPANY_360_URL_,
                {"symbol": self._symbol, "modules": const._PREMIUM_COMPANY_360_MODULES_},
                "company_360",
            )
        return self._company_360

    @property
    def technical_insights(self) -> dict | None:
        """Yahoo Premium Technical Insights.

        Returns:
            dict | None: Technical insights payload, or ``None`` on handled failure.
        """
        if self._technical_insights is None:
            self._technical_insights = self._fetch_json(
                const._PREMIUM_INSIGHTS_URL_,
                {"symbol": self._symbol},
                "technical_insights",
            )
        return self._technical_insights

    @utils.log_indent_decorator
    def _fetch_json(self, url: str, params: dict, label: str):
        """Fetch a premium JSON GET endpoint and return the data payload.

        Yahoo's ``{"finance": {"result": ...}}`` envelope is unwrapped so callers
        get the data directly; endpoints that don't use that envelope (e.g. the
        technical-events feed) are returned unchanged.

        Args:
            url (str): Endpoint URL.
            params (dict): Query parameters (crumb is added by ``YfData``).
            label (str): Short name used in error logging.

        Returns:
            The unwrapped data (dict/list), or ``None`` on handled failure.
        """
        try:
            response = self._premium_get(url, params=params)
            payload = response.json()
            if isinstance(payload, dict):
                finance = payload.get("finance")
                if isinstance(finance, dict) and finance.get("result") is not None:
                    return finance["result"]
            return payload
        except Exception as e:
            if not YfConfig.debug.hide_exceptions:
                raise
            utils.get_yf_logger().error(f"{self._symbol}: Failed to fetch premium {label} for reason: {e}")
            return None

    # ------------------------------------------------------------------
    # Low-level premium HTTP helpers
    #
    # Every premium fetch goes through these so the cookie-wipe landmine is
    # handled in exactly one place: allow_strategy_switch=False is mandatory
    # (a 4xx must NOT toggle the cookie strategy, which clears session cookies
    # and would wipe the user's T/Y login cookies), and 401/403 are mapped to
    # typed exceptions so a long-running caller can detect a dead/insufficient
    # session.
    # ------------------------------------------------------------------

    def _check_premium_status(self, response) -> None:
        """Raise a typed exception for a premium auth/subscription failure.

        Args:
            response: The HTTP response from a premium endpoint.

        Raises:
            YFNotLoggedInError: On HTTP 401 (session missing/expired).
            YFNotSubscribedError: On HTTP 402/403 (insufficient tier).
        """
        status = response.status_code
        if status == 401:
            raise YFNotLoggedInError(f"{self._symbol}: premium endpoint returned 401 (not logged in).")
        if status in (402, 403):
            raise YFNotSubscribedError(
                f"{self._symbol}: premium endpoint returned {status} (account not subscribed to required tier)."
            )

    def _premium_get(self, url: str, params: dict | None = None, *, research: bool = False):
        """Issue a premium GET, never toggling the cookie strategy.

        Args:
            url (str): Endpoint URL.
            params (dict | None): Query parameters (crumb is added by ``YfData``).
            research (bool): When True, send the Referer/Origin headers required
                by the research endpoints (overlays / drilldown).

        Returns:
            The validated HTTP response (status < 400).

        Raises:
            YFNotLoggedInError / YFNotSubscribedError: On 401 / 402 / 403.
        """
        headers = const._PREMIUM_RESEARCH_HEADERS_ if research else None
        response = self._data.get(url, params=params, allow_strategy_switch=False, headers=headers)
        self._check_premium_status(response)
        response.raise_for_status()
        return response

    def _premium_post(self, url: str, body: dict):
        """Issue a premium POST (research listing), never toggling the strategy.

        Args:
            url (str): Endpoint URL.
            body (dict): JSON request body.

        Returns:
            The validated HTTP response (status < 400).

        Raises:
            YFNotLoggedInError / YFNotSubscribedError: On 401 / 402 / 403.
        """
        response = self._data.post(
            url, body=body, allow_strategy_switch=False, headers=const._PREMIUM_RESEARCH_HEADERS_
        )
        self._check_premium_status(response)
        response.raise_for_status()
        return response

    # ------------------------------------------------------------------
    # Extra premium dict endpoints
    # ------------------------------------------------------------------

    @property
    def technical_events(self) -> dict | None:
        """Full Yahoo Premium technical-events feed (Gold tier).

        Distinct from :attr:`technical_insights`: returns detailed
        short/intermediate/long-horizon technical events plus support,
        resistance and stop-loss levels.

        Returns:
            dict | None: ``{total, technicalEvents: {...}, events: [...]}``
            payload, or ``None`` on handled failure.
        """
        if self._technical_events is None:
            self._technical_events = self._fetch_json(
                const._PREMIUM_TECHNICAL_EVENTS_URL_,
                {"symbol": self._symbol, "size": 50, "tradingHorizons": "short,intermediate,long"},
                "technical_events",
            )
        return self._technical_events

    @property
    def visitor_trends(self) -> dict | None:
        """Yahoo Premium page-views visitor-trend feed (Silver+ tier).

        Returns:
            dict | None: ``{timestamp: [...], normalizedValues: [...]}`` payload,
            or ``None`` on handled failure.
        """
        if self._visitor_trends is None:
            # This endpoint requires an explicit period window (symbol-only -> 400).
            end = int(pd.Timestamp.now("UTC").ceil("D").timestamp())
            start = end - 365 * 24 * 3600
            self._visitor_trends = self._fetch_json(
                const._PREMIUM_PAGE_VIEWS_URL_,
                {"symbol": self._symbol, "period1": start, "period2": end},
                "visitor_trends",
            )
        return self._visitor_trends

    @property
    def fair_value_history(self) -> dict | None:
        """Historical fair-value drilldown via the Value Analyzer (Silver+ tier).

        Alias: :attr:`value_analyzer_drilldown`. Unlike :attr:`fair_value`
        (a current snapshot), this returns the historical fair-value series.

        Returns:
            dict | None: ``{currentPrice, revenue, epsGrowth, peRatio, dps,
            priceBookRatio, symbol, valuationDescription, ...}`` payload, or
            ``None`` on handled failure.
        """
        if self._fair_value_history is None:
            end = int(pd.Timestamp.now("UTC").ceil("D").timestamp())
            start = int(_PREMIUM_TS_START.timestamp())
            self._fair_value_history = self._fetch_json(
                const._PREMIUM_VALUE_ANALYZER_DRILLDOWN_URL_,
                {"symbol": self._symbol, "formatted": "false", "start": start, "end": end},
                "fair_value_history",
            )
        return self._fair_value_history

    @property
    def value_analyzer_drilldown(self) -> dict | None:
        """Alias for :attr:`fair_value_history`."""
        return self.fair_value_history

    # ------------------------------------------------------------------
    # Market-level premium research listings (research reports / trade ideas)
    #
    # The listing comes from the premium 'visualization' POST endpoint; each row
    # carries an "ID" that feeds the per-id overlay GET endpoint for the full
    # body. Both require the research Referer/Origin headers.
    # ------------------------------------------------------------------

    @property
    def research_reports(self) -> pd.DataFrame:
        """Argus research reports for this symbol (Silver+ tier).

        The ``ID`` column feeds :meth:`research_report` for the full body.

        Returns:
            pandas.DataFrame: One row per report; empty on handled failure or no
            data.
        """
        if self._research_reports is None:
            self._research_reports = self._fetch_visualization(
                entity_id_type="argus_reports",
                sort_field="report_date",
                include_fields=["report_date", "report_type", "report_title", "ticker",
                                "pdf_url", "id", "investment_rating"],
                label="research_reports",
            )
        return self._research_reports

    def research_report(self, report_id: str) -> dict | None:
        """Fetch the full body of a single research report (Silver+ tier).

        Args:
            report_id (str): Report id from the ``ID`` column of
                :attr:`research_reports`.

        Returns:
            dict | None: ``{id, provider, title, author, date, sectors, symbols,
            frequency, type, body, snapshotUrl, pdfUrl}`` payload, or ``None`` on
            handled failure.
        """
        return self._fetch_overlay(
            const._PREMIUM_RESEARCH_REPORTS_OVERLAY_URL_,
            {"reportId": report_id},
            "researchReportsOverlay",
            "research_report",
        )

    @property
    def trade_ideas(self) -> pd.DataFrame:
        """Trading Central trade ideas (investment ideas) for this symbol (Silver+).

        The ``ID`` column feeds :meth:`trade_idea` for the full body.

        Returns:
            pandas.DataFrame: One row per trade idea; empty on handled failure or
            no data.
        """
        if self._trade_ideas is None:
            self._trade_ideas = self._fetch_visualization(
                entity_id_type="trade_idea",
                sort_field="startdatetime",
                include_fields=["startdatetime", "term", "ticker", "rating", "price_target",
                                "ror", "id", "trade_idea_title", "description"],
                label="trade_ideas",
            )
        return self._trade_ideas

    def trade_idea(self, idea_id: str) -> dict | None:
        """Fetch the full body of a single trade idea (Silver+ tier).

        Args:
            idea_id (str): Idea id from the ``ID`` column of :attr:`trade_ideas`.

        Returns:
            dict | None: ``{eventTypeName, eventTypeDescription, companyName,
            descriptions, date, prices, ticker, sector, priceChange}`` payload,
            or ``None`` on handled failure.
        """
        return self._fetch_overlay(
            const._PREMIUM_TRADE_IDEAS_OVERLAY_URL_,
            {"ideaId": idea_id},
            "tradeIdeasOverlay",
            "trade_idea",
        )

    @utils.log_indent_decorator
    def _fetch_visualization(self, entity_id_type: str, sort_field: str,
                             include_fields: list, label: str, size: int = 50) -> pd.DataFrame:
        """Fetch a premium 'visualization' listing and reshape it to a DataFrame.

        The response holds aligned ``columns``/``rows`` (a list of cell lists);
        this zips them into a DataFrame and filters to ``self._symbol`` (the
        listing is market-wide). The ``ticker`` column is matched
        case-insensitively.

        Args:
            entity_id_type (str): ``"argus_reports"`` or ``"trade_idea"``.
            sort_field (str): Field to sort by (DESC).
            include_fields (list): Columns to request.
            label (str): Short name for error logging.
            size (int): Page size.

        Returns:
            pandas.DataFrame: One row per record; empty on handled failure or no
            data.
        """
        body = {
            "sortType": "DESC",
            "sortField": sort_field,
            "offset": 0,
            "size": size,
            "entityIdType": entity_id_type,
            "includeFields": include_fields,
            "query": {"operator": "eq", "operands": ["ticker", self._symbol]},
        }
        try:
            response = self._premium_post(const._PREMIUM_VISUALIZATION_URL_, body)
            json_data = response.json()
            result = (json_data.get("finance") or {}).get("result") or json_data.get("result") or []
            if not result or not isinstance(result[0], dict):
                return pd.DataFrame()
            documents = result[0].get("documents")
            if not documents:
                return pd.DataFrame()
            doc = documents[0]
            columns = [(c.get("label") or c.get("id")) for c in doc.get("columns", [])]
            rows = doc.get("rows", [])
            if not columns or not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=columns)
            # The listing is market-wide and the ticker column holds a LIST of
            # covered tickers per row (e.g. ['AAPL'] or ['CDNL','AAPL',...] for a
            # multi-company report). Yahoo's server-side query already filters to
            # rows covering this symbol; defensively trim again client-side,
            # matching the 'ticker'/'tickers' column and testing list membership.
            sym = self._symbol.upper()
            ticker_col = next((c for c in df.columns if str(c).lower() in ("ticker", "tickers")), None)
            if ticker_col is not None:
                df = df[df[ticker_col].apply(
                    lambda cell: sym in [str(t).upper() for t in cell]
                    if isinstance(cell, (list, tuple))
                    else str(cell).upper() == sym
                )].reset_index(drop=True)
            return df
        except Exception as e:
            # Includes YFNotLoggedInError / YFNotSubscribedError from a 401/403:
            # re-raised when exceptions aren't hidden, else logged + empty frame.
            if not YfConfig.debug.hide_exceptions:
                raise
            utils.get_yf_logger().error(f"{self._symbol}: Failed to fetch premium {label} for reason: {e}")
            return pd.DataFrame()

    @utils.log_indent_decorator
    def _fetch_overlay(self, url: str, params: dict, top_level_key: str, label: str) -> dict | None:
        """Fetch a premium overlay (per-id body) and return ``result[0]``.

        The overlay response nests the body under a top-level key
        (``researchReportsOverlay`` / ``tradeIdeasOverlay``) -> ``result`` -> a
        single-element list.

        Args:
            url (str): Overlay endpoint URL.
            params (dict): ``{"reportId": ...}`` or ``{"ideaId": ...}``.
            top_level_key (str): The response's top-level wrapper key.
            label (str): Short name for error logging.

        Returns:
            dict | None: The body dict, or ``None`` on handled failure / no data.
        """
        try:
            response = self._premium_get(url, params=params, research=True)
            json_data = response.json()
            overlay = json_data.get(top_level_key) or {}
            result = overlay.get("result")
            if isinstance(result, list) and result:
                return result[0]
            if isinstance(result, dict):
                return result
            return None
        except Exception as e:
            # Includes YFNotLoggedInError / YFNotSubscribedError from a 401/403:
            # re-raised when exceptions aren't hidden, else logged + None.
            if not YfConfig.debug.hide_exceptions:
                raise
            utils.get_yf_logger().error(f"{self._symbol}: Failed to fetch premium {label} for reason: {e}")
            return None
