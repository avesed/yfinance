from yfinance._http import HTTPError

from yfinance import utils
from yfinance.config import YfConfig
from yfinance.const import _BASE_URL_
from yfinance.data import YfData

_INSIGHTS_URL_ = f"{_BASE_URL_}/ws/insights/v2/finance/insights"

# Yahoo-native, algorithmically-derived insight sections exposed by this
# accessor. The premium upsell teasers (``upsell`` / ``upsellSearchDD``) and any
# third-party research-report metadata/bodies are intentionally left out; SEC
# filings are already available via ``Ticker.sec_filings``.
_INSIGHT_SECTIONS = (
    "instrumentInfo",   # technical outlooks (short/intermediate/long), key technicals, valuation
    "companySnapshot",  # innovativeness/hiring/sustainability/insider-sentiment/... scores vs sector
    "recommendation",   # provider target price + rating
    "events",           # technical events
    "sigDevs",          # significant developments
)


class Insights:

    def __init__(self, data: YfData, symbol: str):
        self._data = data
        self._symbol = symbol
        self._insights = None

    @property
    def insights(self) -> dict:
        if self._insights is None:
            self._insights = self._fetch()
        return self._insights

    def _fetch(self) -> dict:
        try:
            data = self._data.get_raw_json(_INSIGHTS_URL_, params={"symbol": self._symbol})
        except HTTPError as e:
            if not YfConfig.debug.hide_exceptions:
                raise
            utils.get_yf_logger().error(str(e) + e.response.text)
            return {}
        result = (data.get("finance") or {}).get("result") or {}
        return {k: result[k] for k in _INSIGHT_SECTIONS if k in result}
