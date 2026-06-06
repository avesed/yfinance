"""Unit tests for yfinance.data.Auth (login state + subscription tier).

Pure unit tests: the OBI subscriptions endpoint response is mocked, so they run
offline with no Yahoo login cookies. Auth derives both login state and the
subscription tier from a single lightweight JSON call (no web-page scraping).
The call is made live each time (not cached), so the answer never goes stale.
"""
import unittest
from unittest import mock

from yfinance.data import Auth, _SUBSCRIPTIONS_URL

_GUID = "ABCDEFGHIJKLMNOPQRSTUV1234"


def _result(guid=_GUID, subscribed=False, features=None):
    view = [{"action": "ACTIVE"}] if subscribed else []
    return {"guid": guid, "subscriptionView": view, "premiumTierFeatures": features or {}}


def _auth(status=200, result=None):
    """Build an Auth whose subscriptions call returns the given status/result."""
    resp = mock.MagicMock()
    resp.status_code = status
    resp.json.return_value = {"result": result} if result is not None else {}
    auth = Auth()
    auth._data = mock.MagicMock()
    auth._data.get.return_value = resp
    return auth


class TestAuthLogin(unittest.TestCase):
    def test_logged_in_when_200_with_guid(self):
        auth = _auth(200, _result(features={"technicalEventsTeaser": True}))
        self.assertTrue(auth.check_login())
        self.assertEqual(auth.user, {"guid": _GUID})

    def test_not_logged_in_when_401(self):
        auth = _auth(401)
        self.assertFalse(auth.check_login())
        self.assertIsNone(auth.user)

    def test_not_logged_in_when_403(self):
        auth = _auth(403)
        self.assertFalse(auth.check_login())
        self.assertIsNone(auth.user)

    def test_not_logged_in_when_200_without_guid(self):
        auth = _auth(200, {"subscriptionView": []})  # 200 but no guid -> not logged in
        self.assertFalse(auth.check_login())
        self.assertIsNone(auth.user)

    def test_not_logged_in_on_transient_error(self):
        # A 429/5xx can't confirm login; report not-logged-in for that call.
        self.assertFalse(_auth(429).check_login())
        self.assertFalse(_auth(500).check_login())

    def test_probe_does_not_switch_cookie_strategy(self):
        # The probe must read the raw status; a 401 is the expected
        # not-logged-in answer, not a cookie-strategy failure.
        auth = _auth(401)
        auth.check_login()
        auth._data.get.assert_called_once_with(
            _SUBSCRIPTIONS_URL, allow_strategy_switch=False)

    def test_check_login_is_live_not_cached(self):
        # Each call re-queries: a session that goes valid/invalid between calls
        # is reflected immediately rather than served from a stale cache.
        auth = _auth(401)
        self.assertFalse(auth.check_login())
        self.assertFalse(auth.check_login())
        self.assertEqual(auth._data.get.call_count, 2)


class TestAuthSubscriptionTier(unittest.TestCase):
    def test_free_when_logged_in_no_subscription(self):
        auth = _auth(200, _result(features={"technicalEventsTeaser": True}))
        self.assertEqual(auth.subscription_tier(), "free")

    def test_bronze(self):
        auth = _auth(200, _result(subscribed=True,
                                  features={"adLite": True, "unlimitedPriceAlerts": True}))
        self.assertEqual(auth.subscription_tier(), "bronze")

    def test_silver(self):
        auth = _auth(200, _result(subscribed=True,
                                  features={"researchReports": True, "fairValue": True}))
        self.assertEqual(auth.subscription_tier(), "silver")

    def test_gold(self):
        auth = _auth(200, _result(subscribed=True,
                                  features={"researchReports": True,
                                            "premiumScreeners": True, "workspace": True}))
        self.assertEqual(auth.subscription_tier(), "gold")

    def test_premium_when_subscribed_with_unknown_features(self):
        # Subscribed (non-empty subscriptionView) but no recognized feature
        # signature -> generic "premium".
        auth = _auth(200, _result(subscribed=True, features={"someFutureFlag": True}))
        self.assertEqual(auth.subscription_tier(), "premium")

    def test_none_when_not_logged_in(self):
        self.assertIsNone(_auth(401).subscription_tier())


if __name__ == "__main__":
    unittest.main()
