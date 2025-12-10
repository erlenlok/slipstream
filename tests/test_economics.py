
import pytest
from slipstream.strategies.brawler.economics import RequestPurse

class TestRequestPurse:
    def test_purse_initialization(self):
        purse = RequestPurse()
        assert purse.request_count == 0
        assert purse.cumulative_volume == 0.0

    def test_purse_deduction(self):
        purse = RequestPurse()
        purse.deduct_request()
        assert purse.request_count == 1
        purse.deduct_request()
        assert purse.request_count == 2

    def test_purse_income(self):
        purse = RequestPurse()
        purse.add_fill_credit(100.0)
        assert purse.cumulative_volume == 100.0
        purse.add_fill_credit(-50.0)  # Should handle absolute value usually? Or just adding volume.
        # The implementation uses abs(volume_usd) for credit
        assert purse.cumulative_volume == 150.0

    def test_purse_sync(self):
        purse = RequestPurse()
        purse.deduct_request()
        # Mock exchange says we actually did 5 requests and 1000 volume
        purse.sync(exchange_requests=5, exchange_volume=1000.0)
        
        assert purse.request_count == 5
        assert purse.cumulative_volume == 1000.0
