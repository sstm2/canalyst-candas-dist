"""
GetSampleDrivers tests module
"""
import unittest
from canalyst_candas.utils.requests import get_sample_drivers


class GetSampleDrivers(unittest.TestCase):
    """
    Tests for the get_sample_drivers function
    """

    ticker_options = {
        "disney": "DIS US",
        "netflix": "NFLX US",
        "tesla": "TSLA US",
    }

    def test_return_correct_sample_drivers(self):
        """
        Test that the drivers in the sample dataset are retrieved
        """
        result = get_sample_drivers("DIS US")
        self.assertIsNotNone(result)

    def test_incorrect_sample_driver(self):
        """
        Test that an incorrect ticker is caught
        """
        ticker = "MSFT US"
        with self.assertRaises(FileNotFoundError):
            get_sample_drivers(ticker)
