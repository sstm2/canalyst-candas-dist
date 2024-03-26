import unittest
from canalyst_candas import settings

from canalyst_candas.utils.transformations import get_forecast_url


class BaseUtilsTests(unittest.TestCase):
    """
    General test cases for the utils file
    """

    def test_get_forecast_url(self):
        # below URL should actually be valid if you want to manually test
        expected_url = f"{settings.MDS_HOST}/api/equity-model-series/S1TQ5V0161/equity-models/Q3-2021.20/forecast-periods/"
        returned_url = get_forecast_url("S1TQ5V0161", "Q3-2021.20")

        assert expected_url == returned_url
