import unittest
import json
from unittest.mock import Mock, patch

from canalyst_candas.utils.logger import LogFile
from canalyst_candas.utils.requests import (
    get_company_info_from_ticker,
    get_csin_from_ticker,
    get_model_info,
)


class MdsGetInfoFromTickerTests(unittest.TestCase):
    """
    Tests for util methods that hit the {canalyst_candas.settings.CSIN_URL} url using a ticker.

    - get_company_info_from_ticker()
    - get_csin_from_ticker()
    - get_model_info()
    """

    def setUp(self) -> None:
        # From https://mds.canalyst.com/api/equity-model-series/?company_ticker_bloomberg=AAPL+US
        # Note the format of the ticker '{ticker}+{country code}' in the URL
        self.ex_json_str = """
        {
            "count": 1,
            "next": null,
            "previous": null,
            "results": [
                {
                    "csin": "Y8S4N80139",
                    "company": {
                        "name": "Apple Inc.",
                        "self": "https://mds.canalyst.com/api/companies/Y8S4N8/"
                    },
                    "latest_equity_model": {
                        "equity_model_series": {
                            "csin": "Y8S4N80139",
                            "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/"
                        },
                        "model_version": {
                            "name": "FY2021.26",
                            "self": "https://mds.canalyst.com/api/model-versions/Y8S4N80139/periods/FY2021/revisions/26/"
                        },
                        "earnings_update_type": "regular",
                        "published_at": "2021-11-24T22:27:12.252465Z",
                        "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/"
                    },
                    "equity_models": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/"
                }
            ]
        }
        """
        self.json_response = json.loads(self.ex_json_str)
        self.csin = self.json_response["results"][0]["csin"]
        self.latest_mv = self.json_response["results"][0]["latest_equity_model"][
            "model_version"
        ]["name"]
        self.mock_log = Mock(spec=LogFile)
        self.mock_get_request_json_content = patch(
            "canalyst_candas.utils.requests.get_request_json_content"
        ).start()
        self.mock_get_historical_periods = patch(
            "canalyst_candas.utils.requests.get_historical_periods"
        ).start()

    def tearDown(self):
        patch.stopall()

    def test_get_company_info_from_ticker(self):
        self.mock_get_request_json_content.return_value = self.json_response

        result = get_company_info_from_ticker(
            "AAPL US",
            {},
            self.mock_log,
            "mds_host",
        )

        self.assertEqual(result, (self.csin, self.latest_mv))

    def test_get_csin_from_ticker(self):
        self.mock_get_request_json_content.return_value = self.json_response

        result = get_csin_from_ticker(
            "AAPL US",
            {},
            self.mock_log,
            "mds_host",
        )

        self.assertEqual(result, self.csin)

    def test_get_model_info(self):
        historical_periods = [
            "FY2021",
            "Q4-2021",
            "Q3-2021",
            "Q2-2021",
            "Q1-2021",
            "FY2020",
            "Q4-2020",
            "Q3-2020",
            "Q2-2020",
            "Q1-2020",
        ]
        self.mock_get_request_json_content.return_value = self.json_response
        self.mock_get_historical_periods.return_value = historical_periods
        mds_host = "mds_host"

        result = get_model_info(
            "AAPL US",
            {},
            self.mock_log,
            mds_host,
        )

        self.assertEqual(
            result,
            (
                (
                    f"{mds_host}/api/equity-model-series/{self.csin}/equity-models/"
                    f"{self.latest_mv}/historical-data-points/?page_size=500"
                ),
                {
                    "AAPL US": (
                        self.csin,
                        "Apple Inc.",
                        self.latest_mv,
                        historical_periods,
                        "regular",
                        "2021-11-24T22:27:12.252465Z",
                    )
                },
            ),
        )
