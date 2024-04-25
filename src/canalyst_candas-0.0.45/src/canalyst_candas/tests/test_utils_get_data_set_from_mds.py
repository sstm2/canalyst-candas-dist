import datetime
from io import StringIO
import io
import sys
from unittest import mock
import unittest
import json
from unittest.mock import Mock, patch

import pandas
from requests.models import Response

from canalyst_candas.settings import BulkDataKeys
from canalyst_candas.utils.logger import LogFile
from canalyst_candas.utils.transformations import _get_mds_bulk_data_url
from canalyst_candas.utils.requests import get_data_set_from_mds


class MdsBulkDataUtilsTests(unittest.TestCase):
    """
    Test cases for the get_data_set_from_mds() method and associated methods
    """

    def setUp(self) -> None:
        self.mock_log = Mock(spec=LogFile)

        self.mock_get_request = patch(
            "canalyst_candas.utils.requests.get_request"
        ).start()

    def tearDown(self):
        sys.stdout = sys.__stdout__
        patch.stopall()

    def test_get_data_set_from_mds_csv_success(self):
        with patch("requests.models.Response") as response_mock:
            some_csv_file = (
                '"header1", "header2", "header3"\n "data1", "data2", "data3"'
            )
            get_csv_response = response_mock
            type(get_csv_response).content = mock.PropertyMock(  # type: ignore
                return_value=bytes(some_csv_file, "utf-8")
            )

            self.mock_get_request.return_value = get_csv_response

            result = get_data_set_from_mds(
                BulkDataKeys.HISTORICAL_DATA,
                "csv",
                "ABCDE12345",
                "Q1-2021.20",
                {},
                self.mock_log,
                "mds_host",
            )

            self.assertEqual(
                result.to_string(), pandas.read_csv(StringIO(some_csv_file)).to_string()
            )

    def test_get_data_set_from_mds_parquet_success(self):
        with patch("requests.models.Response") as response_mock:
            input_df = pandas.DataFrame.from_dict(
                {
                    "period_name": ["FY2020", "Q1-2021", "Q1-2021"],
                    "time_series_name": ["MO_RIS_REV", "MO_MA_EBITDA", "MO_RIS_REV"],
                    "period_start_date": [
                        datetime.date(2020, 1, 1),
                        datetime.date(2021, 1, 1),
                        datetime.date(2021, 1, 1),
                    ],
                    "period_end_date": [
                        datetime.date(2020, 12, 31),
                        datetime.date(2021, 3, 31),
                        datetime.date(2021, 3, 31),
                    ],
                    "value": ["", "0", "3.14"],
                }
            )
            expected_df = pandas.DataFrame.from_dict(
                {
                    "period_name": ["FY2020", "Q1-2021", "Q1-2021"],
                    "time_series_name": ["MO_RIS_REV", "MO_MA_EBITDA", "MO_RIS_REV"],
                    "period_start_date": ["2020-01-01", "2021-01-01", "2021-01-01"],
                    "period_end_date": ["2020-12-31", "2021-03-31", "2021-03-31"],
                    "value": [float("nan"), 0, 3.14],
                }
            )

            get_parquet_response = response_mock

            parquet_df = input_df.to_parquet()
            type(get_parquet_response).content = mock.PropertyMock(  # type: ignore
                return_value=parquet_df
            )

            self.mock_get_request.return_value = get_parquet_response

            result = get_data_set_from_mds(
                BulkDataKeys.HISTORICAL_DATA,
                "parquet",
                "ABCDE12345",
                "Q1-2021.20",
                {},
                self.mock_log,
                "mds_host",
            )

            self.assertEqual(result.to_string(), expected_df.to_string())

    def test_get_data_set_from_mds_failure_model_info_invalid_type(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output

        result = get_data_set_from_mds(
            BulkDataKeys.MODEL_INFO,
            "parquet",
            "ABCDE12345",
            "Q1-2021.20",
            {},
            self.mock_log,
            "mds_host",
        )

        sys.stdout = sys.__stdout__

        self.assertIsNone(result)
        self.assertEqual(
            "model-info is not available in the requested file type: parquet. It is available in the following type(s): ['csv'].\n",
            captured_output.getvalue(),
        )

    def test_get_data_set_from_mds_failure_fcast_invalid_type(self):
        captured_output = io.StringIO()
        sys.stdout = captured_output

        result = get_data_set_from_mds(
            BulkDataKeys.FORECAST_DATA,
            "pickle",
            "ABCDE12345",
            "Q1-2021.20",
            {},
            self.mock_log,
            "mds_host",
        )

        sys.stdout = sys.__stdout__

        self.assertIsNone(result)
        self.assertEqual(
            "forecast-data is not available in the requested file type: pickle. It is available in the following type(s): ['csv', 'parquet'].\n",
            captured_output.getvalue(),
        )

    def test_get_mds_bulk_data_url_csv(self):
        expected = "https://mds.canalyst.com/api/equity-model-series/WZ0R430135/equity-models/FY2021.22/bulk-data/forecast-data.csv"

        result = _get_mds_bulk_data_url(
            BulkDataKeys.FORECAST_DATA,
            "csv",
            "WZ0R430135",
            "FY2021.22",
            "https://mds.canalyst.com",
        )

        self.assertEqual(result, expected)

    def test_get_mds_bulk_data_url_parquet(self):
        expected = "https://mds.canalyst.com/api/equity-model-series/WZ0R430135/equity-models/FY2021.22/bulk-data/name-index.parquet"

        result = _get_mds_bulk_data_url(
            BulkDataKeys.NAME_INDEX,
            "parquet",
            "WZ0R430135",
            "FY2021.22",
            "https://mds.canalyst.com",
        )

        self.assertEqual(result, expected)
