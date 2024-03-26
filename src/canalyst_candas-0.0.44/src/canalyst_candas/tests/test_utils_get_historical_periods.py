import unittest
import json
from unittest.mock import Mock, patch

from canalyst_candas.utils.logger import LogFile
from canalyst_candas.utils.requests import (
    get_company_info_from_ticker,
    get_csin_from_ticker,
    get_historical_periods,
    get_model_info,
)


class MdsGetHistoricalPeriodsTests(unittest.TestCase):
    """
    Tests for util methods that hit the {canalyst_candas.settings.PERIODS_URL} url using a ticker.
    """

    def setUp(self) -> None:
        # From https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/
        self.ex_json_str = """
        {
            "count": 49,
            "next": null,
            "previous": null,
            "results": [
                {
                    "name": "FY2021",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2020-10-01",
                    "end_date": "2021-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2021/"
                },
                {
                    "name": "Q4-2021",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2021-07-01",
                    "end_date": "2021-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2021/"
                },
                {
                    "name": "Q3-2021",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2021-04-01",
                    "end_date": "2021-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2021/"
                },
                {
                    "name": "Q2-2021",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2021-01-01",
                    "end_date": "2021-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2021/"
                },
                {
                    "name": "Q1-2021",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2020-10-01",
                    "end_date": "2020-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2021/"
                },
                {
                    "name": "FY2020",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2019-10-01",
                    "end_date": "2020-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2020/"
                },
                {
                    "name": "Q4-2020",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2020-07-01",
                    "end_date": "2020-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2020/"
                },
                {
                    "name": "Q3-2020",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2020-04-01",
                    "end_date": "2020-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2020/"
                },
                {
                    "name": "Q2-2020",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2020-01-01",
                    "end_date": "2020-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2020/"
                },
                {
                    "name": "Q1-2020",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2019-10-01",
                    "end_date": "2019-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2020/"
                },
                {
                    "name": "FY2019",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2018-10-01",
                    "end_date": "2019-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2019/"
                },
                {
                    "name": "Q4-2019",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2019-07-01",
                    "end_date": "2019-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2019/"
                },
                {
                    "name": "Q3-2019",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2019-04-01",
                    "end_date": "2019-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2019/"
                },
                {
                    "name": "Q2-2019",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2019-01-01",
                    "end_date": "2019-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2019/"
                },
                {
                    "name": "Q1-2019",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2018-10-01",
                    "end_date": "2018-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2019/"
                },
                {
                    "name": "FY2018",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2017-10-01",
                    "end_date": "2018-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2018/"
                },
                {
                    "name": "Q4-2018",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2018-07-01",
                    "end_date": "2018-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2018/"
                },
                {
                    "name": "Q3-2018",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2018-04-01",
                    "end_date": "2018-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2018/"
                },
                {
                    "name": "Q2-2018",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2018-01-01",
                    "end_date": "2018-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2018/"
                },
                {
                    "name": "Q1-2018",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2017-10-01",
                    "end_date": "2017-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2018/"
                },
                {
                    "name": "FY2017",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2016-10-01",
                    "end_date": "2017-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2017/"
                },
                {
                    "name": "Q4-2017",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2017-07-01",
                    "end_date": "2017-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2017/"
                },
                {
                    "name": "Q3-2017",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2017-04-01",
                    "end_date": "2017-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2017/"
                },
                {
                    "name": "Q2-2017",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2017-01-01",
                    "end_date": "2017-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2017/"
                },
                {
                    "name": "Q1-2017",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2016-10-01",
                    "end_date": "2016-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2017/"
                },
                {
                    "name": "FY2016",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2015-10-01",
                    "end_date": "2016-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2016/"
                },
                {
                    "name": "Q4-2016",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2016-07-01",
                    "end_date": "2016-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2016/"
                },
                {
                    "name": "Q3-2016",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2016-04-01",
                    "end_date": "2016-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2016/"
                },
                {
                    "name": "Q2-2016",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2016-01-01",
                    "end_date": "2016-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2016/"
                },
                {
                    "name": "Q1-2016",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2015-10-01",
                    "end_date": "2015-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2016/"
                },
                {
                    "name": "FY2015",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2014-10-01",
                    "end_date": "2015-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2015/"
                },
                {
                    "name": "Q4-2015",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2015-07-01",
                    "end_date": "2015-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2015/"
                },
                {
                    "name": "Q3-2015",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2015-04-01",
                    "end_date": "2015-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2015/"
                },
                {
                    "name": "Q2-2015",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2015-01-01",
                    "end_date": "2015-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2015/"
                },
                {
                    "name": "Q1-2015",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2014-10-01",
                    "end_date": "2014-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2015/"
                },
                {
                    "name": "FY2014",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2013-10-01",
                    "end_date": "2014-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2014/"
                },
                {
                    "name": "Q4-2014",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2014-07-01",
                    "end_date": "2014-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2014/"
                },
                {
                    "name": "Q3-2014",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2014-04-01",
                    "end_date": "2014-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2014/"
                },
                {
                    "name": "Q2-2014",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2014-01-01",
                    "end_date": "2014-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2014/"
                },
                {
                    "name": "Q1-2014",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2013-10-01",
                    "end_date": "2013-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2014/"
                },
                {
                    "name": "FY2013",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2012-10-01",
                    "end_date": "2013-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2013/"
                },
                {
                    "name": "Q4-2013",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2013-07-01",
                    "end_date": "2013-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q4-2013/"
                },
                {
                    "name": "Q3-2013",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2013-04-01",
                    "end_date": "2013-06-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q3-2013/"
                },
                {
                    "name": "Q2-2013",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2013-01-01",
                    "end_date": "2013-03-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q2-2013/"
                },
                {
                    "name": "Q1-2013",
                    "period_duration_type": "fiscal_quarter",
                    "start_date": "2012-10-01",
                    "end_date": "2012-12-31",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/Q1-2013/"
                },
                {
                    "name": "FY2012",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2011-10-01",
                    "end_date": "2012-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2012/"
                },
                {
                    "name": "FY2011",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2010-10-01",
                    "end_date": "2011-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2011/"
                },
                {
                    "name": "FY2010",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2009-10-01",
                    "end_date": "2010-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2010/"
                },
                {
                    "name": "FY2009",
                    "period_duration_type": "fiscal_year",
                    "start_date": "2008-10-01",
                    "end_date": "2009-09-30",
                    "self": "https://mds.canalyst.com/api/equity-model-series/Y8S4N80139/equity-models/FY2021.26/historical-periods/FY2009/"
                }
            ]
        }
        """
        self.json_response = json.loads(self.ex_json_str)
        self.mock_log = Mock(spec=LogFile)
        self.mock_get_request_json_content = patch(
            "canalyst_candas.utils.requests.get_request_json_content"
        ).start()

    def tearDown(self):
        patch.stopall()

    def test_get_historical_periods(self):
        expected_result = ["FY2021", "Q4-2021", "Q3-2021", "Q2-2021", "Q1-2021"]
        self.mock_get_request_json_content.return_value = self.json_response

        result = get_historical_periods(
            "ticker",
            "model_version",
            {},
            "mds_host",
            self.mock_log,
            num_of_periods_to_fetch=len(
                expected_result
            ),  # only returns the 5 list items
        )

        self.assertEqual(result, expected_result)
