"""
Test module for GetTestData set
"""

import unittest

from canalyst_candas.exceptions import BaseCandasException
from canalyst_candas.utils.requests import get_test_data_set
from canalyst_candas.settings import BulkDataKeys


class TestGetTestDataSet(unittest.TestCase):
    """
    Tests that the test data set is retrieved correctly

    Tests the utils function get_test_data_set
    """

    def test_correct_csv_type(self):
        result = get_test_data_set(BulkDataKeys.HISTORICAL_DATA, "NFLX US")
        self.assertIsNotNone(result)

    def test_incorrect_csv_type(self):
        """
        Check that error is raised when csv type is incorrect
        """
        with self.assertRaises(BaseCandasException):
            get_test_data_set(csv_type=BulkDataKeys.MODEL_INFO, ticker="DIS US")
