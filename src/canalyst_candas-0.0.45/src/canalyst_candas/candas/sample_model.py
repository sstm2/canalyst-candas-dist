import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

from canalyst_candas.candas.model import Model
from canalyst_candas.settings import BulkDataKeys
from canalyst_candas.configuration.config import Config
from canalyst_candas.exceptions import BaseCandasException
from canalyst_candas.utils.requests import get_test_data_set, get_sample_drivers


class SampleModel(Model):
    """
    A model with sample data
    """

    def __init__(self, ticker):
        if ticker not in ["DIS US", "NFLX US", "TSLA US"]:
            raise BaseCandasException(
                "Sorry that ticker is not one of the sample tickers. Please use one of DIS US, NFLX US, TSLA US"
            )
        config_info = {
            "canalyst_api_key": "SampleCanalystKey",
            "s3_access_key_id": "S3AccessKeyId",
            "s3_secret_key": "S3SecretKey",
        }
        config = Config(config=config_info)

        super().__init__(ticker, config=config)

    def historical_data_frame(self):
        """
        Overrides historical_data_frame in Model to use local data
        """
        df = get_test_data_set(BulkDataKeys.HISTORICAL_DATA, self.ticker)
        return df

    def forward_data_frame(self):
        """
        Overrides forward_data_frame in Model to use local data
        """
        df = get_test_data_set(BulkDataKeys.FORECAST_DATA, self.ticker)
        return df

    def get_drivers(self):
        """
        Overrides get_drivers in Model to use local data
        """
        drivers = get_sample_drivers(self.ticker)

        drivers = get_sample_drivers(self.ticker)
        df_new = pd.json_normalize(drivers)
        df_driver_array = pd.DataFrame(df_new["time_series.names"])
        df_drivers = df_driver_array.explode("time_series.names")
        df = df_drivers.groupby("time_series.names").first().reset_index()
        df.columns = ["time_series_name"]

        self._model_drivers = df

    def apply_drivers(self):
        """
        Overrides apply_drivers in Model to use local data
        """
        self.get_drivers()
        df = self._model_drivers
        df = df.assign(is_driver=True)

        df2 = pd.merge(
            self._model_frame,
            df,
            how="outer",
            left_on="time_series_name",
            right_on="time_series_name",
        )
        df2 = df2[df2["ticker"].notna()]
        df2[["is_driver"]] = df2[["is_driver"]].fillna(value=False)
        self._model_frame: DataFrame = df2

        df_index = get_test_data_set(BulkDataKeys.NAME_INDEX, self.ticker)

        if df_index is not None:
            df_index.columns = ["time_series_name", "name_index"]
        else:
            mrq = self._model_frame["MRFQ"].loc[0]
            df_index = self._model_frame.loc[self._model_frame["period_name"] == mrq][
                ["time_series_name"]
            ]
            df_index["name_index"] = np.arange(len(df_index))

        self._model_frame = pd.merge(
            self._model_frame,
            df_index,
            how="outer",
            left_on="time_series_name",
            right_on="time_series_name",
        )

        self._model_frame = self._model_frame[self._model_frame["period_name"].notna()]
        return

    def get_company_info(self):
        """
        Returns sample (csin, model_version)
        """
        return ("12345678", "Q1-2020")
