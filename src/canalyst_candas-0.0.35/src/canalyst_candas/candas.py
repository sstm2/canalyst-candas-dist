from canalyst_candas.version import __version__
from pandas.core.frame import DataFrame
from canalyst_candas.datafunctions import DataFunctions
from canalyst_candas.exceptions import BaseCandasException, ScenarioException
from canalyst_candas.utils import (
    BRAND_CONFIG_DEFAULTS,
    Getter,
    LogFile,
    BulkDataKeys,
    SCENARIO_URL,
    create_drivers_dot,
    df_filter,
    filter_dataset,
    get_api_headers,
    get_company_info_from_ticker,
    get_data_set_from_mds,
    get_drivers_from_api,
    get_name_index_from_csv,
    get_forecast_url,
    get_forecast_url_data,
    get_model_info,
    get_request,
    get_request_json_content,
    get_sample_drivers,
    get_scenario_url_data,
    get_test_data_set,
    map_scenario_urls,
    save_guidance_csv,
    send_scenario,
    get_excel_model,
)
import __main__
from pyvis.network import Network
from functools import reduce
import networkx as nx
import plotly.express as px
from statistics import stdev
import os.path
import numpy as np
import datetime
import io

import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.font_manager as fm
import matplotlib as mpl

plt.style.use("fivethirtyeight")

from requests.models import Response
from typing import Any, Dict, Iterable, List, Optional, Union

import os
import string
import urllib3

from joblib import Parallel, delayed

import multiprocessing

num_cores = multiprocessing.cpu_count()

urllib3.disable_warnings()
from pathlib import Path

from canalyst_candas.configuration.config import Config
from canalyst_candas.settings import CONFIG

from canalyst_candas import settings


def help():
    """
    Canalyst Candas help function
    """
    print("Canalyst Candas help")
    print(
        "Please go to https://pypi.org/project/canalyst-candas/#description for installation help"
    )
    print("For support, please contact jed.gore@canalyst.com")
    print("For an API KEY please go to https://app.canalyst.com/u/settings/api-tokens")
    print("For an Excel model download:get_excel_model(ticker, config)")


class FuncMusic(object):
    """
    Helper class for on the fly function generation.

    Used in the ForecastFrame concept where we try to change one param
    across multiple tickers for re-fit in the scenario engine.

    """

    def apply_function(self, value, modifier, argument):
        """
        Apply a specified function to a value

        Parameters
        ----------
        value: int
        modifier: str
        {"add", "subtract", "divide", "multiply","value"}
        argument: int
        """
        self.value = value
        self.modifier = modifier
        self.argument = argument
        method_name = "func_" + str(self.modifier)
        method = getattr(
            self,
            method_name,
            lambda: "Invalid function: use add, subtract, divide, or multiply",
        )
        return method()

    def func_add(self):
        return float(self.value) + float(self.argument)

    def func_divide(self):
        return float(self.value) / float(self.argument)

    def func_multiply(self):
        return float(self.value) * float(self.argument)

    def func_subtract(self):
        return float(self.value) - float(self.argument)


class Search:
    """
    A class to facilitate Search across the Canalyst Modelverse.

    Parameters
    ----------
    Config object as defined by from canalyst_candas.configuration.config import Config.

    Attributes
    ----------
    ticker_list : list[str]
        list of canalyst tickers
    df_guidance : Pandas DataFrame
        DataFrame of guidance data
    df_search : Pandas DataFrame
        DataFrame of search data

    """

    def __init__(self, config: Config = None) -> None:
        self.config: Config = config or CONFIG
        self.df_search: Optional[DataFrame] = None
        self.df_guidance: Optional[DataFrame] = None
        self.ticker_list: Optional[List[str]] = None
        if self.config.s3_access_key_id == "" or self.config.s3_secret_key == "":
            print("Missing S3 keys, please contact Canalyst")

    def score_lists(self, list1, list2, ignore_list):
        """
        A function to perform custom word match score between two lists.

        Parameters
        ----------
        list1: list
        list2: list
        ignore_list: list

        Returns
        ----------
        float
        """

        list1 = [s.replace(",", "") for s in list1]
        list2 = [s.replace(",", "") for s in list2]

        score = 0
        x = len(list2)
        num_matches = 0
        for ii, s in enumerate(list1):

            if s in ignore_list:
                continue

            if s in list2:
                num_matches += 1
                i = list2.index(s) + 1
                z = list1.index(s) + 1
                x = z / i / (ii + 1)
                score = score + x
        if num_matches:
            score = score * num_matches / len(list2)
        return score

    def kpi_match(self, ticker1, ticker2, is_driver=False):
        """
        A function to match KPI between two tickers.

        Parameters
        ----------
        ticker1: str
        ticker2: str
        is_driver: boolean

        Returns
        ----------
        Pandas DataFrame
        """

        from difflib import SequenceMatcher  # we only load this if we need it

        df1 = self.search_time_series(ticker=ticker1, is_driver=is_driver)
        df2 = self.search_time_series(ticker=ticker2, is_driver=is_driver)
        like_list = ["Stats", "Analysis", "Metrics", "Segmented Results", "Revised"]
        list_string = "|".join(like_list).lower()
        df1 = df1.loc[df1["category"].str.lower().str.contains(list_string)]
        df2 = df2.loc[df2["category"].str.lower().str.contains(list_string)]

        list_df = []
        for i, row in df1.iterrows():
            ts = row["time_series_name"]
            td = row["time_series_description"]
            ut = row["unit_type"]
            i_d = row["is_driver"]

            ignore_list = [
                "",
                "significant",
                "growth",
                "effective",
                "other",
                "items",
                "one-time",
                "net",
                "income",
                "y/y",
                "change",
                "consensus",
                "share",
                "price",
                "estimated",
                "dividend",
                "buybacks",
                "adjustments",
                "tax",
                "issuance",
                "%",
                ",",
                "mm",
                " ",
                "-",
                "&",
                "and",
                "#",
                " of",
                "bn",
            ]

            split_description = list(set(td.lower().split(" ")))
            list_out = []
            list_out.append(
                [
                    "ticker_x",
                    "ticker_y",
                    "unit_type",
                    "time_series_description_x",
                    "time_series_name_x",
                    "time_series_description_y",
                    "time_series_name_y",
                    "is_driver_x",
                    "is_driver_y",
                    "word_score",
                    "diff_score",
                ]
            )

            for i2, row2 in df2.iterrows():
                row_description = str(row2["time_series_description"])
                row_name = str(row2["time_series_name"])
                row_unit_type = str(row2["unit_type"])

                if row_unit_type == ut:
                    split_row = list(set(row_description.lower().split(" ")))
                    word_score = self.score_lists(
                        split_description, split_row, ignore_list
                    )
                    diff_score = SequenceMatcher(None, ts, row_name).ratio()
                    if word_score > 0:
                        list_out.append(
                            [
                                ticker1,
                                ticker2,
                                ut,
                                td,
                                ts,
                                row_description,
                                row_name,
                                i_d,
                                row["is_driver"],
                                word_score,
                                diff_score,
                            ]
                        )

            df = pd.DataFrame(list_out)
            df.columns = df.iloc[0]
            df = df[1:]
            df["diff_rank"] = df["diff_score"].rank(pct=True)
            df["word_score_rank"] = df["word_score"].rank(pct=True)
            df["word_plus_diff_score"] = df["word_score"] + df["diff_score"]
            list_df.append(df)
        df = (
            pd.concat(list_df)
            .groupby("time_series_description_x")
            .first()
            .reset_index()
        )
        df = df[
            [
                "ticker_x",
                "ticker_y",
                "unit_type",
                "time_series_name_x",
                "time_series_description_x",
                "time_series_name_y",
                "time_series_description_y",
                "is_driver_x",
                "is_driver_y",
                "word_score",
                "diff_score",
                "word_plus_diff_score",
            ]
        ]

        return df.sort_values("word_plus_diff_score", ascending=False)

    def remove_stopwords(self, sentence):
        """
        A function to remove stopwords from a sentence.

        Parameters
        ----------
        sentence: str

        Returns
        ----------
        str
        """

        my_stopwords = [
            "in",
            "Change",
            "YoY",
            ",",
            "%",
            "(Calculated)",
            "(QoQ)",
            "QoQ",
            "mm",
        ]
        tokens = sentence.split(" ")
        tokens_filtered = [word for word in tokens if not word in my_stopwords]
        tokens_filtered = [
            "".join(c for c in s if c not in string.punctuation)
            for s in tokens_filtered
        ]
        return (" ").join(tokens_filtered)

    def kpi_statistics(self, ticker="", ticker2="", mo_only=False, subsector=""):
        """
        Return the time series correlations inside GIC subsectors.

        Parameters
        -------
        ticker : str
            Ticker with which to filter the available result set.
        ticker2 : str
            Ticker to filter against previous ticker
        mo_only : boolean "False"
            Filter to only common (MO_) names
        subsector : str
            Filter subsector

            Return the rsq between time series for two stocks

        Returns
        -------
        DataFrame

        """

        GET = Getter(config=self.config)
        path_name = f"DATA/rsq.csv"
        df = GET.get_csv_from_s3(path_name)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df[
            ~df["time_series_description_x"].str.contains("other", na=False, case=False)
        ]
        df = df[
            ~df["time_series_description_y"].str.contains("other", na=False, case=False)
        ]
        df = df[
            ~df["time_series_description_x"].str.contains("tax", na=False, case=False)
        ]
        df = df[
            ~df["time_series_description_y"].str.contains("tax", na=False, case=False)
        ]

        if subsector != "":
            df = df.loc[df["subsector"] == subsector]
            if mo_only == True:
                df1 = df.loc[df["time_series_name_y"].str.startswith("MO")]
                df2 = df.loc[df["time_series_name_x"].str.startswith("MO")]
                df = pd.concat([df1, df2])
            return df.sort_values("rsquared", ascending=False)
        if ticker2 != "":
            df1 = df[(df["ticker_y"] == ticker) & (df["ticker_x"] == ticker2)]
            df2 = df[(df["ticker_x"] == ticker) & (df["ticker_y"] == ticker2)]
        else:
            df1 = df[df["ticker_y"] == ticker]
            df2 = df[df["ticker_x"] == ticker]
        df = pd.concat([df1, df2]).sort_values("rsquared", ascending=False)
        df = df[~df["time_series_description_y"].str.contains("Tax")]
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        if mo_only == True:
            df1 = df.loc[df["time_series_name_y"].str.startswith("MO")]
            df2 = df.loc[df["time_series_name_x"].str.startswith("MO")]
            df = pd.concat([df1, df2])
            if df.shape[0] != 0:
                df = df.sort_values("rsquared", ascending=False)

        return df

    def get_ticker_list(self, ticker="", company_name=""):
        """
        Return the tickers available on the Canalyst system.

        Parameters
        -------
        ticker : str
            Ticker with which to filter the available result set.

            Easy way to check whether we cover a given stock.

        Returns
        -------
        DataFrame

        """
        if self.ticker_list is None:
            GET = Getter(config=self.config)
            path_name = f"DATA/cwl_all.csv"
            df = GET.get_csv_from_s3(path_name)
            df = df.drop(columns=["CSIN"])
            self.ticker_list = df
        else:
            df = self.ticker_list

        if company_name != "":
            df = df.loc[df["File"].str.contains(company_name, case=False)]

        if ticker != "":
            df_ret = df.loc[df["Ticker Alternate 1"] == ticker]
            if df_ret.shape[0] == 0:
                df_ret = df.loc[
                    df["Ticker Alternate 1"].str.contains(ticker, case=False, na=False)
                ]
            df = df_ret
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        return df

    def search_guidance_time_series(
        self,
        ticker="",
        sector="",
        time_series_name="",
        time_series_description="",
        query="",
        most_recent="",
    ):
        """
        Search guidance time series across all available guidance data.

        Parameters
        ----------
            ticker : str or list
                ticker in AXP US format
            sector : str
                searches the Path string
            time_series_name : str
                simple regex match for time_series_name contains the input string
            time_series_description : str
                whole word match and returns number of whole word matches
            most_recent : True or False
                filter to only the most recent for each time series
            query : str
                filter value against a pandas query e.g. Mid > 5

        Returns
        --------
            DataFrame
        """
        if self.df_guidance is None:
            body = Getter(config=self.config).get_zip_csv_from_s3(
                f"DATA/df_guidance.zip"
            )
            self.df_guidance = pd.read_csv(io.StringIO(body), low_memory=False)

        df = self.df_guidance
        df = df.loc[df["Type.1"] == "Estimate"]

        df = df.rename(
            columns={"Item": "time_series_description", "Item Name": "time_series_name"}
        )

        if most_recent == True:
            df = df.sort_values(["ticker", "time_series_name", "Date"], ascending=False)
            df = df.groupby(["ticker", "time_series_name"]).first().reset_index()

        if type(ticker) == list:
            df = df.loc[df["ticker"].isin(ticker)]
        elif ticker != "":
            df = df.loc[df["ticker"] == ticker]

        if sector != "":
            df = df.loc[df["Path"].str.contains(sector, case=True, regex=True)]
        if time_series_name != "":
            df = df.loc[
                df["time_series_name"].str.contains(
                    time_series_name, case=False, regex=True
                )
            ]

        if time_series_description != "":

            ts_lower = time_series_description.lower()
            split_strings = ts_lower.split(" ")
            df["ts_lower"] = df["time_series_description"].str.lower()
            df["search_matches"] = df["ts_lower"].str.count("|".join(split_strings))
            df = df.loc[df["search_matches"] > 0]

        else:
            df["search_matches"] = 1

        df = df[
            [
                "ticker",
                "Path",
                "Filename",
                "time_series_description",
                "time_series_name",
                "Fiscal Period",
                "Low",
                "Mid",
                "High",
                "Type.1",
                "Date",
                "Link",
                "search_matches",
            ]
        ].sort_values(
            ["search_matches", "ticker", "Date", "time_series_name"], ascending=False
        )

        if query != "":
            df = df.query(query)
            df = df.sort_values(["search_matches", "Mid"], ascending=False)
        return df

    def match_time_series(self, ticker="", time_series=""):
        if type(time_series) == "str":
            time_series = [time_series]

        df_list = []
        for item in time_series:
            df = self.search_time_series(ticker=ticker, time_series_description=item)
            df["target"] = item
            df_list.append(df)

        return pd.concat(df_list)

    def search_time_series_excel(
        self,
        ticker="",
        time_series_name="",
        time_series_description="",
        query="",
    ):
        """
        Search time series across all available time series data.
        Optimized for Excel.
        Uses a different (smaller) zip file and reads in chunks.

        Parameters
        ----------
        ticker : str or list[str]
            ticker in AXP US format
        time_series_name : str
            simple regex match for time_series_name contains the input string
        time_series_description : str
            whole word match and returns number of whole word matches
        query : str
            example: "value > 5"

        Returns
        --------
        DataFrame
        """
        pd.options.mode.chained_assignment = None
        print("Building search index...")
        body = Getter(config=self.config).get_zip_csv_from_s3(
            f"DATA/df_search_excel.zip"
        )

        df_list = []
        chunksize = 100000

        for chunk in pd.read_csv(
            io.StringIO(body),
            usecols=["ticker", "time_series_description", "time_series_name", "value"],
            chunksize=chunksize,
        ):
            if type(ticker) == list:
                if len(ticker[0]) > 0:
                    df = chunk.loc[chunk["ticker"].isin(ticker)]
                else:
                    df = chunk
            elif ticker != "":
                print(ticker)
                df = chunk.loc[chunk["ticker"] == ticker]
            else:
                df = chunk

            if time_series_name != "":
                df = df.loc[
                    df["time_series_name"].str.contains(
                        time_series_name, case=False, regex=True
                    )
                ]

            if time_series_description != "":

                ts_lower = time_series_description.lower()

                split_strings = ts_lower.split(" ")
                df["ts_lower"] = df["time_series_description"].str.lower()
                df["search_matches"] = df["ts_lower"].str.count("|".join(split_strings))
                df = df.drop(columns=["ts_lower"])
                df = df.loc[df["search_matches"] > 0]

            else:
                df["search_matches"] = 1

            if df.shape[0] > 0:
                df_list.append(df)
            df = None
            chunk = None

        if len(df_list):
            df = pd.concat(df_list)

        df = df[
            [
                "ticker",
                "time_series_description",
                "time_series_name",
                "value",
                "search_matches",
            ]
        ].sort_values(
            ["search_matches", "ticker", "time_series_description"], ascending=False
        )
        if query != "":
            df = df.query(query)
            df = df.sort_values(["search_matches", "value"], ascending=False)

        return df

    def search_time_series(
        self,
        ticker="",
        sector="",
        time_series_name="",
        time_series_description="",
        category="",
        is_driver="",
        unit_type="",
        mo_only=False,
        exact_match=False,
        period_duration_type="",
        query="",
    ):
        """
        Search time series across all available time series data.

        Parameters
        ----------
            ticker : str or list
                ticker in AXP US format
            time_series_name : str
                simple regex match for time_series_name contains the input string
            time_series_description : str
                whole word match and returns number of whole word matches
            category : str
                category from the path string
            is_driver : str
                is this a driver series
            unit_type : str
                unit type currency, percentage, count, ratio, time
            mo_only : str
                mo names only
            exact_match : str
                only return exact matches
            period_duration_type : str
                fiscal_year or fiscal_quarter
            query : str
                whole word match and returns number of whole word matches

        Returns
        --------
            DataFrame
        """

        pd.options.mode.chained_assignment = None
        if self.df_search is None:
            print("Building search index...")
            body = Getter(config=self.config).get_zip_csv_from_s3(f"DATA/df_search.zip")
            self.df_search = pd.read_csv(
                io.StringIO(body),
                usecols=[
                    "ticker",
                    "CSIN",
                    "Filename",
                    "Path",
                    "publish_date",
                    "update_type",
                    "category",
                    "time_series_description",
                    "time_series_name",
                    "is_driver",
                    "period_duration_type",
                    "unit_type",
                    "value",
                ],
            )

        df = self.df_search

        if type(ticker) == list:
            df = df.loc[df["ticker"].isin(ticker)]
        elif ticker != "":
            df = df.loc[df["ticker"] == ticker]

        if mo_only == True:
            df = df.loc[df["time_series_name"].str.startswith("MO_")]

        if sector != "":
            df = df.loc[df["Path"].str.contains(sector, case=True, regex=True)]

        if category != "":
            df = df.loc[df["category"].str.contains(category, case=True, regex=True)]

        if time_series_name != "":
            if exact_match == True:
                df = df.loc[
                    df["time_series_name"] == time_series_name
                ]  # dont lowercase this
            else:
                df = df.loc[
                    df["time_series_name"].str.contains(
                        time_series_name, case=False, regex=True
                    )
                ]

        if time_series_description != "":

            ts_lower = time_series_description.lower()
            if exact_match == True:
                df = df.loc[df["time_series_description"].str.lower() == ts_lower]
            else:
                split_strings = ts_lower.split(" ")
                df["ts_lower"] = df["time_series_description"].str.lower()
                df["search_matches"] = df["ts_lower"].str.count("|".join(split_strings))
                df = df.drop(columns=["ts_lower"])
                df = df.loc[df["search_matches"] > 0]
        else:
            df["search_matches"] = 1

        if is_driver != "":
            df = df.loc[df["is_driver"] == is_driver]

        if unit_type != "":
            df = df.loc[
                df["unit_type"] == unit_type
            ]  # currency, percentage, count, ratio, time

        if period_duration_type != "":
            df["period_duration_type"] = np.where(
                df["value"].isna(), "fiscal_year", "fiscal_quarter"
            )

        df["period_duration_type"] = np.where(
            df["value"].isna(), "fiscal_year", "fiscal_quarter"
        )

        df = df[
            [
                "ticker",
                "time_series_description",
                "time_series_name",
                "is_driver",
                "category",
                "Path",
                "Filename",
                "period_duration_type",
                "unit_type",
                "value",
                "CSIN",
                "search_matches",
            ]
        ].sort_values(
            ["search_matches", "ticker", "time_series_description"], ascending=False
        )
        if query != "":
            df = df.query(query)
            df = df.sort_values(["search_matches", "value"], ascending=False)

        n = len(pd.unique(df["ticker"]))
        print("No.of.unique tickers: ", n)
        n = len(pd.unique(df["time_series_name"]))
        print("No.of.unique time series: ", n)

        return df


# a class of multiple models
class ModelSet:
    """
    Generate a dictionary of Model objects, provide ease-of-use functions across the dictionary.

    Parameters
    ----------
    allow_nulls : boolean, default False
        Either return the intersection of all model data (False) or the union (True)
    company_info : boolean, default True
        Include additional meta data calls
    file_type : str, default "parquet"
        {"csv":csv from MDS, "parquet":parquet from MDS}
    """

    def __init__(
        self,
        ticker_list: List[str],
        config: Config = None,
        extract_drivers: bool = True,
        allow_nulls: bool = False,
        company_info: bool = True,
        file_type: str = "parquet",
    ):
        """
        Generate a dictionary of Model objects, provide ease of use functions across the dictionary.

        Parameters
        ----------
        ticker_list : list
        allow_nulls : boolean default True
            Either return the intersection of all model data (False) or the union (True)
        company_info : boolean default True
            Include additional meta data calls
        file_type : str, default "parquet"
            {"csv":csv from MDS, "parquet":parquet from MDS}
        """
        self.file_type = file_type
        self.allow_nulls = allow_nulls
        self.config = config or CONFIG
        self.extract_drivers = extract_drivers
        self.log = LogFile()

        if type(ticker_list) is not list:
            if type(ticker_list) is str:
                try:
                    self.ticker_list = ticker_list.split(",")
                except:
                    self.ticker_list = [self.ticker_list]
        else:
            self.ticker_list = ticker_list

        self.drivers: DataFrame = None
        self.models: Dict[str, "Model"] = {}
        self.company_info = company_info
        self.api_headers: Dict[str, str] = get_api_headers(self.config.canalyst_api_key)

        self.get_featurelibrary()  # set self._features

    def help(self, function_name=""):
        dict_help = {
            "create_model_map": "Create a model map.  params: ticker, col_for_labels = 'time_series_name', time_series_name = 'MO_RIS_REV', tree = True, notebook = True",
            "create_time_series_chart": "Create a time series chart.  params: ticker, time_series_name",
            "guidance": "Return guidance from a mmodel.  params: ticker",
            "mrq": "Return most recent quarter from a mmodel.  params: ticker",
            "time_series_search": "Return time series regex match. params: time_series_name",
            "driver_search": "Return driver regex match. params: driver_name",
            "model_frame": "Return a DataFrame of the full modelset. params: time_series_name,period_name,is_driver='',pivot=False,mrq=False,period_duration_type='',is_historical='',n_periods='',mrq_notation=False",
            "forecast_frame": "Return a params DataFrame for use in the fit function. params: time_series_name, n_periods, function_name='value', function_value='' where function name can be add, subtract, multiply, divide, or value",
            "fit": "Return a return series for a fitted model. params: params DataFrame, return_series.  params DataFrame columns are: ticker period time_series_name value new_value",
            "model_set.models[Bloomberg Ticker].get_most_recent_model_date()": "Return most recent model upload date for a ticker",
        }
        if function_name != "":
            return dict_help[function_name]
        else:
            df = pd.DataFrame(dict_help, index=[0]).T
            df.columns = ["help"]
            return df

    def unstacked_frame(
        self, ticker="", time_series_name="", period_duration_type="", is_historical=""
    ):
        DFunc = DataFunctions()
        df = self.model_frame(
            period_duration_type=period_duration_type,
            is_historical=is_historical,
        )
        df = DFunc.unstack(df, ticker, time_series_name, "period_name_sorted")

        return df

    def time_series_formula(
        self,
        arguments=[],
        modifier="",
        time_series_name="",
        time_series_description="",
    ):
        df = self.model_frame(warning=False)
        DFunc = DataFunctions()
        df = DFunc.time_series_function(
            df,
            arguments,
            modifier,
            time_series_name,
            time_series_description,
        )
        self._features = df  # modify in place
        return

    def plot_time_series(
        self,
        ticker="",
        time_series_name="",
        mrq_notation=False,
        period_duration_type="fiscal_quarter",
        axis_labels=[["Periods", "Value"]],
    ):
        df = self.model_frame(
            ticker=ticker,
            time_series_name=time_series_name,
            period_duration_type=period_duration_type,
            is_historical=True,
        )

        df = (
            df.groupby(["ticker", "period_name_sorted"])
            .first()
            .reset_index()
            .sort_values(["ticker", "period_name_sorted"], ascending=False)
        )
        df = df[["ticker", "period_name_sorted", "value"]]
        if mrq_notation == True:
            df["Period"] = df.groupby(["ticker"]).cumcount() + 1
            df["Period"] = len(df["Period"]) - df["Period"]
            df["Period"] = df["Period"] - len(df["Period"]) - 1
        else:
            df["Period"] = df["period_name_sorted"]

        df = df.pivot(values="value", index="Period", columns="ticker")
        df.dropna()
        df_plot = df.dropna().reset_index()

        if ticker == "":
            ticker_list = self.ticker_list
        else:
            ticker_list = [ticker]

        model_chart = Chart(
            x_value=df_plot["Period"],
            y_values=df_plot[ticker_list],
            labels=ticker_list,
            title=time_series_name,
            axis_labels=axis_labels,
        )
        model_chart.show()
        return

    def common_time_series_names(self):
        """
        Return a list of time series names common to all models in the ModelSet
        """
        return self._common_time_series_names

    def plot_guidance(self, ticker, time_series_name):
        if ticker not in self.ticker_list:
            print("Please choose a ticker in this ModelSet")
            return
        self.models[ticker].plot_guidance(time_series_name)
        return

    def pe_dataset(self, ticker, yahoo_ticker, index_ticker="^GSPC", n_periods=24):
        """
        Create a join between Yahoo price data and the base _features DataFrame of the modelset.

        This function returns long form data outer joined with price data.
        Also creates a rolling beta calculation against the index input.

        Parameters
        ----------
        ticker : str
            Canalyst ticker
        yahoo_ticker : str
            Yahoo ticker
        index_ticker : str, default ^GSPC
            Index to use for betas
        n_periods : int, default 24
            number of canalyst periods

        Returns
        -------
        DataFrame

        """

        import canalyst_candas.candas_datareader as cdr

        df_earnings = cdr.get_earnings_and_prices(yahoo_ticker, index_ticker)
        if df_earnings is None:
            return None

        df_earnings = df_earnings[
            [
                "ticker",
                "earnings_date",
                "beta_252",
                "alpha_1_day",
                "alpha_5_day",
                "alpha_10_day",
            ]
        ]
        df_earnings = df_earnings.sort_values("earnings_date", ascending=False)

        df_model = self.model_frame(
            ticker=ticker, is_historical=True, period_duration_type="fiscal_quarter"
        )

        edq_list = list(
            cdr.calendar_quarter(df_earnings, "earnings_date", datetime=True)[
                "earnings_date_CALENDAR_QUARTER"
            ]
        )

        mdq_list = list(
            df_model.groupby("period_name_sorted")
            .first()
            .reset_index()
            .sort_values("period_end_date", ascending=False)["period_name_sorted"]
        )

        if len(edq_list) < n_periods:
            n_periods = len(edq_list)
            print(
                "Price earnings dataset using " + str(n_periods) + " periods available"
            )

        d = {
            "earnings_date_q": edq_list[0:n_periods],
            "period_name_sorted": mdq_list[0:n_periods],
        }

        df_dates = pd.DataFrame(d)
        df_model = pd.merge(df_model, df_dates)
        df_earnings["earnings_date_q"] = cdr.calendar_quarter(
            df_earnings, "earnings_date", datetime=True
        )["earnings_date_CALENDAR_QUARTER"]
        df_model["price_ticker"] = yahoo_ticker
        df_model = df_model.drop(columns=["ticker"])
        df_data = pd.merge(
            df_model,
            df_earnings,
            how="inner",
            left_on=["price_ticker", "earnings_date_q"],
            right_on=["ticker", "earnings_date_q"],
        )
        df_data["ticker"] = ticker
        df_data = df_data.drop(
            columns=[
                "earnings_dateshift",
                "earnings_date_CALENDAR_QUARTER",
                "category_type_slug",
                "time_series_slug",
            ]
        )
        return df_data

    def regress_dataframe_time_series_groups(
        self, df_data=None, y_name="alpha_10_day", return_grouped=True
    ):
        """
        Create grouped regressions across time series groups.

        For use with datasets created with pe_dataset function above.

        Parameters
        ----------
        df_data : DataFrame
            Dataset created with pe_dataset
        y_name : str, default "alpha_10_day"
            window for alpha (outperformance vs market) calculation
            { "alpha_10_day", "alpha_5_day", "alpha_1_day" }
        return_grouped : boolean, default True
            grouped is the default, non-grouped only for debug

        Returns
        -------
        DataFrame

        """

        ticker = df_data.iloc[0]["ticker"]
        import canalyst_candas.candas_datareader as cdr

        df = cdr.regress_dataframe_groups(
            df_data, y_name=y_name, return_grouped=return_grouped
        )
        df = df.dropna()
        df["ticker"] = ticker
        return df

    def create_model_map(
        self,
        ticker,
        col_for_labels="time_series_name",
        time_series_name="MO_RIS_REV",
        tree=True,
        notebook=False,
    ):
        """
        Return a ModelMap for a given time series name.

        Use show() to display the ModelMap.

        Parameters
        ----------
        col_for_labels : str, default time_series_name
            Label to be used on the map.
            { "time_series_name", "time_series_description" }
        time_series_name : str, default MO_RIS_REV
            String of time_series_name to be used for the ROOT node of the tree
        tree : boolean, default True
            Very complex trees might be rendered better with tree=False
        notebook : boolean, default False
            Launches tree in a separate browser window (False) or in the notebook (True)

        Returns
        -------
        ModelMap

        """
        if type(ticker) == list:
            print("Please request one ticker at a time.")
            return

        if ticker in self.ticker_list:
            model = self.models[ticker]
            return model.create_model_map(
                time_series_name=time_series_name,
                tree=tree,
                col_for_labels=col_for_labels,
                notebook=notebook,
                common_time_series_names=self._common_time_series_names,
            )
        else:
            print("Please choose a ticker in this ModelSet's ticker list")
            return

    def create_time_series_chart(
        self, ticker_list="", time_series_name="", historical=False
    ):  # MODELSET
        """
        This function will be replaced by Chart in the ModelFrame class
        """

        if ticker_list == "":
            df = self._features
            if type(time_series_name) == list:
                df = df.loc[self._features["time_series_name"].isin(time_series_name)]
            else:
                df = df.loc[self._features["time_series_name"] == time_series_name]

            if historical:
                df = df.loc[df["is_historical"] == historical]

            df = df.loc[df["period_name"].str.contains("Q")].sort_values(
                ["ticker", "period_end_date"]
            )
            df = df.loc[df["period_end_date"] > "2015-01-01"]
            df = df.dropna(subset=["value"])
            # return df
            # df = df.groupby('ticker').size().order(ascending=False).reset_index()

            row1 = df.iloc[0]
            # title = self.ticker + " " + str(row1["time_series_description"])
            xlabel = "Fiscal Quarter"
            # ylabel = row1["time_series_description"]

            colors = {
                "red": "#ff207c",
                "grey": "#C3C2C3",
                "blue": "#00838F",
                "orange": "#ffa320",
                "green": "#00ec8b",
            }

            plt.rc("figure", figsize=(12, 9))
            if type(time_series_name) != list:

                df_plot = df.loc[
                    df["time_series_name"] == time_series_name
                ].sort_values(["ticker", "time_series_name"])

                if ticker_list == "":
                    ticker_list = self.ticker_list
                elif type(ticker_list) == list:
                    ticker_list = ticker_list
                else:
                    ticker_list = [ticker_list]

                for ticker in ticker_list:

                    df_plot_one = df_plot[df_plot["ticker"] == ticker]
                    df_plot_one = df_plot_one.sort_values("period_end_date")
                    df_plot_one = df_plot_one.loc[df_plot_one["value"].notna()]

                    if df_plot_one.shape[0] > 0:
                        plt.plot(
                            df_plot_one["period_end_date"],
                            df_plot_one["value"],
                            label=ticker + " " + time_series_name,
                            linewidth=2,
                        )
                        plt.axvline(
                            x=self.models[ticker].mrq()["period_end_date"][0],
                            color="black",
                        )
            else:
                if ticker_list == "":
                    ticker_list = self.ticker_list
                elif type(ticker_list) == list:
                    ticker_list = ticker_list
                else:
                    ticker_list = [ticker_list]

                for tsn in time_series_name:
                    df_plot = df.loc[df["time_series_name"] == tsn]
                    for ticker in ticker_list:
                        df_plot_one = df_plot[df_plot["ticker"] == ticker]
                        df_plot_one = df_plot_one.sort_values("period_end_date")
                        df_plot_one = df_plot_one.loc[df_plot_one["value"].notna()]
                        if df_plot_one.shape[0] > 0:
                            plt.plot(
                                df_plot_one["period_end_date"],
                                df_plot_one["value"],
                                label=ticker + " " + tsn,
                                linewidth=2,
                            )
                            plt.axvline(
                                x=self.models[ticker].mrq()["period_end_date"][0],
                                color="black",
                            )

            plt.xticks(rotation=90)
            plt.legend()
            plt.show()
            return
        if ticker_list == "":
            ticker_list = self.ticker_list
        elif type(ticker_list) == list:
            ticker_list = ticker_list
        else:
            ticker_list = [ticker_list]

        if time_series_name == "":
            print("Please enter one time_series_name")
            return
        for ticker in ticker_list:
            df = self._features
            if historical:
                df = df.loc[df["is_historical"] == historical]

            df_plot = df.loc[df["time_series_name"] == time_series_name]
            df_plot_one = df_plot[df_plot["ticker"] == ticker]
            df_plot_one = df_plot_one.sort_values("period_end_date")
            df_plot_one = df_plot_one.loc[df_plot_one["value"].notna()]
            if df_plot_one.shape[0] > 0:
                plt.plot(
                    df_plot_one["period_end_date"],
                    df_plot_one["value"],
                    label=ticker + " " + str(time_series_name),
                    linewidth=2,
                )
                plt.axvline(
                    x=self.models[ticker].mrq()["period_end_date"][0],
                    color="black",
                )

        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

        # if ticker in self.ticker_list:
        #    self.models[ticker].create_time_series_chart(time_series_name, historical)
        # else:
        #    print("Please choose a ticker in this ModelSet's ticker list")
        return

    def guidance(self, tickers: List[str], time_series_name="") -> Optional[DataFrame]:
        """
        Return a DataFrame of guidance for a given ticker(s).

        Renames the default guidance col headers with time_series_name and time_series_description for consistency.

        Parameters
        ----------
        tickers : list[str]
            A ticker in the ModelSet ticker list
        time_series_name : str, default ""
            Allow for str.contains() filtering of time series name

        Returns
        -------
        DataFrame

        """
        if self.config.s3_access_key_id == "" or self.config.s3_secret_key == "":
            print("Missing S3 keys, please contact Canalyst")

        list_df = []
        for ticker in tickers:
            if ticker in self.ticker_list:
                df = self.models[ticker].guidance()
                df["ticker"] = ticker
                col_name = "ticker"
                first_col = df.pop(col_name)
                df.insert(0, col_name, first_col)
                list_df.append(df)
            else:
                print(
                    "Please enter a list or choose a ticker in this ModelSet's ticker list"
                )
        df = pd.concat(list_df)
        df = df.rename(
            columns={"Item": "time_series_description", "Item Name": "time_series_name"}
        )
        if time_series_name != "":
            df = df.loc[
                df["time_series_description"].str.contains(time_series_name, case=False)
            ]
        return df

    def mrq(self, tickers: List[str]) -> Optional[DataFrame]:
        """
        Return the most recent quarter for a list of tickers
        """
        list_df = []
        for ticker in tickers:
            if ticker in self.ticker_list:
                df = self.models[ticker].mrq()
                df["ticker"] = ticker
                list_df.append(df)
            else:
                print("Please choose a ticker in this ModelSet's ticker list")
        return pd.concat(list_df)

    def get_featurelibrary(self, col="time_series_name"):
        """
        Creates the default DataFrame features and drivers.
        """
        list_all = []
        list_df = []

        for ticker in self.ticker_list:
            list_rows = []
            try:
                md = Model(
                    config=self.config,
                    ticker=ticker,
                    company_info=self.company_info,
                    file_type=self.file_type,
                )
                if hasattr(md, "_model_frame"):
                    self.models[ticker] = md

                    df = md._model_frame

                    list_df.append(df)

                    if df is not None:
                        list_rows = list(set(list(df[col])))
                        list_all.append(list_rows)
            except Exception as ex:
                print(f"Model error: {ticker}. Error: {ex}")

        if len(list_df) == 0:
            return

        from itertools import chain

        # faster than concat
        def fast_flatten(input_list):
            return list(chain.from_iterable(input_list))

        COLUMN_NAMES = list_df[0].columns
        df_dict: Dict[Any, list] = dict.fromkeys(COLUMN_NAMES, [])
        for col in COLUMN_NAMES:
            extracted = (df[col] for df in list_df)
            # Flatten and save to df_dict
            df_dict[col] = fast_flatten(extracted)

        df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
        list_df = []
        df_dict = {}

        res = list(reduce(lambda i, j: i & j, (set(x) for x in list_all)))  # type: ignore
        self._common_time_series_names: list = res

        if self.allow_nulls == False:
            df = df_filter(df, "time_series_name", res)
            self._features = df
        else:
            self._features = df

            if df is not None:
                list_rows = list(set(list(df[col])))
                list_all.append(list_rows)
        return

    # allows for filtering and shaping the default DataFrame
    # MODELSET model_frame
    def model_frame(
        self,
        time_series_name="",
        period_name="",
        is_driver="",
        pivot=False,
        mrq=False,
        period_duration_type="",
        is_historical="",
        n_periods="",
        mrq_notation=False,
        unit_type="",
        category="",
        warning=True,
        ticker="",
    ):
        """
        Create a Pandas DataFrame with domain-specific filters.

        This function returns long form Pandas DataFrame filtered by input.

        Parameters
        ----------
        time_series_name : str or list[str], default ""
            pass a list for an exact match or a string for a regex match
        period_name : str
            a specific period name EG Q1-2022
        is_driver : boolean, default ""
            return only drivers if True or non-drivers if False, both if ""
        pivot : boolean, default ""
            pivot the DataFrame from long to wide
        mrq : boolean, default False
            return only the most recent quarter
        period_duration_type : str, default ""
            fiscal_year or fiscal_quarter
        is_historical : boolean, default ""
            only historical data if True, only forecast if False, both if ""
        n_periods : int, default ""
            number of fiscal year and or fiscal quarters to return
            use positive integers for historical periods
            use negative integers to return mrq+n periods in the case where is_historical=False
        mrq_notation : boolean, default False
            applies to pivot and applies most recent quarter / fiscal year notation
            MRQ1 MRQ2 FY1 FY2
        unit_type : str, default ""
            ratio, etc, "" returns all
        category : str, default ""
            Balance Sheet, etc, "" returns all
        warning : boolean, default True
            warns if fiscal_year or fiscal_quarter is not set
        ticker : str or list[str], default ""
            filters to a specific ticker or list of tickers, all if ""

        Returns
        -------
        DataFrame

        """

        if ticker != "":
            df = self.models[ticker]._model_frame
        else:
            df = self._features

        return filter_dataset(
            df,  # this is the core dataset of the modelset class
            time_series_name,
            period_name,
            is_driver,
            pivot,
            mrq,
            period_duration_type,
            is_historical,
            n_periods,
            mrq_notation,
            unit_type,
            category,
            warning,
        )

    def forecast_frame(
        self, time_series_name, n_periods, function_name="value", function_value=""
    ):
        """
        Builds a param DataFrame for use in the fit() function.

        Parameters
        ----------
        n_periods : int
            Number of periods to return.
        function_name : str default value
            { "multiply", "add", "divide", "subtract", "value" }
        function_value : str
            the number by which you want to apply the function or just the value to assign
            Example: if function_name = multiply then 1.1 multiplies by 10pct
            Example: if function_name = value then 5 assigns the value 5

        Returns
        ------
        DataFrame
        """

        df = self.model_frame(
            [time_series_name],
            period_duration_type="fiscal_quarter",
            is_historical=False,
            n_periods=n_periods,
        )

        if function_name != "value":
            d = FuncMusic()
            df["new_value"] = df.apply(
                lambda row: d.apply_function(
                    row["value"], modifier=function_name, argument=function_value
                ),
                axis=1,
            )
        else:
            df["new_value"] = function_value

        df = df[
            [
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
                "value",
                "new_value",
                "period_end_date",
            ]
        ]

        return df

    # MODELSET fit()
    def fit(self, params, return_series):
        """
        Runs the scenario engine against a param DataFrame created by forecast frame above.

        Or a user generated param DataFrame with the columns:
        ticker | period_name | time_series_name | time_series_description | value new_value | period_end_date

        Parameters
        ----------
        params : DataFrame
            Described above
        return series : str
            series to return. "MO_RIS_REV" returns revenue

        Returns
        ------
        DataFrame
        """

        # ticker period time_series_name value new_value
        dict_summary = {}
        if type(params) is dict:
            df = pd.DataFrame.from_dict(params, orient="index").reset_index()
        else:
            df = params

        df_grouped = df.groupby("ticker").first().reset_index()

        dict_data = {}
        for _, row in df_grouped.iterrows():

            df_ticker = df[df["ticker"] == row["ticker"]]
            ticker = row["ticker"]  # .iloc[0]
            self.models[ticker].set_new_uuid()
            list_changes = []

            for _, row in df_ticker.iterrows():
                feature_value = row["new_value"]
                feature_name = row["time_series_name"]
                feature_period = row["period_name"]
                ticker = row["ticker"]

                list_changes.append(
                    {
                        "time_series": feature_name,
                        "period": feature_period,
                        "value_expression": {"type": "literal", "value": feature_value},
                    }
                )

                data = {"changes": list_changes, "name": self.models[ticker].uuid}
                dict_data[ticker] = data

        ticker_list = list(set(list(df["ticker"])))

        for ticker in ticker_list:
            res = send_scenario(
                ticker,
                dict_data[ticker],
                self.api_headers,
                self.log,
                self.models[ticker].csin,
                self.models[ticker].latest_version,
                self.config.mds_host,
                self.config.verify_ssl,
            )
            try:
                self.models[ticker].record_scenario_url(res)
            except ScenarioException as e:
                error_message = {}
                print("Error code : ", e, "error")
                error_message = res.json()
                print("\n", error_message)
                continue

            try:
                self.models[ticker].model_fit(return_series)
                dict_summary[ticker] = self.models[ticker].summary()
            except Exception as e:
                print("Error on fit for " + return_series)
                print(f"Error: {e}")

        return dict_summary

    def filter_summary(self, dict_summary, period_type="Q"):

        pd.set_option("mode.chained_assignment", None)
        pd.set_option("display.float_format", lambda x: "%.5f" % x)

        list_out = []
        for ticker in dict_summary.keys():

            df = dict_summary[ticker]

            list_out.append(df)
        df = pd.concat(list_out)
        df["sort_period_name"] = (
            df["period_name"].str.split("-").str[1]
            + df["period_name"].str.split("-").str[0]
        )
        df = df.sort_values(["ticker", "sort_period_name"])
        df = df[df["period_name"].str.contains(period_type)]
        df = df.drop(columns="sort_period_name")

        return df

    def chart(
        self,
        time_series_name: Union[str, List[str]] = "",
        ticker: Union[str, List[str]] = "",
        period_duration_type: str = "fiscal_quarter",
    ) -> None:
        """
        Charts a time series with a ticker

        Usage: Can chart single time series for a single ticker: chart(time_series_name="MO_MA_Fuel", ticker="MESA US")
           or a single time series for multiple tickers: chart(time_series_name="MO_MA_Fuel", ticker=['AZUL US','MESA US'])
           or multiple time series for a single ticker: chart(time_series_name=["MO_MA_Fuel", "MO_RIS_REV"], ticker='MESA US')

        Period duration type is used to filter data by fiscal quarter or by fiscal year.

        Parameters
        ----------
        time_series_name: str or List[str]
        ticker: str or List[str]
        period_duration_type: str
        """

        def _check_for_error_handling(
            ticker: Union[str, List[str]],
            time_series_name: Union[str, List[str]],
            period_duration_type: str,
        ):
            """
            Basic error handling to check that the correct types are passed into the function
            """
            if type(ticker) == list and type(time_series_name) == list:
                raise TypeError(
                    "Ticker and time_series_name cannot both be lists. Please change one of the parameters to a single string value."
                )

            if period_duration_type not in ["fiscal_quarter", "fiscal_year"]:
                raise ValueError(
                    "Please specify fiscal_year or fiscal_quarter for period_duration_type"
                )

        def _check_for_time_series_name(
            df: DataFrame, time_series_name: Union[str, List[str]]
        ):
            """
            Checks that the time series is in the ModelSet
            """
            if type(time_series_name) == str:
                time_series_check = df[df["time_series_name"] == time_series_name]
                if time_series_check.empty is True:
                    raise ValueError(
                        f"No time series found by the name {time_series_name}, please try another time series name"
                    )
            if type(time_series_name) == list:
                for item in time_series_name:
                    if self._features[
                        self._features["time_series_name"].isin([item])
                    ].empty:
                        raise ValueError(
                            f"No time series found by the name {item}, please try another time series name"
                        )

        def _check_for_ticker(ticker_list: List[str], ticker: Union[str, List[str]]):
            """
            Checks that the ticker is in the ModelSet
            """
            if type(ticker) == str and ticker not in ticker_list:
                raise ValueError(
                    f"Ticker {ticker} not found in ModelSet. Please use a ticker from the ModelSet"
                )
            if type(ticker) == list:
                for item in ticker:
                    if item not in self.ticker_list:
                        raise ValueError(
                            f"Ticker {item} not found in ModelSet. Please use a ticker from the ModelSet"
                        )

        def _sort_chronological(df: DataFrame, period_duration_type: str):
            """
            Returns a chronoglically sorted dataframe for fiscal quarters

            Ex. Ordering of these values Q1-2021, Q2-2018 would return Q2-2018, Q1-2021
            """
            if period_duration_type == "fiscal_quarter":
                df["year"] = df.period_name.str[3:]
                df = df.sort_values(["year", "period_name"])
            return df

        _check_for_error_handling(ticker, time_series_name, period_duration_type)
        _check_for_ticker(self.ticker_list, ticker)
        _check_for_time_series_name(self._features, time_series_name)

        if type(ticker) == str and type(time_series_name) == str:
            df_plot = self._features[
                (self._features.ticker == ticker)
                & (self._features.time_series_name == time_series_name)
                & (self._features.period_duration_type == period_duration_type)
            ]

            df_plot = (
                df_plot[["time_series_name", "period_name", "value"]]
                .pivot_table(
                    values="value", index=["period_name"], columns=["time_series_name"]
                )
                .reset_index()
            )
            df_plot = _sort_chronological(df_plot, period_duration_type)

            chart = Chart(
                x_value=df_plot["period_name"],
                y_values=df_plot[time_series_name],
                labels=[time_series_name],
                axis_labels=[["Periods", "Value"]],
                title=time_series_name,
            )
            chart.show()

        elif type(ticker) == list and type(time_series_name) == str:
            df_plot = self._features[
                (self._features.period_duration_type == period_duration_type)
                & (self._features.time_series_name == time_series_name)
            ]

            df_plot = (
                df_plot[df_plot["ticker"].isin(ticker)][
                    ["ticker", "period_name", "value"]
                ]
                .pivot_table(values="value", index=["period_name"], columns=["ticker"])
                .reset_index()
            )
            df_plot = _sort_chronological(df_plot, period_duration_type)

            chart = Chart(
                x_value=df_plot["period_name"],
                y_values=df_plot[ticker],
                labels=ticker,
                axis_labels=[["Periods", "Value"]],
                title=time_series_name,
            )

            chart.show()

        elif type(time_series_name) == list and type(ticker) == str:
            df_plot = self._features[
                (self._features.period_duration_type == period_duration_type)
                & self._features["time_series_name"].isin(time_series_name)
            ]

            df_plot = (
                df_plot[df_plot["ticker"].isin([ticker])][
                    ["time_series_name", "period_name", "value"]
                ]
                .pivot_table(
                    values="value", index=["period_name"], columns=["time_series_name"]
                )
                .reset_index()
            )
            df_plot = _sort_chronological(df_plot, period_duration_type)

            chart = Chart(
                x_value=df_plot["period_name"],
                y_values=df_plot[time_series_name],
                labels=time_series_name,
                axis_labels=[["Periods", "Value"]],
                title=ticker,
            )
            chart.show()
        else:
            raise TypeError(
                "Incorrect parameter types. Please refer to the function documentation for proper usage of the function"
            )


class ModelMap:
    def __init__(
        self,
        config: Config = None,
        ticker: str = None,
        model: Optional["Model"] = None,
        time_series_name: str = "MO_RIS_REV",
        col_for_labels: str = "time_series_name",
        tree: bool = True,
        common_size_tree: bool = True,
        notebook: bool = True,
        auto_download: bool = True,
        tree_complexity_limit: Optional[
            int
        ] = None,  # model map will not display as a tree if its complexity is above this limit
        common_time_series: Optional[
            Iterable
        ] = None,  # optional list of common features. nodes with this feature will be made triangles
    ) -> None:
        self.config = config or CONFIG
        self.s3_client = Getter(config=self.config)
        self.api_headers: Dict[str, str] = get_api_headers(self.config.canalyst_api_key)
        self.log = LogFile()

        if model:
            self.model: Optional["Model"] = model
            self.ticker = self.model.ticker
        elif ticker:
            self.ticker = ticker
            self.model = None
        else:
            raise TypeError("You must pass in either a ticker or model")

        self.time_series = time_series_name
        self.col_for_labels = col_for_labels
        self.tree = tree
        self.common_size_tree = common_size_tree
        self.notebook = notebook
        self.auto_download = auto_download
        self.tree_complexity_limit = tree_complexity_limit
        if common_time_series:
            self.common_time_series: Optional[Iterable] = set(common_time_series)
        else:
            self.common_time_series = common_time_series

        # will be defined when create_model is called
        self.dot_file = None
        self.nodes: list = []
        self.path_distances: dict = {}
        self.complexity: Optional[int] = None
        self.mean_end_node_distance: float = 0
        self.max_end_node_distance = 0
        self.std_dev_node_distance = 0

        # Create DataFrame for precedent/dependent tree, graph of the tree, and a network of that graph (for visualization)
        self.df = self.load_data()
        try:
            self.network = self.create_model()
        except Exception as ex:
            print(f"Modelmap error: {ticker}. Error: {ex}")

        self.fig = None

    def load_data(self) -> pd.DataFrame:
        """
        Helper function to load data from candas.

        Loads model_frame for chosen ticker and creates a dot file.
        """
        if self.model:
            cdm = self.model
        else:
            cdm = Model(config=self.config, ticker=self.ticker)

        df = cdm.model_frame(period_duration_type="fiscal_quarter", n_periods="")

        self.dot_file = create_drivers_dot(
            self.time_series,
            self.api_headers,
            self.ticker,
            self.log,
            self.config,
            self.s3_client,
        )

        return df

    def create_model(self, toggle_drag_nodes=True):
        """
        Create a model of the dot file and optionally download in an html file.
        """
        # read dot file into a graph
        G = nx.drawing.nx_pydot.read_dot(self.dot_file)

        # Create Network from nx graph and turn off physics so dragging is easier
        graph = Network(
            "100%",
            "100%",
            notebook=self.notebook,  # enables displaying in a notebook
            directed=True,  # makes edges arrows
            layout=self.tree,  # Creates tree structure
            bgcolor="#FFFFFF",
        )
        graph.toggle_physics(True)
        graph.from_nx(G)

        # update the complexity of the graph, and if it is greater than the limit, set self.Tree to False
        # won't do anything if tree_complexity_limit wasn't set
        self.complexity = len(graph.nodes)
        if self.tree_complexity_limit and self.complexity > self.tree_complexity_limit:
            self.tree = False

        # model_frame filtered with only needed time_series
        time_series = {}
        for node in graph.nodes:
            id_split = node["id"].split("|")
            if len(id_split) > 1:
                time_series[id_split[0]] = id_split[1]

        df = self.df[self.df["time_series_slug"].isin(time_series)]

        # root node id
        root_id = graph.nodes[0]["id"]

        # column to use for labels
        col_for_labels = self.col_for_labels

        # reformat nodes
        # path lengths between each node
        self.path_distances = dict(nx.all_pairs_shortest_path_length(G))
        num_drivers = 0
        distances = []

        for node in graph.nodes:

            # set level of node based on distance from root
            node["level"] = self.path_distances[root_id][node["id"]]

            # parse node id into time series slug and period name
            # AND get row with the given period and time series slug in DataFrame
            id_split = node["id"].split("|")

            ###### SOLVE FOR MODELS CONTAINING MO.Lastprice (which is not in candas DataFrame)
            # account for time series without a period (last price)
            if len(id_split) < 2:
                node["title"] = node["label"]
                continue
            ######

            slug = id_split[0]

            period = id_split[1]
            rows = df[df["period_name"] == period]
            try:
                row = rows[rows["time_series_slug"] == slug].iloc[0]
            except:
                print("ModelMap Error: " + str(node))
                print("period: " + period)
                continue

            # format colors of nodes and update data
            if row["is_driver"]:
                color = "rgba(200, 0, 0, 0.75)"

                ########## STATS #############
                # get dist from root node, add to mean and find new max
                distance_from_root = self.path_distances[root_id][node["id"]]
                distances.append(distance_from_root)
                self.mean_end_node_distance += distance_from_root
                if distance_from_root > self.max_end_node_distance:
                    self.max_end_node_distance = distance_from_root
                num_drivers += 1
                ##########
            else:
                color = "rgba(0, 0, 200, 0.75)"
            node["color"] = color

            # make the node a triangle if the it is one of the common_time_series
            if (
                self.common_time_series
                and row["time_series_name"] in self.common_time_series
            ):
                node["borderWidth"] = 5
                node["shape"] = "triangle"

            # store description, mo_name, ticker, and driver status of time series in node
            node["description"] = str(row["time_series_description"])
            node["time_series_name"] = str(row["time_series_name"])
            node["ticker"] = str(row["ticker"])
            node["is_driver"] = str(row["is_driver"])

            # set the label to be whatever column the user decided to take labels from
            label = str(row[col_for_labels])
            # Format the label to be max n characters per line
            n = 13  # characters per line
            new_label = "\n".join([label[i : i + n] for i in range(0, len(label), n)])
            node["label"] = new_label

            # add value and units and ismm (time_series_description) attributes to node for future use
            node["amount"] = row["value"]
            node["units"] = row["unit_type"]
            node["ismm"] = ", mm" in row["time_series_description"]

            # Add value to node label
            value = ""
            if node["units"] == "currency":
                value = ":\n{}{:,.2f}".format(row["unit_symbol"], row["value"])
            elif node["units"] == "percentage":
                value = ":\n{:.2f}{}".format(row["value"], row["unit_symbol"])
            # elif node["units"] == "count":
            #     value = ":\n{:.2f} {}".format(row["value"], row["unit_symbol"])
            # elif node["units"] == "ratio":
            #     value = ":\n{:.2f} {}".format(row["value"], row["unit_symbol"])
            # elif node["units"] == "time":
            #     value = ":\n{:.2f} {}".format(row["value"], row["unit_symbol"])
            else:
                value = ":\n{:.2f} {}".format(row["value"], row["unit_symbol"])
            node["label"] += value

            # title is label without newlines
            node["title"] = node["label"].replace("\n", "")

        # reformat edges
        for edge in graph.edges:
            temp = edge["from"]
            edge["from"] = edge["to"]
            edge["to"] = temp
            # label the edge with percentages
            if self.common_size_tree:
                precedent = graph.get_node(edge["from"])
                dependent = graph.get_node(edge["to"])

                # WORKAROUND FOR MO.Lastprice ##################
                if "|" not in precedent["id"] or "|" not in dependent["id"]:
                    continue
                ################################################

                if (
                    precedent["units"] == "currency"
                    and precedent["ismm"]
                    and dependent["ismm"]
                    and precedent["level"] != dependent["level"]
                ):
                    if np.isnan(precedent["amount"]):
                        pct = 0
                    else:
                        if dependent["amount"].sum() > 0:
                            pct = precedent["amount"] / dependent["amount"] * 100
                        else:
                            pct = 0
                    edge["label"] = "{:.2f}%".format(pct)

        # reformat tree so desired time series is at the root and it is a different color
        root = graph.get_node(root_id)
        root.update({"color": "rgba(0, 200, 0, 0.75)", "size": 40})

        if self.notebook:
            graph.width, graph.height = "1000px", "1000px"

        # toggles node dragging and disables physics
        graph.toggle_drag_nodes(toggle_drag_nodes)
        graph.toggle_physics(False)

        # make sure models folder exists
        models_folder = os.path.join(
            os.path.dirname(os.path.abspath(__name__)), "models"
        )
        if not os.path.isdir(models_folder):
            os.mkdir(models_folder)

        # create path for model in models folder adjacent to this file
        self.model_path = os.path.join(
            self.config.default_dir,
            f"{self.ticker.split()[0]}_{self.time_series}_model_map.html",
        )

        if self.auto_download:
            graph.write_html(self.model_path, notebook=self.notebook)

        self.nodes = graph.nodes
        self.edges = graph.edges

        # update stats
        self.mean_end_node_distance /= num_drivers
        self.std_dev_node_distance = stdev(distances)

        return graph

    def show(self):
        """
        Shows the model map
        """
        try:
            if os.path.exists(self.model_path):
                if "tmp" in self.model_path or "Temp" in self.model_path:
                    return self.network.show(self.model_path)
                else:
                    return self.network.show(
                        f"{self.ticker.split()[0]}_{self.time_series}_model_map.html"
                    )
            elif os.path.exists(
                os.path.dirname(__main__.__name__),
                f"{self.ticker.split()[0]}_{self.time_series}_model_map.html",
            ):
                return self.network.show(
                    os.path.dirname(__main__.__name__),
                    f"{self.ticker.split()[0]}_{self.time_series}_model_map.html",
                )
            elif os.path.exists(
                f"/canalyst/{self.ticker.split()[0]}_{self.time_series}_model_map.html"
            ):
                return self.network.show(
                    f"/canalyst/{self.ticker.split()[0]}_{self.time_series}_model_map.html"
                )
        except:
            print("ModelMap html file not found.")
            return
        return

    def create_node_df(self):
        """
        Creates a DataFrame of each node's MO_name and distance from root
        """
        graph = self.network

        # root node id
        root_id = graph.nodes[0]["id"]

        # columns for DataFrame
        tickers = []
        time_series_names = []
        distances_to_root = []
        is_driver_col = []
        for node in graph.nodes:
            tickers.append(node["ticker"])
            time_series_names.append(node["time_series_name"])
            distances_to_root.append(self.path_distances[root_id][node["id"]])
            is_driver_col.append(node["is_driver"])

        node_df = pd.DataFrame(
            {
                "ticker": tickers,
                "time_series_name": time_series_names,
                "distance_to_root": distances_to_root,
                "is_driver": is_driver_col,
            }
        )

        return node_df

    def list_time_series(self, search=""):
        """
        Lists out all the nodes
        """
        list_out: List[str] = []
        if self.nodes:
            for node in self.nodes:
                if re.match(f".*{search}.*", node["title"]):
                    list_out.append(node["title"].split(":")[0])
        return list_out

    def time_series_names(self, search=""):
        """
        (Obsolete) Lists out all the nodes

        This method is kept for backwards-compatibility. New code should use list_time_series instead.
        """
        list_out = []
        if self.nodes:
            for node in self.nodes:
                if re.match(f".*{search}.*", node["title"]):
                    list_out.append(node["title"].split(":")[0])
        return list_out

    def create_time_series_chart(self, time_series_name):
        """
        Create a chart of given time series
        """
        # get needed data from DataFrame
        df = self.df[self.df["time_series_name"] == time_series_name]

        if df.shape[0] == 0:
            print("Time series not found")
            return None

        df = df[df["period_name"].str.contains("Q")].sort_values("period_end_date")
        # use subset=['value'] to only drop rows with a null value
        df = df.dropna(subset=["value"])
        # NEED TO ACCOUNT FOR EMPTY DataFrameS
        row1 = df.iloc[0]
        title = row1["time_series_description"]
        xlabel = "Fiscal Quarter"
        ylabel = row1["time_series_description"]
        # plot data
        fig = px.line(
            df,
            x="period_name",
            y="value",
            title=title,
            labels={"period_name": xlabel, "value": ylabel},
        )
        return fig

    def show_time_series_chart(self, time_series_name):
        """
        Shows a chart of the given time series
        """
        fig = self.create_time_series_chart(time_series_name)
        fig.show()
        return


class Model:
    # functions which are CamelCase will become Classes at some point
    def __init__(
        self,
        ticker: str,
        config: Config = None,
        force: bool = False,
        extract_drivers: bool = True,
        company_info: bool = True,
        file_type: str = "parquet",
    ):
        self.file_type = file_type
        self.config = config or CONFIG
        self.ticker = ticker
        self.force = force
        self.company_info = company_info
        self.extract_drivers = extract_drivers
        self.log = LogFile()

        self.api_headers = get_api_headers(self.config.canalyst_api_key)
        self.gt = Getter(self.config)

        self.set_new_uuid()
        self.scenario_url_map: Dict[str, str] = {}

        self.csin: str = ""
        self.latest_version: str = ""
        self._revenue_drivers = None

        if self.force == True:
            print("model_frame")

        if self.company_info == True:
            try:
                self.csin, self.latest_version = self.get_company_info()
            except Exception as ex:
                print(f"ModelSet Error: {self.ticker}. Error: {ex}")
                return
        try:
            self.get_model_frame()

            if self.extract_drivers == True:
                self.apply_drivers()

            if self.force == True:
                print("model_drivers")

            if self.force == True:
                print("guidance")
                self.create_guidance_csv()
        except Exception as ex:
            print(f"Model Error: {self.ticker}. Error: {ex}")
            return

    def time_series_formula(
        self,
        arguments=[],
        modifier="",
        time_series_name="",
        time_series_description="",
    ):
        df = self.model_frame()
        DF = DataFunctions()
        df = DF.time_series_function(
            df,
            arguments,
            modifier,
            time_series_name,
            time_series_description,
        )
        self._model_frame = df  # modify in place
        return

    def model_info(self):
        self.return_string, self.model_info = get_model_info(
            auth_headers=self.api_headers, log=self.log, ticker=self.ticker
        )
        df = pd.DataFrame(self.model_info).T.reset_index()
        df.columns = [
            "ticker",
            "CSIN",
            "name",
            "model_version",
            "historical_periods",
            "update_type",
            "publish_date",
        ]
        self.model_info = df
        return self.model_info

    def key_driver_map(self, time_series_name="MO_RIS_REV"):
        """
        Returns a dataframe of key drivers for a given time series

        Parameters

        time_series_name : str, default "MO_RIS_REV"
        """
        # defaulting to MO_RIS_REV, but you could choose another time_series_name as a starting point like MO_RIS_EBIT
        model_map = self.create_model_map(
            time_series_name=time_series_name,
            col_for_labels="time_series_name",
            notebook=False,
        )

        # list all the nodes in the model map
        model_map_time_series_list = model_map.list_time_series()

        # make it a dataframe because I'm a reformed R user
        df = pd.DataFrame(model_map_time_series_list)

        # columns for joining
        df.columns = ["time_series_name"]
        df["ticker"] = self.ticker

        # create a model frame with only key drivers and just the most recent quarter
        df_drivers = self.model_frame(
            is_driver=True, period_duration_type="fiscal_quarter", mrq=True
        )
        # merge (inner join)
        df = pd.merge(
            df,
            df_drivers,
            how="inner",
            left_on=["ticker", "time_series_name"],
            right_on=["ticker", "time_series_name"],
        )

        return df

    def plot_guidance(self, time_series_name):
        df_guidance = self.guidance()
        df_guidance["time_series_name"] = time_series_name
        df_guidance["period_name"] = df_guidance["Fiscal Period"]
        df_guidance["Actual"] = df_guidance["Output"]
        df_guidance = df_guidance[df_guidance["Item Name"] == time_series_name][
            [
                "time_series_name",
                "period_name",
                "Low",
                "High",
                "Mid",
                "Type.1",
                "Actual",
            ]
        ]
        df_guidance = (
            df_guidance.groupby(["time_series_name", "period_name"])
            .first()
            .reset_index()
        )

        i_low = df_guidance["Low"].sum()
        i_mid = df_guidance["Mid"].sum()
        i_high = df_guidance["High"].sum()

        if len({i_low, i_mid, i_high}) == 1:
            i_three = True
        else:
            i_three = False

        colors = {
            "red": "#ff207c",
            "grey": "#C3C2C3",
            "blue": "#00838F",
            "orange": "#ffa320",
            "green": "#00ec8b",
        }

        plt.rc("figure", figsize=(12, 9))
        plt.title(time_series_name)
        plt.plot(
            df_guidance["period_name"],
            df_guidance["Actual"],
            label="Actual",
            color=colors["blue"],
            linewidth=2,
        )
        plt.plot(
            df_guidance["period_name"],
            df_guidance["Mid"],
            label="Mid",
            color=colors["orange"],
            linewidth=2,
        )

        if i_three == False:
            plt.plot(
                df_guidance["period_name"],
                df_guidance["Low"],
                label="Low",
                color=colors["red"],
                linewidth=2,
            )
            plt.plot(
                df_guidance["period_name"],
                df_guidance["High"],
                label="High",
                color=colors["green"],
                linewidth=2,
            )

        plt.legend()
        plt.show()
        return

    def get_most_recent_model_date(self):
        """
        Get published date for latest workbook model
        """
        csin = self.csin
        auth_headers = self.api_headers
        mds_host = self.config.mds_host
        wp_host = self.config.wp_host

        from python_graphql_client import GraphqlClient

        client = GraphqlClient(endpoint=f"{wp_host}/model-workbooks")

        # Create the query string and variables required for the request.
        query = """
        query driversWorksheetByCSIN($csin: ID!) {
            modelSeries(id: $csin) {
            latestModel {
                id
                name
                publishedAt
                variantsByDimensions(
                    driversWorksheets: [STANDARD_FCF],
                    periodOrder: [CHRONOLOGICAL],
                ) {
                id
                downloadUrl
                variantDimensions {
                    driversWorksheets
                    periodOrder
                }
                }
            }
            }
        }
        """
        variables = {"csin": csin}

        # Synchronous request
        data = client.execute(
            query=query,
            variables=variables,
            headers=auth_headers,
            verify=self.config.verify_ssl,
        )
        url = data["data"]["modelSeries"]["latestModel"]["variantsByDimensions"][0][
            "downloadUrl"
        ]
        file_ticker = self.ticker.split(" ")[0]
        date_name = data["data"]["modelSeries"]["latestModel"]["publishedAt"]
        date_name = date_name.split("T")[0]
        return date_name

    def get_excel_model_name(self):
        """
        Get the model workbook's file name
        """
        csin = self.csin
        auth_headers = self.api_headers
        mds_host = self.config.mds_host
        wp_host = self.config.wp_host

        from python_graphql_client import GraphqlClient

        client = GraphqlClient(endpoint=f"{wp_host}/model-workbooks")

        # Create the query string and variables required for the request.
        query = """
        query driversWorksheetByCSIN($csin: ID!) {
            modelSeries(id: $csin) {
            latestModel {
                id
                name
                publishedAt
                variantsByDimensions(
                    driversWorksheets: [STANDARD_FCF],
                    periodOrder: [CHRONOLOGICAL],
                ) {
                id
                downloadUrl
                variantDimensions {
                    driversWorksheets
                    periodOrder
                }
                }
            }
            }
        }
        """
        variables = {"csin": csin}

        # Synchronous request
        data = client.execute(
            query=query,
            variables=variables,
            headers=auth_headers,
            verify=self.config.verify_ssl,
        )
        url = data["data"]["modelSeries"]["latestModel"]["variantsByDimensions"][0][
            "downloadUrl"
        ]
        file_ticker = self.ticker.split(" ")[0]
        file_name = data["data"]["modelSeries"]["latestModel"]["name"]
        return file_name

    def forecast_frame(
        self, time_series_name, n_periods, function_name="value", function_value=""
    ):
        df = self.model_frame(
            [time_series_name],
            period_duration_type="fiscal_quarter",
            is_historical=False,
            n_periods=n_periods,
        )

        if function_name != "value":
            d = FuncMusic()
            df["new_value"] = df.apply(
                lambda row: d.apply_function(
                    row["value"], modifier=function_name, argument=function_value
                ),
                axis=1,
            )
        else:
            df["new_value"] = function_value

        df = df[
            [
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
                "value",
                "new_value",
                "period_end_date",
            ]
        ]

        return df

    def create_time_series_chart(self, time_series_name, historical=False):  # MODEL
        # get needed data from DataFrame
        df = self._model_frame[
            self._model_frame["time_series_name"] == time_series_name
        ]
        if df.shape[0] == 0:
            print("Time series not found")
            return None

        if historical:
            df = df.loc[df["is_historical"] == historical]

        df = df[df["period_name"].str.contains("Q")].sort_values("period_end_date")
        # use subset=['value'] to only drop rows with a null value
        df = df.dropna(subset=["value"])
        # NEED TO ACCOUNT FOR EMPTY DataFrameS
        row1 = df.iloc[0]
        title = self.ticker + " " + row1["time_series_description"]
        xlabel = "Fiscal Quarter"
        ylabel = row1["time_series_description"]

        colors = {
            "red": "#ff207c",
            "grey": "#C3C2C3",
            "blue": "#00838F",
            "orange": "#ffa320",
            "green": "#00ec8b",
        }

        plt.rc("figure", figsize=(12, 9))
        plt.title(time_series_name)
        plt.plot(
            df["period_name"],
            df["value"],
            label=ylabel,
            color=colors["blue"],
            linewidth=2,
        )
        plt.axvline(x=self.mrq()["period_name"][0], color="black")
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

        return

    def create_model_map(
        self,
        time_series_name: str = "MO_RIS_REV",
        col_for_labels: str = "time_series_name",
        tree: bool = True,
        notebook: bool = False,
        common_time_series_names: Optional[Iterable] = None,
    ) -> ModelMap:
        """
        Create a model map from this model, rooted at the specified time series

        Defaults to a model map rooted at MO_RIS_REV if no time series name is provided
        """
        mm = ModelMap(
            config=self.config,
            model=self,
            time_series_name=time_series_name,
            col_for_labels=col_for_labels,
            tree=tree,
            notebook=notebook,
            common_time_series=common_time_series_names,
        )
        self.model_map = mm
        return mm

    def get_params(self, search_term=""):
        df = self.default_df
        if search_term != "":
            search_term = "(?i)" + search_term
            df1 = df[df["time_series_description"].str.contains(search_term)]
            df2 = df[df["time_series_name"].str.contains(search_term)]
            df = pd.concat([df1, df2])
            df = (
                df.groupby(["time_series_name", "time_series_description"])
                .first()
                .reset_index()
            )
        df = df.sort_values("time_series_name")
        return df[["time_series_name", "time_series_description"]]

    def set_params(self, list_params=[]):
        self.set_new_uuid()
        scenario_name = self.uuid
        list_changes = []
        for param in list_params:
            list_changes.append(
                {
                    "time_series": param["feature_name"],
                    "period": param["feature_period"],
                    "value_expression": {
                        "type": "literal",
                        "value": param["feature_value"],
                    },
                }
            )

        data = {"changes": list_changes, "name": scenario_name}

        response = send_scenario(
            self.ticker,
            data,
            self.api_headers,
            self.log,
            self.csin,
            self.latest_version,
            self.config.mds_host,
            self.config.verify_ssl,
        )

        self.record_scenario_url(response)

        return

    def show_model_map(
        self,
        time_series="MO_RIS_REV",
        tree=True,
        notebook=True,
        common_time_series=None,
    ):
        """
        Display model_map
        """
        mm = ModelMap(
            config=self.config,
            model=self,
            time_series_name=time_series,
            tree=tree,
            notebook=notebook,
            common_time_series=common_time_series,
        )
        self.model_map = mm
        return mm.show()

    def create_guidance_csv(self):
        """
        Write a CSV file containing the model's guidance data

        Parses guidance from the model's Excel workbook
        """
        file_ticker = self.ticker.split(" ")[0]
        path_name = f"{self.config.default_dir}/DATA/{file_ticker}/"
        os.makedirs(path_name, exist_ok=True)

        files = os.listdir(path_name)

        for filename in files:

            if "xlsx" in str(filename):
                try:
                    print("read excel for guidance")
                    df = pd.read_excel(
                        open(f"{path_name}" + filename, "rb"),
                        index_col=False,
                        sheet_name="Guidance",
                        engine="openpyxl",
                    )
                except:
                    print("Read excel for guidance error")
                    return
                try:
                    save_guidance_csv(df, self.ticker, path_name)
                    return
                except:
                    print("Save guidance csv error")

    def set_new_uuid(self):
        import uuid

        self.uuid = str(uuid.uuid4())[:8]

    def mrq(self):
        df = self._model_frame
        df = df[df["is_historical"] == True]
        df = df[~df["period_name"].str.contains("FY")]
        df = df.sort_values("period_end_date", ascending=False)
        return pd.DataFrame(
            df.iloc[0][["period_name", "period_end_date"]]
        ).T.reset_index()[["period_name", "period_end_date"]]

    def guidance(self):
        df = self.gt.get_file(self.ticker, "guidance", file_type="csv")
        df = df.dropna()
        self._guidance = df
        return df

    def get_drivers(self):
        """
        Gets the associated list of drivers for the model.
        """
        drivers = get_drivers_from_api(
            self.config.mds_host,
            self.csin,
            self.latest_version,
            self.api_headers,
            self.log,
            self.config.verify_ssl,
        )

        df_new = pd.json_normalize(drivers)
        df_driver_array = pd.DataFrame(df_new["time_series.names"])
        df_drivers = df_driver_array.explode("time_series.names")
        df = df_drivers.groupby("time_series.names").first().reset_index()
        df.columns = ["time_series_name"]

        self._model_drivers = df

    def get_company_info(self):
        """
        Get CSIN and latest version for a ticker
        """
        return get_company_info_from_ticker(
            self.ticker, self.api_headers, self.config.mds_host
        )

    def get_name_index(self, force=False):

        self.df_name_index = get_name_index_from_csv(self.ticker, self.config)

        ticker = self.ticker.split(" ")[0]

        self.df_name_index.to_csv(
            f"{self.config.default_dir}/DATA/{ticker}/{self.ticker}_name_index.csv",
            index=False,
        )
        return

    def revenue_drivers(self, search_term="", is_driver=False):
        pd.set_option("display.max_colwidth", None)
        df = self._model_frame
        df_only_drivers = self._revenue_drivers
        df = pd.merge(
            df,
            df_only_drivers,
            how="inner",
            left_on="time_series_name",
            right_on="time_series_name_dependent",
        )
        df1 = df[
            df["time_series_description"].str.contains(
                search_term, flags=re.IGNORECASE, regex=True
            )
        ]
        df2 = df[
            df["time_series_name"].str.contains(
                search_term, flags=re.IGNORECASE, regex=True
            )
        ]
        df = pd.concat([df1, df2])
        df = df.dropna()
        df = (
            df.groupby(["time_series_name", "time_series_description"])
            .first()
            .reset_index()
        )
        df = df.drop(
            columns=[
                "period_duration_type",
                "category_type_slug",
                "time_series_slug",
                "category_type_name",
                "category_slug",
                "is_historical",
                "value",
                "period_start_date",
                "period_end_date",
                "period_name",
                "time_series_name",
                "time_series_description",
            ]
        )
        df = df.sort_values("index")
        df = df.reset_index()
        df = df.drop(columns=["index", "level_0"])
        if is_driver == True:
            df = df.loc[df["is_driver"] == True]
            return df
        return df

    def model_drivers(self, search_term=""):
        mrq = self.mrq()["period_name"][0]
        df = self._model_drivers
        df2 = self._model_frame
        df2 = df2.loc[df2["period_name"] == mrq]
        df = pd.merge(
            df,
            df2,
            how="inner",
            left_on="time_series_name",
            right_on="time_series_name",
        )[
            ["name_index", "category", "time_series_name", "time_series_description"]
        ].sort_values(
            "name_index", ascending=True
        )

        if search_term != "":
            df = df[
                df["time_series_name"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
        return df

    # MODEL model_frame
    def model_frame(
        self,
        time_series_name="",
        period_name="",
        is_driver="",
        pivot=False,
        mrq=False,
        period_duration_type="",
        is_historical="",
        n_periods=36,
        mrq_notation=False,
        unit_type="",
        category="",
        warning=True,
    ):
        if self.csin == "":
            print("Error: CSIN not found")
            return

        return filter_dataset(
            self._model_frame,
            time_series_name,
            period_name,
            is_driver,
            pivot,
            mrq,
            period_duration_type,
            is_historical,
            n_periods,
            mrq_notation,
            unit_type,
            category,
            warning,
        )

    def apply_drivers(self):

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

        df_index = get_data_set_from_mds(
            BulkDataKeys.NAME_INDEX,
            self.file_type,
            self.csin,
            self.latest_version,
            self.api_headers,
            self.log,
            settings.MDS_HOST,
            self.config.verify_ssl,
        )

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

    def get_model_frame(self):
        df_hist = self.historical_data_frame()

        df_hist = df_hist.assign(is_historical=True)
        df_hist = df_hist[~df_hist["period_name"].isna()]

        mrq = df_hist["period_name"].iloc[0]
        pd.options.mode.chained_assignment = None

        df_fwd = self.forward_data_frame()

        df_fwd = df_fwd.assign(is_historical=False)
        df_fwd = df_fwd[~df_fwd["period_name"].isna()]

        df_concat = pd.concat([df_hist, df_fwd]).sort_values("period_name")

        self._model_frame = df_concat

        mrq = self.mrq()["period_name"][0]
        self._model_frame["MRFQ"] = [mrq for i in range(len(self._model_frame))]

        self._model_frame["period_name_sorted"] = np.where(
            self._model_frame["period_name"].str.contains("-"),
            self._model_frame["period_name"].str.split("-").str[1]
            + self._model_frame["period_name"].str.split("-").str[0],
            self._model_frame["period_name"].str[2:6]
            + self._model_frame["period_name"].str[0:2],
        )
        self._model_frame["CSIN"] = self.csin
        return

    def feature_search(self, search_term=""):
        if search_term != "":
            df = self._model_frame
            df1 = df[
                df["time_series_description"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
            df2 = df[
                df["time_series_name"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
            df = pd.concat([df1, df2])
            df = (
                df.groupby(["time_series_name", "time_series_description"])
                .first()
                .reset_index()
            )
            df = df[["time_series_name", "time_series_description"]]
        return df

    def driver_search(self, search_term=""):
        """
        Search modelset drivers
        """
        df = self._model_drivers
        if search_term != "":
            df = self._model_frame
            df_only_drivers = self._model_drivers
            df_drivers = pd.merge(
                df,
                df_only_drivers,
                how="inner",
                left_on="time_series_name",
                right_on="time_series_name",
            )
            df1 = df[
                df["time_series_description"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
            df2 = df[
                df["time_series_name"].str.contains(
                    search_term, flags=re.IGNORECASE, regex=True
                )
            ]
            df = pd.concat([df1, df2])
            df = (
                df.groupby(["time_series_name", "time_series_description"])
                .first()
                .reset_index()
            )
            df = df[["time_series_name", "time_series_description"]]
        return df

    def describe_drivers(self, filter_string=""):
        """
        Get a summary DataFrame of the model's drivers
        """
        # if self.describe_drivers is not None:
        #    return self.describe_drivers
        # get the historical DataFrame from candas
        df = self.historical_data_frame()

        df_only_drivers = self._model_drivers
        # merge the drivers and the historical data... and return a description of the values for each
        df_drivers = pd.merge(
            df,
            df_only_drivers,
            how="inner",
            left_on="time_series_name",
            right_on="time_series_name",
        )
        df_drivers_summary = (
            df_drivers.groupby(["time_series_name"]).describe().reset_index()
        )
        if filter_string != "":
            self._describe_drivers = df_drivers_summary
            return df_drivers_summary[
                df_drivers_summary["time_series_name"].str.contains(
                    filter_string, case=False
                )
            ]
        self._describe_drivers = df_drivers_summary
        return df_drivers_summary

    def historical_data_frame(self):  # if cache = true save locally?
        """
        Get a DataFrame of the model's historical data
        """
        df = get_data_set_from_mds(
            BulkDataKeys.HISTORICAL_DATA,
            self.file_type,
            self.csin,
            self.latest_version,
            self.api_headers,
            self.log,
            self.config.mds_host,
            self.config.verify_ssl,
        )
        if df is not None:
            return df

    def summary(self, filter_term=""):
        pd.set_option("mode.chained_assignment", None)
        pd.set_option("display.float_format", lambda x: "%.5f" % x)

        df = pd.merge(
            self.model_frame(
                period_duration_type="fiscal_quarter",
                is_historical=False,
                warning=False,
            ),
            self.scenario_df,
            how="inner",
            left_on=[
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
            ],
            right_on=[
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
            ],
        )[
            [
                "ticker",
                "period_name",
                "time_series_name",
                "time_series_description",
                "value_x",
                "value_y",
            ]
        ]
        df.columns = [
            "ticker",
            "period_name",
            "time_series_name",
            "time_series_description",
            "default",
            "scenario",
        ]
        df["diff"] = df["scenario"].astype(float) / df["default"].astype(float)
        if filter_term != "":
            df = df[df["time_series_name"].str.contains(filter_term, case=False)]
        return df

    def forward_data_frame(self):
        if self.force == False:
            df = get_data_set_from_mds(
                BulkDataKeys.FORECAST_DATA,
                self.file_type,
                self.csin,
                self.latest_version,
                self.api_headers,
                self.log,
                self.config.mds_host,
                self.config.verify_ssl,
            )
            if df is not None:
                self.default_df = df
                return df

        file_ticker = self.ticker.split(" ")[0]
        path_name = f"{self.config.default_dir}/DATA/{file_ticker}/"
        file_name = f"{path_name}{self.ticker}_forecast_data.csv"

        if not os.path.exists(path_name):
            os.makedirs(path_name)

        if self.csin == "":
            self.csin, self.latest_version = get_company_info_from_ticker(
                self.ticker,
                self.api_headers,
                self.log,
                self.config.mds_host,
                self.config.verify_ssl,
            )

        url = get_forecast_url(self.csin, self.latest_version, self.config.mds_host)

        res = get_request(url, self.api_headers, self.log, self.config.verify_ssl)

        list_out = []

        for res_dict in res.json()["results"]:
            df = get_forecast_url_data(
                res_dict,
                self.ticker,
                self.api_headers,
                self.log,
                self.config.verify_ssl,
            )
            list_out.append(df)

        self.default_df = pd.concat(list_out)
        self.default_df.to_csv(file_name, index=False)

        return df

    def model_fit(self, time_series_name=""):

        if self.csin == "":
            self.csin, self.latest_version = get_company_info_from_ticker(
                self.ticker,
                auth_headers=self.api_headers,
                logger=self.log,
                mds_host=self.config.mds_host,
                verify_ssl=self.config.verify_ssl,
            )

        url = get_forecast_url(self.csin, self.latest_version, self.config.mds_host)

        scenario_url = (
            f"{self.config.mds_host}/"
            f"{SCENARIO_URL.format(csin=self.csin, version=self.latest_version)}"
        ) + "?page_size=200"

        if self.scenario_url_map.get(self.uuid) is None:
            scenario_response = get_request(
                scenario_url,
                headers=self.api_headers,
                logger=self.log,
                verify_ssl=self.config.verify_ssl,
            )
            scenario_json = scenario_response.json()
            scenario_id_url = map_scenario_urls(scenario_json).get(self.uuid)
        else:
            scenario_id_url = self.scenario_url_map[self.uuid]

        if scenario_id_url is None:

            print("Scenario ID for " + scenario_url + " is None.")

            return

        print(self.ticker + " scenario_id_url: " + str(scenario_id_url))

        if time_series_name != "":

            url = scenario_id_url + "time-series/?name=" + time_series_name

            res_loop = get_request(
                url, self.api_headers, self.log, self.config.verify_ssl
            )
            url = res_loop.json()["results"][0]["self"]
            res_loop = get_request(
                url, self.api_headers, self.log, self.config.verify_ssl
            )
            url = res_loop.json()["forecast_data_points"]
            res_loop = get_request(
                url, self.api_headers, self.log, self.config.verify_ssl
            )
            res_data = res_loop.json()["results"]
            dict_out = {}
            list_out = []
            for res_data_dict in res_data:
                dict_out["time_series_slug"] = res_data_dict["time_series"]["slug"]
                dict_out["time_series_name"] = res_data_dict["time_series"]["names"][0]
                dict_out["time_series_description"] = res_data_dict["time_series"][
                    "description"
                ]
                dict_out["category_slug"] = res_data_dict["time_series"]["category"][
                    "slug"
                ]
                dict_out["category_type_slug"] = res_data_dict["time_series"][
                    "category"
                ]["type"]["slug"]
                dict_out["category_type_name"] = res_data_dict["time_series"][
                    "category"
                ]["type"]["name"]
                dict_out["unit_description"] = res_data_dict["time_series"]["unit"][
                    "description"
                ]
                dict_out["unit_symbol"] = res_data_dict["time_series"]["unit"]["symbol"]
                dict_out["period_name"] = res_data_dict["period"]["name"]
                dict_out["period_duration_type"] = res_data_dict["period"][
                    "period_duration_type"
                ]
                dict_out["period_start_date"] = res_data_dict["period"]["start_date"]
                dict_out["period_end_date"] = res_data_dict["period"]["end_date"]
                dict_out["value"] = res_data_dict["value"]
                dict_out["ticker"] = self.ticker
                df = pd.DataFrame.from_dict(dict_out, orient="index").T
                list_out.append(df)
            self.scenario_df: DataFrame = pd.concat(list_out)
            return

        scenario_id_url = scenario_id_url + "forecast-periods/"

        scenario_response = get_request(
            scenario_id_url,
            headers=self.api_headers,
            logger=self.log,
            verify_ssl=self.config.verify_ssl,
        )
        all_df = []

        all_df = Parallel(n_jobs=num_cores)(
            delayed(get_scenario_url_data)(
                res_dict,
                self.ticker,
                self.api_headers,
                verify_ssl=self.config.verify_ssl,
            )
            for res_dict in scenario_response.json()["results"]
        )
        self.scenario_df = pd.concat(all_df)
        print("Done")
        return

    def record_scenario_url(self, scenario_response: Optional[Response]) -> None:

        if scenario_response is None:
            return None

        if not scenario_response.ok:
            raise ScenarioException(scenario_response.status_code)

        """
        Extract scenario name and location from the response and add to scenario url dictionary
        """
        scenario_name = self.uuid
        scenario_url = scenario_response.headers["Location"]

        if self.scenario_url_map.get(scenario_name) is None:
            self.scenario_url_map[scenario_name] = scenario_url


class Chart:
    """
    Charting class to create consistent and branded charts

    Attributes
    ----------
    x_value : pd.Series
        the data for the x-axis
    y_values : pd.DataFrame
        the data for the y-axis
    labels : list
        a list of strings to specify labels of y_value columns respectively (Default: None)
    axis_labels : list
        a list of tuples of strings to specify x,y axis labels (Default: None)
    title : str
        the title of the chart (Default: None)
    plot_styles : list
        list of chart styles for y-value columns (can specify only 1 style to apply to all columns) (Default: ["line"])
    use_subplots : bool
        bool to indicate if subplots should be used (i.e chart columns on the same graph or not) (Default: False)
    subplot_scale_width : list
        list of ints to specify width scaling for subplots (length of list must match # of subplots) (Default: None)
    subplot_scale_height : list
        list of ints to specify height scaling for subplots (length of list must match # of subplots) (Default: None)
    figure_size : tuple
        tuple to define figure size (Default: (12,9))
    vertical_line : str or pd.Series
        str or DataFrame series to define where to place a vertical line on the graph on the x-axis (Default: None)
    marker : str
        str to define marker type (Default: "o")
    markersize : int
        the size of markers (Default: 8)
    markevery : int or float
        int argument will define plotting of every nth marker from the first data point.  (Default: None)
    plot_config : dict
        extra arguments to be passed into the .plot() command. Arguments must be valid for the plot style being used. (Default: None)
    xtick_rotation : int
        specifies the rotation of x-axis tick labels (Default: 90)
    display_charts_horizontal : bool
        specifies if subplots should be displayed horizontally instead of vertically (Default: False)
    subplot_adjustment : list
        list in the form of [left, bottom, right, top] to tune subplot layout. (Default: [0.125, 0.2, 0.9, 0.9])

    Methods
    -------
    show()
        Displays the graph generated
    build_chart(force)
        Builds a matplotlib chart. if force is set to True, chart will be built from scratch regardless if cache is present.
    """

    def __init__(
        self,
        x_value,
        y_values,
        labels=None,
        axis_labels=None,
        title="",
        plot_styles=[
            "line"
        ],  # If only 1 style defined, it will be applied to all columns
        use_subplots=False,
        subplot_scale_width=None,
        subplot_scale_height=None,
        figure_size=(12, 9),
        vertical_line=None,
        marker="o",
        markersize=8,
        markevery=None,
        plot_config={},
        xtick_rotation=90,
        display_charts_horizontal=False,
        subplot_adjustment=[0.125, 0.2, 0.9, 0.9],
        brand_config=None,
        include_logo_watermark=True,
    ):

        self.x_value = x_value
        self.y_values = y_values
        self.labels = labels
        self.axis_labels = axis_labels
        self.title = title
        self.use_subplots = use_subplots
        self.plot_styles = plot_styles
        self.subplot_scale_width = subplot_scale_width
        self.subplot_scale_height = subplot_scale_height
        self.figure_size = figure_size
        self.vertical_line = vertical_line
        self.markevery = markevery
        self.marker = marker
        self.markersize = markersize
        self.plt_plot_style = "fivethirtyeight"
        self.plot_config = plot_config
        self.xtick_rotation = xtick_rotation
        self.display_charts_horizontal = display_charts_horizontal
        self.subplot_adjustment = subplot_adjustment
        self.include_logo_watermark = include_logo_watermark

        self.brand_config = brand_config or BRAND_CONFIG_DEFAULTS

        self._validate_brand_config()
        self._validate_args()

        self.main_fpath = Path(self.brand_config["title_font_path"])
        self.secondary_fpath = Path(self.brand_config["body_font_path"])

        plt.rc("figure", figsize=self.figure_size)
        self.fig, self.axs = self.build_chart()

    def _validate_brand_config(self):
        try:
            self.brand_config["title_font_path"]
        except KeyError:
            raise KeyError(
                "Title font path not found in chart brand configuration. Please reset or update chart brand configs"
            )

        try:
            self.brand_config["chart_bg_color"]
        except KeyError:
            raise KeyError(
                "Chart background color not found in chart brand configuration. Please reset or update chart brand configs"
            )

        try:
            self.brand_config["body_font_path"]
        except KeyError:
            raise KeyError(
                "Body font path not found in chart brand configuration. Please reset or update chart brand configs"
            )

        try:
            self.brand_config["logo_path"]
        except KeyError:
            raise KeyError(
                "Logo path not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["figure_bg_color"]
        except KeyError:
            raise KeyError(
                "Figure background color not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["font_color"]
        except KeyError:
            raise KeyError(
                "Font color not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["vertical_line_color"]
        except KeyError:
            raise KeyError(
                "Vertical line color not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["chart_plot_colors"]
        except KeyError:
            raise KeyError(
                "First chart plot color not found in configuration. Please reset or update configs."
            )

        try:
            self.brand_config["ax_spine_color"]
        except KeyError:
            raise KeyError(
                "Ax spine color not found in configuration. Please reset or update configs."
            )

    def _validate_args(self):
        """
        Validate arguments used in initialization
        """
        if not type(self.plot_styles) == list:
            print(
                "Plot styles must be specified in the form of a list. Please re-instantiate the "
                'object with plot style in the form of "["plot_style_here"]". Available options '
                'are "line", "bar", "barh", "scatter".'
            )
        if not type(self.labels) == list:
            print(
                "Labels must be specified in the form of a list. Please re-instantiate the object"
                'with labels in the form of "["Label", "Label"]" depending on how many columns'
                "are being passed in."
            )

        if isinstance(self.y_values, pd.DataFrame):
            if type(self.labels) == list and not len(self.labels) == len(
                self.y_values.columns
            ):
                print("Warning: Labels have not been appointed for all columns.")

        if isinstance(self.y_values, pd.Series):
            if type(self.plot_styles) == list and len(self.plot_styles) > 1:
                print(
                    "Only 1 plot style needed for the specified y data. Only the first value "
                    "in the list will be used."
                )
        elif isinstance(self.y_values, pd.DataFrame):
            columns = len(self.y_values.columns)
            # If user has specified only 1 plot style for multiple columns, extend it to all columns
            if type(self.plot_styles) == list and len(self.plot_styles) == 1:
                self.plot_styles = self.plot_styles * columns
            elif (
                not type(self.plot_styles) == list
                or not len(self.plot_styles) == columns
            ):
                print(
                    "Please explicitly specify the plot style of all Y value columns in a list "
                    "or 1 style to be applied for all."
                )

    def _add_watermark(self, ax):
        """
        Adds a Canalyst Logo to the top left corner of the graph
        """

        logo_path = self.brand_config["logo_path"]

        logo = image.imread(logo_path)
        starting_x, starting_y, width, height = ax.get_position().bounds

        # Units seem to be in inches:
        # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/figure_size_units.html#:~:text=The%20native%20figure%20size%20unit,units%20like%20centimeters%20or%20pixels.
        logo_width = 0.06
        logo_height = 0.06

        # Change the numbers in this array to position your image [left, bottom, width, height])
        ax = plt.axes(
            [
                starting_x + 0.01,
                starting_y + height - logo_height,
                logo_width,
                logo_height,
            ],
            frameon=True,
        )
        ax.imshow(logo)
        ax.axis("off")

    def setup_basic_chart_options(self, fig):
        """
        Sets up figure with basic chart options
        """
        plt.style.use(self.plt_plot_style)

        fig_bg_color = self.brand_config["figure_bg_color"]

        fig.patch.set_facecolor(fig_bg_color)
        # Adjust subplot chart display positions
        left, bottom, right, top = self.subplot_adjustment
        plt.subplots_adjust(left, bottom, right, top)

        font_color = self.brand_config["font_color"]

        mpl.rcParams["text.color"] = font_color
        mpl.rcParams["axes.labelcolor"] = font_color
        mpl.rcParams["xtick.color"] = font_color
        mpl.rcParams["ytick.color"] = font_color

    def build_chart(self, force=False):
        """
        Builds a matplotlib chart. Returns figure and ax(s).
        """

        # Avoid rebuilding if not required
        try:
            if self.fig and self.axs and not force:
                print(
                    f"Skipping re-build as the chart is already built. "
                    f"Please provide a `force=True` argument to force the rebuilding of the chart"
                )
                return self.fig, self.axs
        except AttributeError:
            pass

        if isinstance(self.y_values, pd.Series):
            fig, ax = self._get_chart_without_subplots(is_series=True)
        elif isinstance(self.y_values, pd.DataFrame):
            if not self.use_subplots:
                fig, ax = self._get_chart_without_subplots(is_series=False)
            else:
                # Use subplots
                fig, ax = self._get_chart_with_subplots()

        # If a vertical line is specified, and there are subplots: apply it on all subplots
        # If not, apply it to just the one graph.
        vertical_line_color = self.brand_config["vertical_line_color"]

        if self.vertical_line:
            if type(ax) == np.ndarray:
                for index, item in enumerate(ax):
                    ax[index].axvline(
                        x=self.vertical_line,
                        color=vertical_line_color,
                    )
            else:
                ax.axvline(
                    x=self.vertical_line,
                    color=vertical_line_color,
                    linewidth=2,
                )

        return fig, ax

    def _get_axis_label(self, index):
        """
        Return axis label at position specified.

        Returns None if class not initialized with axis_labels.
        """
        if self.axis_labels is not None:
            try:
                return self.axis_labels[index]
            except IndexError:
                pass

        return None

    def _get_label(self, index):
        """
        Return label at position specified.

        Returns None if class not initialized with labels.
        """
        if self.labels is not None:
            try:
                return self.labels[index]
            except IndexError:
                pass
        return None

    def _get_chart_without_subplots(self, is_series=True):
        """
        Returns figure and ax for charts without subplots for series or DataFrames
        """
        fig, ax = plt.subplots()
        first_element = 0

        font_color = self.brand_config["font_color"]

        # Get and set custom font
        fig.suptitle(
            self.title,
            font=self.main_fpath,
            size="x-large",
            weight="bold",
            y=0.96,
            color=font_color,
        )

        self.setup_basic_chart_options(fig)

        label = self._get_label(first_element)
        axis_label = self._get_axis_label(first_element)

        first_chart_plot_color = self.brand_config["chart_plot_colors"][0]

        # If series is specified, we only have 1 y-value to graph
        if is_series:
            self._set_graph_ax(
                ax,
                self.plot_styles[0],
                self.x_value,
                self.y_values,
                label,
                axis_label,
                first_chart_plot_color,
                self.marker,
                self.markersize,
                self.markevery,
            )
        else:
            # Want to graph multiple y-values on the same graph
            color_count = len(self.brand_config["chart_plot_colors"])
            for index, column in enumerate(self.y_values):
                label = self._get_label(index)
                self._set_graph_ax(
                    ax,
                    self.plot_styles[index],
                    self.x_value,
                    self.y_values[column],
                    label,
                    axis_label,
                    self.brand_config["chart_plot_colors"][
                        index % color_count
                    ],  # Use index modded by length of color_options to select unique colors, up to color_count, then colors repeat.
                    self.marker,
                    self.markersize,
                    self.markevery,
                )

        self._format_graph(ax, fig)
        return fig, ax

    def _get_scale_arguments(self):
        """
        Return width, height scale arguments, if any
        """
        scale_args = {}

        if self.subplot_scale_width and self.subplot_scale_height:
            scale_args = {
                "gridspec_kw": {
                    "width_ratios": self.subplot_scale_width,
                    "height_ratios": self.subplot_scale_height,
                }
            }
        elif self.subplot_scale_width:
            scale_args = {"gridspec_kw": {"width_ratios": self.subplot_scale_width}}
        elif self.subplot_scale_height:
            scale_args = {"gridspec_kw": {"height_ratios": self.subplot_scale_height}}

        return scale_args

    def _get_chart_with_subplots(self):
        """
        Returns figure and ax for charts with subplots for DataFrames
        """
        color_count = len(self.brand_config["chart_plot_colors"])
        columns = len(self.y_values.columns)
        scale_args = self._get_scale_arguments()

        # If specified, display subplots horizontally
        if columns % 2 == 0 and self.display_charts_horizontal:
            row = int(columns / 2)
            row_column = {"nrows": row, "ncols": columns}
        else:
            row_column = {"nrows": columns}

        fig, axs = plt.subplots(
            **row_column,
            **scale_args,
            sharex=True,
        )

        font_color = self.brand_config["font_color"]

        # Set custom Font and figure title
        fig.suptitle(
            self.title,
            font=self.main_fpath,
            size="x-large",
            weight="bold",
            y=0.96,
            color=font_color,
        )

        self.setup_basic_chart_options(fig)

        # Plot each column from the Y-column dataset provided
        for index, column in enumerate(self.y_values):
            label = self._get_label(index)
            axis_label = self._get_axis_label(index)

            self._set_graph_ax(
                axs[index],
                self.plot_styles[index],
                self.x_value,
                self.y_values[column],
                label,
                axis_label,
                self.brand_config["chart_plot_colors"][index % color_count],
                self.marker,
                self.markersize,
                self.markevery,
            )
            self._format_graph(axs[index], fig)

        return fig, axs

    def _set_graph_ax(
        self,
        ax,
        plot_style,
        x_value,
        y_value,
        label,
        axis_labels,
        color,
        marker,
        markersize,
        markevery,
    ):
        """
        Builds a matplotlib chart on the ax provided based on configurations specified.
        """
        font_color = self.brand_config["font_color"]

        # Set up axis labels
        if axis_labels is not None:
            x_label, y_label = axis_labels
            ax.set_xlabel(
                x_label,
                font=self.main_fpath,
                labelpad=15,
                color=font_color,
            )
            ax.set_ylabel(
                y_label,
                font=self.main_fpath,
                labelpad=15,
                color=font_color,
            )

        # Plot the type of bar specified
        if plot_style == "bar":
            ax.bar(
                x_value,
                y_value,
                label=label,
                color=color,
                width=0.4,
                zorder=-1,
                **self.plot_config,
            )
        elif plot_style == "scatter":
            ax.scatter(
                x_value,
                y_value,
                label=label,
                color=color,
                zorder=3,
            )
        elif plot_style == "barh":
            ax.barh(
                x_value,
                y_value,
                label=label,
                color=color,
                zorder=-2,
                **self.plot_config,
            )
        else:  # line graph or catch all
            ax.plot(
                x_value,
                y_value,
                label=label,
                color=color,
                marker=marker,
                markersize=markersize,
                markevery=markevery,
                linewidth=2,
                zorder=1,
                **self.plot_config,
            )

    def _format_graph(self, ax, fig):
        """
        Formats ax's
        """
        ax_spine_color = self.brand_config["ax_spine_color"]

        chart_bg_color = self.brand_config["chart_bg_color"]

        # General grid and chart options
        ax.grid(linewidth=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(ax_spine_color)
        ax.spines["bottom"].set_color(ax_spine_color)
        ax.set_facecolor(chart_bg_color)

        # Setup Legend
        columns = 1
        if isinstance(self.y_values, pd.DataFrame):
            columns = len(self.y_values.columns)

        # Set custom Font
        font_name = fm.FontProperties(fname=self.secondary_fpath)

        fig.legend(
            loc="lower left",
            bbox_to_anchor=(0.1, 0),
            fancybox=True,
            shadow=True,
            ncol=columns,
            frameon=False,
            prop=font_name,
        )

        font_color = self.brand_config["font_color"]

        # Set up x,y axis tick labels
        config_ticks = {
            "size": 0,
            "labelcolor": font_color,
            "labelsize": 12,
            "pad": 10,
        }
        ax.tick_params(axis="both", **config_ticks)

        if self.include_logo_watermark:
            # Add Canalyst watermark
            self._add_watermark(ax)

        for tick in ax.get_xticklabels():
            tick.set_font(self.secondary_fpath)
        for tick in ax.get_yticklabels():
            tick.set_font(self.secondary_fpath)

        # Set the rotation for tick labels
        for tick in ax.get_xticklabels():
            tick.set_rotation(self.xtick_rotation)

    def show(self):
        """
        Displays Chart
        """
        plt.show()


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
