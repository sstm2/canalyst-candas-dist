import io
import string
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from canalyst_candas.configuration.config import Config, create_config
from canalyst_candas.utils.requests import Getter


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
        self.config: Config = config or create_config()
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
