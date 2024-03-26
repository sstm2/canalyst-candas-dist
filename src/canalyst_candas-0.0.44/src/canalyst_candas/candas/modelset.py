"""
a class of multiple models

NOTE: prevent the possibility of having circular imports
by avoiding direct dependency on other core classes
"""

from functools import reduce
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

from canalyst_candas.datafunctions import DataFunctions
from canalyst_candas.candas.model import Model
from canalyst_candas.candas.chart import Chart
from canalyst_candas.candas.func_music import FuncMusic
from canalyst_candas.configuration.config import Config, create_config
from canalyst_candas.exceptions import ScenarioException
from canalyst_candas.utils.logger import LogFile
from canalyst_candas.utils.transformations import (
    df_filter,
    filter_dataset,
)
from canalyst_candas.utils.requests import (
    get_api_headers,
    send_scenario,
    check_time_series_name,
)

plt.style.use("fivethirtyeight")


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
        allow_nulls: bool = True,
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
        self.config = config or create_config()
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
        notebook=True,
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
        notebook : boolean, default True
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

        df = check_time_series_name(
            df, time_series_name, self.api_headers, self.log, self.config.verify_ssl
        )

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
                self.config.proxies,
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
