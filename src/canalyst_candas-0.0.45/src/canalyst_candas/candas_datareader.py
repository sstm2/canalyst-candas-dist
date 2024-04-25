# from canalyst_candas.configuration.config import Config

# from configuration.config import resolve_config
# from canalyst_candas.utils.transformations import calendar_quarter, df_filter
import yahoo_fin.stock_info as si
import pandas as pd
import numpy as np
import string
import statsmodels.tsa.stattools as ts
from statsmodels.regression.rolling import RollingOLS

import statsmodels.api as ssm
import statsmodels.formula.api as smf

from fredapi import Fred
import pandas as pd
import json

import os
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")

from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from urllib.request import urlopen, Request

import warnings


# from nltk.sentiment.vader import SentimentIntensityAnalyzer

web_url = "https://finviz.com/quote.ashx?t="


def get_news_sentiment(ticker_list):
    news_tables = {}

    for tick in ticker_list:
        print(tick)
        url = web_url + tick
        req = Request(url=url, headers={"User-Agent": "Chrome"})
        try:
            response = urlopen(req)
        except:
            continue
        html = BeautifulSoup(response, "html.parser")
        news_table = html.find(id="news-table")
        news_tables[tick] = news_table

    news_list = []

    for file_name, news_table in news_tables.items():
        for i in news_table.findAll("tr"):

            text = i.a.get_text()

            date_scrape = i.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]

            tick = file_name.split("_")[0]

            news_list.append([tick, date, time, text])

    vader = SentimentIntensityAnalyzer()  # type: ignore

    columns = ["ticker", "date", "time", "headline"]

    news_df = pd.DataFrame(news_list, columns=columns)

    scores = news_df["headline"].apply(vader.polarity_scores).tolist()

    scores_df = pd.DataFrame(scores)

    news_df = news_df.join(scores_df, rsuffix="_right")

    news_df["date"] = pd.to_datetime(news_df.date).dt.date

    return news_df


def get_betas_wide(betas_list):
    list_df = []
    for item in betas_list:

        df = get_price_data(item)
        list_df.append(df)

    df_long = pd.concat(list_df, axis=0).reset_index()
    df_wide = df_long.pivot_table(
        index=["index"], columns="ticker", values="pct_change"
    )
    return df_wide


def get_parallel_betas_from_list(ticker_list, betas_list):
    df_wide = get_betas_wide(betas_list)
    from joblib import Parallel, delayed
    import multiprocessing

    inputs = range(10)

    def processInput(i):
        return i * i

    num_cores = multiprocessing.cpu_count()
    all_df = Parallel(n_jobs=num_cores)(
        delayed(get_beta_data)(ticker, df_wide, betas_list) for ticker in ticker_list
    )
    df_betas = pd.concat(all_df)
    return df_betas


def get_beta_data(ticker, df_wide, betas_list):
    list_out = []
    print(ticker)
    try:
        df = get_price_data(ticker).reset_index()
    except:
        print("failed")
        return

    dataset = pd.merge(df, df_wide, how="inner", left_on="index", right_on="index")

    beta_lens = [60, 90, 252]  # window for betas
    dict_lens = {}
    for beta_len in beta_lens:
        betas = {}
        ns = {}
        r2s = {}
        for col in dataset.columns:
            if col in betas_list:

                df = dataset[["pct_change", col]].dropna()
                # try:
                beta, r2, n = get_betas(df["pct_change"], df[col], beta_len)
                # except:
                #    print("get_betas error")
                #    return
                betas[col] = beta
                ns[col] = n
                r2s[col] = r2
        dict_lens[beta_len] = [betas, ns, r2s]

        df1 = pd.DataFrame.from_dict(
            dict_lens[beta_len][2], orient="index"
        ).reset_index()

        try:
            df1.columns = ["BETA", ticker]
        except:
            print("df columns error")
            return
        df1 = df1.T
        headers = df1.iloc[0]
        df1 = pd.DataFrame(df1.values[1:], columns=headers)
        df1["ticker"] = ticker
        df1["beta_days"] = beta_len
        list_out.append(df1)
    df = pd.concat(list_out)
    return df


def get_price_data(
    ticker, reset=False, rolling_betas=True, rolling_window=60, index_ticker="^GSPC"
):
    price_data = si.get_data(ticker, start_date="01/01/2009")
    df = pd.DataFrame(price_data)
    df = df[["adjclose"]]
    df["pct_change"] = df.adjclose.pct_change()
    df["log_return"] = np.log(1 + df["pct_change"].astype(float))
    df["ticker"] = ticker
    if reset == True:
        df = df.reset_index()
        df["price_date"] = df["index"]
        df = df.drop(columns=["index"])
    if rolling_betas == True:
        beta = "beta_" + str(rolling_window)
        df["adjclose_plus_1"] = df["adjclose"].shift(-1)
        df["adjclose_plus_5"] = df["adjclose"].shift(-5)
        df["adjclose_plus_10"] = df["adjclose"].shift(-10)

        sp_ticker = index_ticker
        sp_data = si.get_data(sp_ticker, start_date="01/01/2009")
        df_sp = pd.DataFrame(sp_data)
        df_sp = df_sp[["adjclose"]]
        df_sp["pct_change"] = df_sp.adjclose.pct_change()
        df_sp["log_return"] = np.log(1 + df_sp["pct_change"].astype(float))
        df_sp["ticker"] = ticker
        df_sp["mkt_change"] = df_sp["pct_change"]
        df_sp = df_sp.reset_index()
        df_sp["price_date"] = df_sp["index"]
        df_sp = df_sp.drop(columns=["index"])
        df_sp["mktclose"] = df_sp["adjclose"]
        df_sp["mkt_plus_1"] = df_sp["mktclose"].shift(-1)
        df_sp["mkt_plus_5"] = df_sp["mktclose"].shift(-5)
        df_sp["mkt_plus_10"] = df_sp["mktclose"].shift(-10)
        df = pd.merge(
            df,
            df_sp[
                [
                    "ticker",
                    "price_date",
                    "mkt_change",
                    "mktclose",
                    "mkt_plus_1",
                    "mkt_plus_5",
                    "mkt_plus_10",
                ]
            ],
        )

        model = RollingOLS.from_formula(
            "pct_change ~ mkt_change",
            data=df[["pct_change", "mkt_change"]],
            window=rolling_window,
        )
        rres = model.fit()
        df[beta] = rres.params[["mkt_change"]]

        df["plus_1_return"] = df["adjclose_plus_1"] / df["adjclose"] - 1
        df["plus_5_return"] = df["adjclose_plus_5"] / df["adjclose"] - 1
        df["plus_10_return"] = df["adjclose_plus_10"] / df["adjclose"] - 1
        df["mktclose_plus_1_return"] = df["mkt_plus_1"] / df["mktclose"] - 1
        df["mktclose_plus_5_return"] = df["mkt_plus_5"] / df["mktclose"] - 1
        df["mktclose_plus_10_return"] = df["mkt_plus_10"] / df["mktclose"] - 1
        df["expected_1_day_return"] = df[beta] * df["mktclose_plus_1_return"]
        df["expected_5_day_return"] = df[beta] * df["mktclose_plus_5_return"]
        df["expected_10_day_return"] = df[beta] * df["mktclose_plus_10_return"]
        df["alpha_1_day"] = df["plus_1_return"] - df["expected_1_day_return"]
        df["alpha_5_day"] = df["plus_5_return"] - df["expected_5_day_return"]
        df["alpha_10_day"] = df["plus_10_return"] - df["expected_10_day_return"]

    return df


def get_betas(x, y, n=0):
    if n > 0:
        x = x.iloc[
            -n:,
        ]
        y = y.iloc[
            -n:,
        ]
    res = ssm.OLS(y, x).fit()
    beta = res.params[0]
    r2 = res.rsquared
    n = len(x)
    return [beta, r2, n]


def get_fred_data(series_list, config, quarters=True):

    end_date = datetime.today().strftime("%Y-%m-%d")
    FRED = Fred(config.fred_key)  # Fred(config.fred_key)

    df_fred = pd.DataFrame()
    for fred_series in series_list:
        s = pd.DataFrame(
            FRED.get_series(
                fred_series, observation_start="2016-01-01", observation_end=end_date
            )
        ).reset_index()
        s.columns = ["end_date", fred_series]
        s["quarter"] = pd.PeriodIndex(s.end_date, freq="Q")
        # s = calendar_quarter(s, "end_date")
        if quarters == True:
            s = s.groupby("quarter").mean().reset_index()
        if df_fred.shape[0] > 0:
            s = s.drop(columns="end_date")
            df_fred = pd.merge(
                df_fred, s, how="inner", left_on="quarter", right_on="quarter"
            )
        else:
            df_fred = s
    return df_fred


def append_price_data(df):

    df = calendar_quarter(df, "end_date")

    pd.set_option("mode.chained_assignment", None)
    ticker = df.iloc[0]["ticker"]
    stock_ticker = ticker.split(" ")[0]
    df_prices = get_price_data(stock_ticker)
    df_prices = df_prices.reset_index()
    df_prices = df_prices[["pricing_date", "adjclose"]]
    df_prices.columns = ["end_date", "adjclose"]

    df_prices = calendar_quarter(df_prices, "end_date")
    df_prices = df_prices.sort_values("end_date")
    df_prices = df_prices.groupby("end_date_CALENDAR_QUARTER").last().reset_index()

    # fiscal_quarter
    df = df_filter(df, "period_duration_type", ["fiscal_quarter"])  # type: ignore
    dates = list(set(list(df["end_date_CALENDAR_QUARTER"])))
    df_p = df_filter(df_prices, "end_date_CALENDAR_QUARTER", dates)  # type: ignore

    df_p.columns = ["end_date_CALENDAR_QUARTER", "end_date", "value"]
    df_p["ticker"] = ticker
    df_p["period"] = ""
    df_p["period_duration_type"] = "fiscal_quarter"
    df_p["category"] = ""
    df_p["type"] = ""
    df_p["row_header"] = "Stock Price"
    df_p["unit"] = "$"
    df_p = df_p.sort_values("end_date")
    df_1 = df_p[
        [
            "ticker",
            "period",
            "period_duration_type",
            "end_date",
            "category",
            "type",
            "row_header",
            "unit",
            "value",
            "end_date_CALENDAR_QUARTER",
        ]
    ]

    df = pd.concat([df, df_1])

    return df


def calendar_quarter(df, col, datetime=True):
    pd.set_option("mode.chained_assignment", None)
    # translate a date into sort-able and group-able YYYY-mm format.
    df[col] = pd.to_datetime(df[col])

    df[col + "shift"] = df[col] + pd.Timedelta(days=-5)

    df[col + "_CALENDAR_QUARTER"] = df[col + "shift"].dt.to_period("Q")

    df = df.drop(columns=[col + "shift"])
    df[col + "_CALENDAR_QUARTER"] = df[col + "_CALENDAR_QUARTER"].astype(str)

    return df


def get_earnings_dates(ticker):
    df = pd.DataFrame.from_dict(si.get_earnings_history(ticker))
    df["earnings_date"] = df["startdatetime"].astype(str)
    df["earnings_date"] = df["earnings_date"].str.split("T").str[0]
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df = df.dropna()

    return df[
        [
            "ticker",
            "companyshortname",
            "earnings_date",
            "epsestimate",
            "epsactual",
            "epssurprisepct",
        ]
    ]


def regress_dataframe(df, y_col, x_col, y_filter, sort_col="", n=24, z_score=False):
    import datetime as ddt

    if z_score == True:
        df[x_col] = (df[x_col] - df[x_col].mean()) / df[x_col].std(ddof=0)
        df[y_col] = (df[y_col] - df[y_col].mean()) / df[y_col].std(ddof=0)

    if df.dropna().shape[0] == 0:
        print("Not enough datapoints")
        return

    if y_filter != "":
        df = df.loc[df[y_col] == y_filter]

    if sort_col != "":
        df = df.sort_values(sort_col, ascending=False)

    X = df.tail(n)[x_col]
    Y = df.tail(n)[y_col]

    X = ssm.add_constant(X)  # to add constant value in the model
    ols_model = ssm.OLS(
        Y.astype(float), X.astype(float), missing="drop"
    ).fit()  # fitting the model
    predictions = ols_model.summary()  # summary of the model
    return predictions


def get_earnings_and_prices(ticker, index_ticker):
    df_earnings = get_earnings_dates(ticker)
    if df_earnings.shape[0] == 0:
        print("No earnings data for " + ticker)
        return None

    df_prices = get_price_data(
        ticker,
        reset=True,
        rolling_betas=True,
        rolling_window=252,
        index_ticker=index_ticker,
    )
    df = pd.merge(
        df_prices,
        df_earnings,
        how="inner",
        left_on=["ticker", "price_date"],
        right_on=["ticker", "earnings_date"],
    )
    beta = "beta_60"
    if "beta_60" not in df.columns:
        for col in df.columns:
            if "beta" in col:
                beta = col
    df = df[
        [
            "ticker",
            "companyshortname",
            "price_date",
            "earnings_date",
            "epsestimate",
            "epsactual",
            "epssurprisepct",
            "alpha_1_day",
            "alpha_5_day",
            "alpha_10_day",
            beta,
            "plus_1_return",
            "plus_5_return",
            "plus_10_return",
            "mktclose_plus_1_return",
            "mktclose_plus_5_return",
            "mktclose_plus_10_return",
        ]
    ].sort_values("earnings_date", ascending=False)
    df = df.groupby(["ticker", "earnings_date"]).first().reset_index()
    return df


def regress_dataframe_groups(
    df_data, y_name="", return_grouped=True, n_periods=12, regression_warnings=False
):
    regression_warning = {}

    max_time_period = df_data["period_end_date"].max()

    period_names = list(
        df_data.groupby("period_name")
        .first()
        .sort_values("period_end_date")
        .reset_index()
        .tail(n_periods)["period_name"]
    )

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    df_data = df_data.sort_values(["time_series_name", "period_end_date"])
    df_data = df_data[df_data[y_name].notna()]
    df_data = df_data[df_data["value"].notna()]

    def extract_lr(x, y_name):
        test_time_period = x["period_end_date"].max()
        if test_time_period != max_time_period:
            time_series_name = x.iloc[0]["time_series_name"]
            regression_warning[time_series_name] = [
                "Time series ends at " + str(test_time_period)
            ]
            return None

        x = x.loc[x["period_name"].isin(period_names)]
        x = x[x["value"].notna()]
        if x.shape[0] < n_periods and x.shape[0] > 0:
            time_series_name = x.iloc[0]["time_series_name"]
            regression_warning[time_series_name] = [
                "Time series has " + str(x.shape[0]) + " periods"
            ]
        if x.shape[0] == 0:
            return None

        model = smf.ols(formula="value ~ " + y_name, data=x)
        rres = model.fit()

        x["intercept"] = rres.params[0]
        x["n_periods"] = x.shape[0]
        x["rsquared"] = rres.rsquared
        x["slope"] = rres.params[1]

        return pd.DataFrame(x)

    grouped = df_data.groupby("time_series_name")  # group by each time series name
    grouped = grouped.apply(lambda x: extract_lr(x, y_name))

    if regression_warning and regression_warnings:
        print("Regression warning:")
        print(pd.DataFrame.from_dict(regression_warning, orient="index"))

    if return_grouped == True:
        try:
            df_output = (
                grouped.reset_index(drop=True)
                .groupby(
                    [
                        "name_index",
                        "time_series_description",
                        "time_series_name",
                        "category",
                    ]
                )
                .mean()
                .reset_index()
                .sort_values("rsquared", ascending=False)
            )
            # df_output = df_output.loc[df_output["n_periods"] >= n_periods]
            return df_output[
                [
                    "name_index",
                    "time_series_description",
                    "time_series_name",
                    "category",
                    "rsquared",
                    "slope",
                    "intercept",
                    "n_periods",
                ]
            ]
        except:
            print("Regression error")
    else:
        return grouped


def format_borders(plot, colors):
    plot.spines["top"].set_visible(False)
    plot.spines["left"].set_visible(False)
    plot.spines["left"].set_color(colors["grey"])
    plot.spines["bottom"].set_color(colors["grey"])


def plot_prices_against_time_series(
    df_data, df_prices, time_series_name, value_name="value"
):
    ticker = df_data.iloc[0]["ticker"]

    df_filter = df_data.loc[df_data["time_series_name"] == time_series_name]

    time_series_description = df_filter.iloc[0]["time_series_description"]

    df_plot = pd.merge(
        df_prices[["adjclose", "price_date"]],
        df_filter[["time_series_name", "earnings_date", value_name]],
        how="outer",
        left_on="price_date",
        right_on="earnings_date",
    )
    df_plot = df_plot.loc[df_plot["price_date"] > "2020-01-01"]

    df_plot["row_num"] = np.arange(len(df_plot))
    markers_on = df_plot.dropna()

    markers_on = list(markers_on["row_num"])

    colors = {
        "red": "#ff207c",
        "grey": "#42535b",
        "blue": "#00838F",
        "orange": "#ffa320",
        "green": "#00ec8b",
    }
    plt.rc("figure", figsize=(12, 9))

    config_ticks = {"size": 14, "color": colors["grey"], "labelcolor": colors["grey"]}
    config_title = {"size": 18, "color": colors["grey"], "ha": "left", "va": "baseline"}

    fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.tight_layout(pad=3)
    fig.suptitle(str(ticker) + " price vs " + str(time_series_description), fontsize=16)

    date = df_plot["price_date"]
    close = df_plot["adjclose"]
    vol = df_plot[value_name]

    plot_price = axes[0]
    plot_price.plot(
        date,
        close,
        color=colors["blue"],
        linewidth=2,
        label="Price",
        marker="*",
        ms=10,
        markevery=markers_on,
        markerfacecolor="black",
    )

    plot_vol = axes[1]
    plot_vol.bar(date, vol, width=25, color="darkgrey")
    plot_vol.set_ylabel(time_series_name, fontsize=14)
    plot_vol.yaxis.set_label_position("left")
    plot_vol.yaxis.label.set_color(colors["grey"])

    plot_price.yaxis.tick_right()
    plot_price.tick_params(axis="both", **config_ticks)
    plot_price.set_ylabel("Price (in USD)", fontsize=14)
    plot_price.yaxis.set_label_position("right")
    plot_price.yaxis.label.set_color(colors["grey"])
    plot_price.grid(axis="y", color="gainsboro", linestyle="-", linewidth=1)
    plot_price.set_axisbelow(True)

    format_borders(plot_price, colors)
    format_borders(plot_vol, colors)

    return  # plt.show()


def plot_eps_surprise_data(df, alpha_days="alpha_10_days"):

    ticker = df.iloc[0]["ticker"]

    df["z_score_" + alpha_days] = (df[alpha_days] - df[alpha_days].mean()) / df[
        alpha_days
    ].std(ddof=0)

    # df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()

    df["z_score_epssurprisepct"] = (
        df["epssurprisepct"] - df["epssurprisepct"].mean()
    ) / df["epssurprisepct"].std(ddof=0)

    df_plot = df
    i_correlation = round(
        df_plot["z_score_epssurprisepct"].corr(df["z_score_" + alpha_days]), 2
    )

    colors = {
        "red": "#ff207c",
        "grey": "#1d2224",
        "blue": "#00838F",
        "orange": "#ffa320",
        "green": "#00ec8b",
    }
    plt.rc("figure", figsize=(12, 9))

    config_ticks = {"size": 14, "color": colors["grey"], "labelcolor": colors["grey"]}
    config_title = {"size": 18, "color": colors["grey"], "ha": "left", "va": "baseline"}

    # fig, axes = plt#.subplots(, 1, gridspec_kw={"height_ratios": [1, 1]})
    plt.tight_layout(pad=3)

    date = df["price_date"]
    close = df["z_score_" + alpha_days]
    vol = df_plot["z_score_epssurprisepct"]

    plt.scatter(date, close, color=colors["blue"], label="Alpha Z Score")

    plt.scatter(date, vol, color="black", label="Eps Surprise Z Score")

    plt.gca().yaxis.tick_right()
    plt.gca().tick_params(axis="both", **config_ticks)
    plt.gca().set_ylabel("Z Score", fontsize=14)
    plt.gca().yaxis.set_label_position("right")
    plt.grid(axis="y", color="gainsboro", linestyle="-", linewidth=0.5)
    plt.legend(loc="upper left")
    plt.title(
        ticker + " earnings suprise and alpha correlation:" + str(i_correlation),
        fontsize=16,
    )
    format_borders(plt.gca(), colors)

    return


def regress_dataframe_time_series_groups(
    df_data=None,
    y_name="",
    return_grouped=True,
    use_qoq_deltas=False,
    use_yoy_deltas=False,
    category_filter=[],
    category="",
    n_periods=16,
):
    df_data = df_data[
        ~df_data["time_series_name"].str.contains("other", case=False, regex=True)
    ]

    if len(category_filter):
        df_data = df_data.loc[~df_data["category"].isin(category_filter)]

    if type(category) is list:
        df_data = df_data.loc[df_data["category"].isin(category)]
    elif category != "":
        df_data = df_data.loc[df_data["category"] == category]

    if use_qoq_deltas == False:
        df_data = df_data.loc[
            ~df_data["time_series_name"].str.contains("quarter_over_quarter")
        ]

    if use_yoy_deltas == False:
        df_data = df_data.loc[
            ~df_data["time_series_name"].str.contains("year_over_year")
        ]

    ticker = df_data.iloc[0]["ticker"]

    df = regress_dataframe_groups(
        df_data, y_name=y_name, return_grouped=return_grouped, n_periods=n_periods
    )

    if df is None:
        print("Regression error: perhaps not enough datapoints")
        return None

    df["ticker"] = ticker
    df["name_index"] = df["name_index"].astype(int)
    col_name = "ticker"
    first_col = df.pop(col_name)
    df.insert(0, col_name, first_col)

    df = df.loc[df["rsquared"] < 0.98]
    df = df.loc[df["rsquared"] > 0.01]
    df = df.loc[df["n_periods"] >= n_periods]
    return df


def rolling_r_squared(
    df_data, time_series_name, window, y_name="alpha_10_day", plot=False
):
    # df_data = df_data.sort_values('period_end_date',ascending=False)
    category_filter = []
    df = df_data[
        df_data["time_series_name"] == time_series_name
    ]  # [['period_name','value','earnings_date','alpha_10_day']]
    time_series_description = df.iloc[0]["time_series_description"]
    model = RollingOLS.from_formula(
        "value ~ " + str(y_name), data=df[["value", y_name]], window=window
    )
    rres = model.fit()
    df["intercept"] = rres.params[["Intercept"]]
    df["slope"] = rres.params[[y_name]]
    df["rsquared"] = rres.rsquared
    df = df[
        [
            "period_name",
            "earnings_date",
            y_name,
            "value",
            "slope",
            "rsquared",
            "intercept",
        ]
    ].dropna()

    if df["rsquared"].isnull().all():
        print(
            "Regression warning: All rsquareds are null. Time series length is less than the window. Try a shorter window."
        )

    if plot == False:
        return df

    colors = {
        "red": "#ff207c",
        "grey": "#C3C2C3",
        "blue": "#00838F",
        "orange": "#ffa320",
        "green": "#00ec8b",
    }
    plt.rc("figure", figsize=(12, 9))

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
    fig.suptitle(time_series_description, fontsize=16)
    fig.tight_layout(pad=1)

    ax1.plot(df["slope"], label="Regression Slope", color=colors["blue"], linewidth=2)
    ax1.legend(loc="upper left")

    ax2.plot(df["rsquared"], label="R Squared", color=colors["blue"], linewidth=2)
    ax2.legend(loc="upper left")

    ax3.plot(df["value"], label="Time Series Value", color=colors["blue"], linewidth=2)
    ax3.legend(loc="upper left")

    import warnings

    warnings.filterwarnings("ignore")

    ax1.set_xticklabels(df["period_name"])  # annoying matplotlib warning in error

    ax1.get_shared_x_axes().join(ax1, ax2, ax3)

    # ax1.title.set_text(time_series_description,fontsize=14)
    format_borders(ax1, colors)
    format_borders(ax2, colors)
    format_borders(ax3, colors)

    plt.show()


def get_earnings_estimates(ticker):
    return si.get_analysts_info(ticker)["Earnings Estimate"]


def get_revenue_estimates(ticker):
    return si.get_analysts_info(ticker)["Revenue Estimate"]


def get_eps_revisions(ticker):
    return si.get_analysts_info(ticker)["EPS Revisions"]
