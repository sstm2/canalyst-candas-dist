"""
Helper functions for running transformations
"""

import re
import os
import json
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from canalyst_candas.version import __version__ as version
from canalyst_candas import settings


def get_api_headers(canalyst_api_key: Optional[str]) -> Dict[str, str]:
    """
    Return the authorization bearer header to use for API requests and user agent
    """
    return {
        "Authorization": f"Bearer {canalyst_api_key}",
        "User-Agent": f"canalyst-sdk-{version}",
    }


def _get_mds_bulk_data_url(
    bulk_data_key: settings.BulkDataKeys,
    file_type,
    csin,
    model_version,
    mds_host: str = settings.MDS_HOST,
):
    return (
        f"{mds_host}/"
        f"{settings.BULK_DATA_URL.format(csin=csin, version=model_version, bulk_data_key=bulk_data_key.value, file_type=file_type)}"
    )


def _get_mds_time_series_url(
    csin: str, time_series_name: str, mds_host: str = settings.MDS_HOST
):
    return f"{mds_host}/{settings.TIME_SERIES_URL.format(csin=csin, time_series_name=time_series_name)}"


def save_guidance_csv(df, ticker, filename):
    """
    Save guidance CSV
    """
    pathname = filename  # CWD + "\\DATA\\" + filename + "\\"  # .replace(r,'')

    df = df.iloc[3:, 1:]  # remove extraneous columns & rows
    df = df.dropna(axis=1, how="all")  # remove columns that have no data
    df.columns = df.iloc[0]  # fix columns to have proper titles
    df = df.drop(df.index[0])  # drop duplicated row
    df.to_csv(pathname + "/" + ticker + "_guidance.csv")
    print(ticker + " guidance done")
    return df


def filter_dataset(
    df,
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
    n_diff="",
):
    # NOTE: consider using pydantic's validation model instead
    if (
        period_duration_type == ""
        and period_name == ""
        and warning is True
        and mrq is False
    ):
        print(
            "Warning: Returning both FY and Q.  Use period_duration_type = 'fiscal_quarter' or "
            "period_duration_type = 'fiscal_year' to filter."
        )

    if mrq is True:
        mrq = df["MRFQ"].iloc[0]
        df = df.loc[df["period_name"] == df["MRFQ"]]

    df = df.sort_values(["period_end_date", "name_index"])

    if type(category) is list:
        df = df.loc[df["category"].isin(category)]
    elif category != "":
        df = df.loc[df["category"] == category]

    if period_name != "":
        df = df.loc[df["period_name"] == period_name]

    if period_duration_type != "":
        df = df.loc[df["period_duration_type"] == period_duration_type]

    if is_driver != "":
        df = df.loc[df["is_driver"] == is_driver]

    if is_historical != "":
        df = df.loc[df["is_historical"] == is_historical]

    if unit_type != "":
        df = df.loc[df["unit_type"] == unit_type]

    if type(n_periods) is range:  # is this a range
        list_numbers = []
        list_numbers = list(n_periods)
        df = df.groupby(["ticker", "time_series_name"]).take(list_numbers).reset_index()

    elif type(n_periods) is int:
        if n_periods > 0:
            df = (
                df.groupby(["ticker", "time_series_name", "period_duration_type"])
                .tail(n_periods)
                .reset_index()
            )
        else:
            df = (
                df.groupby(["ticker", "time_series_name", "period_duration_type"])
                .head(-1 * n_periods)
                .reset_index()
            )

    if (
        type(time_series_name) is not list and time_series_name != ""
    ):  # this should be replaced with a search function
        df1 = df[
            df["time_series_name"].str.contains(
                time_series_name, flags=re.IGNORECASE, regex=True
            )
        ]

        df2 = df[
            df["time_series_description"].str.contains(
                time_series_name, flags=re.IGNORECASE, regex=True
            )
        ]

        df = pd.concat([df1, df2])

        df = df.sort_values("period_end_date")

    if type(time_series_name) is list:
        list_df = []
        for item in time_series_name:
            df_copy = df.loc[df["time_series_name"] == item]
            df_copy = df_copy.sort_values("period_end_date")
            list_df.append(df_copy)
        df = pd.concat(list_df)

    if (
        n_diff != ""
    ):  # this should be replaced with a diff function in the new ModelFrame() class
        df = df.sort_values(["ticker", "time_series_name", "period_end_date"])
        df["value"] = df["value"].pct_change(periods=int(n_diff))

    if pivot is True:
        df = pivot_df(df, mrq_notation)

    return df


def make_scenario_data(feature_name, feature_period, feature_value, scenario_name):
    """
    Create JSON structured data from scenario information

    TODO: possible unused function: delete if necessary
    """
    data = {
        "changes": [
            {
                "time_series": feature_name,
                "period": feature_period,
                "value_expression": {"type": "literal", "value": feature_value},
            }
        ],
        "name": scenario_name,
    }
    return data


def map_scenario_urls(json):
    scenario_map = {}
    results = json.get("results", {})
    for result in results:
        name = result.get("name")
        url = result.get("self")
        scenario_map[name] = url

    return scenario_map


def print_data_dir(config):
    print(f"The data is downloaded in {config.default_dir}")


def scandir(dir: str, ext: list) -> Tuple[List[str], List[str]]:  # dir: str, ext: list
    subfolders, files = [], []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    # to iterate is human, to recurse is divine
    for dir in list(subfolders):
        sf, f = scandir(dir, ext)  # type: ignore
        subfolders.extend(sf)
        files.extend(f)  # type: ignore
    return subfolders, files


def create_pdf_from_dot(dot_file):
    """
    Create a PDF file from a Dot file

    TODO: possible unused function: delete if necessary
    """
    pdf_file = dot_file.replace(".dot", ".pdf")
    import graphviz  # NOTE: move import to top

    s = graphviz.Source.from_file(dot_file)
    s.render()
    return pdf_file


def get_forecast_url(csin, latest_version, mds_host=settings.MDS_HOST):
    """
    Return a formatted MDS forecast endpoint
    """
    return (
        f"{mds_host}/api/equity-model-series/{csin}/equity-models/"
        f"{latest_version}/forecast-periods/"
    )


def select_distinct(df, col):
    # TODO: possible unused function: delete if necessary
    d = {}
    d[col] = df[col].unique()
    d = pd.DataFrame(d)
    return d


def get_model_url(ticker, wp_host=settings.WP_HOST):
    """
    Return MDS endpoint URL for model

    TODO: possible unused function: delete if necessary
    """
    file_ticker = ticker.split(" ")[0]
    url = f"{wp_host}/files/search?query={file_ticker}"
    return url


def read_json(ticker, default_dir=settings.DEFAULT_DIR):
    # TODO: possible unused function: delete if necessary
    ticker = ticker.split(" ")[0]
    files_path = f"{default_dir}/DATA/{ticker}/"
    arr = os.listdir(files_path)
    dict_js = {}
    for file_name in arr:
        if ".json" in file_name:
            file_name = f"{default_dir}/DATA/{ticker}/{file_name}"
            with open(file_name, "r") as j:
                contents = json.loads(j.read())
            dict_js[file_name] = contents
            os.remove(file_name)
    return dict_js


def write_json(json_data, file_name):
    """
    Write JSON to file
    """
    with open(file_name, "w") as f:
        json.dump(json_data, f)


def json_to_df(dict_json, ticker):
    """
    Creates a dataframe out of JSON data

    TODO: possible unused function: delete if necessary
    """
    json_key_list = sorted(dict_json.keys())
    list_out = []
    for key in json_key_list:
        for i in range(len(dict_json[key])):
            dict_out = {}
            content = dict_json[key][i]
            dict_out["ticker"] = ticker
            dict_out["period_name"] = content["period"]["name"]
            dict_out["period_duration_type"] = content["period"]["period_duration_type"]
            dict_out["period_start_date"] = content["period"]["start_date"]
            dict_out["period_end_date"] = content["period"]["end_date"]
            dict_out["category"] = content["time_series"]["category"]["description"]
            dict_out["category_type_slug"] = content["time_series"]["category"]["type"][
                "slug"
            ]  # ?
            dict_out["time_series_slug"] = content["time_series"][
                "slug"
            ]  # api to use slugs vs names? ... can map two or more slugs to the same name ...
            dict_out["time_series_name"] = content["time_series"]["names"][
                0
            ]  # can't apply the same name to more than one time series model - an excel thing
            dict_out["category_type_slug"] = content["time_series"]["category"]["type"][
                "slug"
            ]  # financial or operating stats or other
            dict_out["category_type_name"] = content["time_series"]["category"]["type"][
                "name"
            ]  # financial or operating stats or other
            dict_out["time_series_description"] = content["time_series"][
                "description"
            ]  # for use when we have not applied MO names
            dict_out["unit_description"] = content["time_series"]["unit"]["description"]
            dict_out["unit_symbol"] = content["time_series"]["unit"]["symbol"]
            dict_out["unit_type"] = content["time_series"]["unit"]["unit_type"]
            dict_out["value"] = content["value"]
            df = pd.DataFrame.from_dict(dict_out, orient="index").T
            list_out.append(df)

    df_data = pd.concat(list_out)
    return df_data


def mrq_df(df):
    # need to loop this by ticker
    ticker_list = list(set(list(df["ticker"])))
    df_out = []
    df2 = df
    name_cols = [
        "name_index",
        "ticker",
        "category",
        "time_series_name",
        "time_series_description",
        "is_driver",
        "MRFQ",
    ]

    for ticker in ticker_list:
        df = df2.loc[df2["ticker"] == ticker]
        mrq = df.iloc[0]["MRFQ"]
        mry = mrq[-4:]
        if int(mrq[-6]) != 4:
            mry = str(int(mry) - 1)

        df_copy = df.copy()
        df = df.loc[df["period_duration_type"] == "fiscal_quarter"]

        i_switch = 0  # handle that when we get here, we may only have fiscal years or only fiscal quarters or both

        if df.shape[0] != 0:
            i_switch = 1  # we have fiscal quarters

            df = df.sort_values(["ticker", "time_series_name", "period_end_date"])
            df["temp_n"] = np.where(df["period_name"] == mrq, 0, 1)
            df["temp_c"] = df.groupby(["ticker", "time_series_name"]).cumcount() + 1

            df = df.sort_values("temp_n")
            if df[df["period_name"].str.contains(mrq)].shape[0] == 0:
                i_value = 0
            else:
                i_value = df.iloc[0]["temp_c"]

            df["n_index"] = df["temp_c"] - i_value
            df["FYFQ"] = df["n_index"]
            df = df.sort_values("n_index")
            df = df.drop(columns=["temp_c", "n_index", "temp_n"])
            df = pd.pivot_table(
                df,
                values="value",
                index=[
                    "name_index",
                    "ticker",
                    "category",
                    "time_series_name",
                    "time_series_description",
                    "is_driver",
                    "MRFQ",
                ],
                columns=["FYFQ"],
                aggfunc=np.sum,
            ).reset_index()

            rename_cols = []
            for col in df.columns:
                if col not in name_cols:
                    col = "FQ" + str(col)
                    rename_cols.append(col)
            df.columns = name_cols + rename_cols
            df_q = df

        df = df2.loc[df2["ticker"] == ticker]
        df = df.loc[df["period_duration_type"] == "fiscal_year"]

        if df.shape[0] != 0:
            if i_switch:
                i_switch = (
                    2  # we have both fiscal quarters and years otherwise i switch is 0
                )

            df = df.sort_values(["ticker", "time_series_name", "period_end_date"])
            df["temp_n"] = np.where(df["period_name"].str.contains(mry), 0, 1)
            df["temp_c"] = df.groupby(["ticker", "time_series_name"]).cumcount() + 1
            df = df.sort_values("temp_n")

            if df[df["period_name"].str.contains(mry)].shape[0] == 0:
                i_value = 0
            else:
                i_value = df.iloc[0]["temp_c"]

            df["n_index"] = df["temp_c"] - i_value
            df["FYFQ"] = df["n_index"]
            df = df.sort_values("n_index")
            df = df.drop(columns=["temp_c", "n_index", "temp_n"])
            df = pd.pivot_table(
                df,
                values="value",
                index=[
                    "name_index",
                    "ticker",
                    "category",
                    "time_series_name",
                    "time_series_description",
                    "is_driver",
                    "MRFQ",
                ],
                columns=["FYFQ"],
                aggfunc=np.sum,
            ).reset_index()

            rename_cols = []
            for col in df.columns:
                if col not in name_cols:
                    col = "FY" + str(col)
                    rename_cols.append(col)
            df.columns = name_cols + rename_cols
            df_y = df

        if i_switch == 2:  # both
            df = pd.merge(
                df_q, df_y, how="inner", left_on=name_cols, right_on=name_cols
            )
        if i_switch == 1:  # quarters only
            df = df_q
        if i_switch == 0:  # years only
            df = df_y
        df_out.append(df)

    df = pd.concat(df_out, join="inner", ignore_index=True)
    return df.sort_values(["ticker", "time_series_name"])


def pivot_df(df, mrq_notation):

    df["period_name_sorted"] = np.where(
        df["period_name_sorted"].isna(), df["period_name"], df["period_name_sorted"]
    )

    if mrq_notation == True:
        df = mrq_df(df)
        return df

    df = pd.pivot_table(
        df,
        values="value",
        index=[
            "ticker",
            "name_index",
            "category",
            "time_series_name",
            "time_series_description",
            "is_driver",
            "MRFQ",
        ],
        columns=["period_name_sorted"],
        aggfunc=np.sum,
    ).reset_index()

    df = df.sort_values("name_index")
    return df


def get_cache_tickers(default_dir=settings.DEFAULT_DIR):
    # TODO: possible unused function: delete if necessary
    folder = "DATA"
    folder = f"{default_dir}/{folder}"
    subfolders, files = scandir(folder, [".xlsx"])
    ticker_list = []
    for filename in files:
        if "data" not in filename:
            ticker_list.append(filename)
    return ticker_list


def find_in_list(feature_list, search_term):
    """
    Finds search term in items from passed in list

    TODO: possible unused function: delete if necessary
    """
    return_list = []
    for item in feature_list:
        if search_term.lower() in item.lower():
            return_list.append(item)
    return return_list


# helper function ... simply df loc
def df_filter(df, col, feature_list):
    if col == "":
        col = "time_series_name"
    if type(feature_list) == str:
        df = df.loc[df[col].str.contains(feature_list)]
        return df
    df = df.loc[df[col].isin(feature_list)]
    return df


def plot_all(
    df,
    index_col,
    group_col,
    value_col,
    title_text,
    plot_kind="line",
    allow_na=False,
    n="",
):
    # TODO: possible unused function: delete if necessary
    df[value_col] = df[value_col].astype(float)
    df = df.pivot(index=index_col, columns=group_col, values=value_col)
    if allow_na == False:
        df = df.dropna()
    if n != "":
        df.head(n)
    plt = df.plot(title=title_text, kind=plot_kind)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.plot()


def plot_all_labels(
    df, index_col, group_col, value_col, title_text, plot_kind="line", labels=None
):
    # TODO: possible unused function: delete if necessary
    df[value_col] = df[value_col].astype(float)
    df = df.pivot(index=index_col, columns=group_col, values=value_col)
    # df = df.dropna()
    return df.plot(title=title_text, kind=plot_kind, labels=labels)


def calendar_quarter(df, col, datetime=True):
    pd.set_option("mode.chained_assignment", None)
    # translate a date into sort-able and group-able YYYY-mm format.
    df[col] = pd.to_datetime(df[col])

    df[col + "shift"] = df[col] + pd.Timedelta(days=-12)

    df[col + "_CALENDAR_QUARTER"] = df[col + "shift"].dt.to_period("Q")

    df = df.drop(columns=[col + "shift"])
    df[col + "_CALENDAR_QUARTER"] = df[col + "_CALENDAR_QUARTER"].astype(str)

    return df


def _validate_colors(
    colors,
    figure_bg_color,
    vertical_line_color,
    ax_spine_color,
    chart_bg_color,
):
    """
    Validate color arguments

    NOTE: consider using pydantic's validation model instead
    """

    # Validate arguments
    check_for_hex = "^#([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$"

    if not isinstance(colors, List):
        return False
    else:
        for color in colors:
            match = re.search(check_for_hex, color)
            if not match:
                return False

    match_figure_bg_color = re.search(check_for_hex, figure_bg_color)
    match_vertical_line_color = re.search(check_for_hex, vertical_line_color)
    match_ax_spine_color = re.search(check_for_hex, ax_spine_color)
    match_chart_bg_color = re.search(check_for_hex, chart_bg_color)

    return (
        match_figure_bg_color
        and match_vertical_line_color
        and match_ax_spine_color
        and match_chart_bg_color
    )


def get_chart_brand_config(
    chart_colors: List[str] = settings.BRAND_CONFIG_DEFAULTS["chart_plot_colors"],
    figure_bg_color: str = settings.BRAND_CONFIG_DEFAULTS["figure_bg_color"],
    vertical_line_color: str = settings.BRAND_CONFIG_DEFAULTS["vertical_line_color"],
    ax_spine_color: str = settings.BRAND_CONFIG_DEFAULTS["ax_spine_color"],
    title_font_path: str = settings.BRAND_CONFIG_DEFAULTS["title_font_path"],
    body_font_path: str = settings.BRAND_CONFIG_DEFAULTS["body_font_path"],
    chart_bg_color: str = settings.BRAND_CONFIG_DEFAULTS["chart_bg_color"],
    font_color: str = settings.BRAND_CONFIG_DEFAULTS["font_color"],
    logo_path: str = settings.BRAND_CONFIG_DEFAULTS["logo_path"],
):
    """
    Get a brand config for custom chart branding

    Returns a dict of the complete set of config options.
    """
    is_validated = _validate_colors(
        chart_colors,
        figure_bg_color,
        vertical_line_color,
        ax_spine_color,
        chart_bg_color,
    )

    if not is_validated:
        raise ValueError(
            "All color values must be in Hex format (e.g #FFF or #FFFFFF)."
        )

    chart_brand_config = {
        "chart_plot_colors": chart_colors,
        "figure_bg_color": figure_bg_color,
        "chart_bg_color": chart_bg_color,
        "vertical_line_color": vertical_line_color,
        "ax_spine_color": ax_spine_color,
        "title_font_path": title_font_path,
        "body_font_path": body_font_path,
        "font_color": font_color,
        "logo_path": logo_path,
    }

    return chart_brand_config
