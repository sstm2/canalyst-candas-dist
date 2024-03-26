"""
Utilities module
"""
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple
import pandas as pd
import canalyst_candas
from canalyst_candas.configuration.exceptions import ConfigException
from python_graphql_client import GraphqlClient
import numpy as np
from urllib.parse import quote_plus
from io import BytesIO, StringIO
from canalyst_candas import settings
import re
import boto3
from boto3.session import Session
import os
import string
from os import DirEntry, path
import requests
import time
import json
from openpyxl import load_workbook, styles
import datetime
import csv
from canalyst_candas.version import __version__ as version
import pickle
import pkg_resources
from canalyst_candas.exceptions import BaseCandasException

from canalyst_candas.configuration.config import Config, resolve_config
import urllib3
from pathlib import Path

urllib3.disable_warnings()


EQUITY_MODEL_URL = "api/equity-model-series/{csin}/equity-models/{version}/"
CSIN_URL = "api/equity-model-series/?company_ticker_bloomberg={ticker}"
PERIODS_URL = (
    "api/equity-model-series/{csin}/equity-models/{version}/historical-periods/"
)
SCENARIO_URL = "api/equity-model-series/{csin}/equity-models/{version}/scenarios/"
SCENARIO_URL_FORECAST = (
    "api/equity-model-series/{csin}/equity-models/"
    "{version}/scenarios/{scenario_id}/forecast-periods/"
)
BULK_DATA_URL = "api/equity-model-series/{csin}/equity-models/{version}/bulk-data/{bulk_data_key}.{file_type}"

HISTORICAL_PERIODS_TO_FETCH = 36

BRAND_CONFIG_DEFAULTS = {
    "chart_plot_colors": [
        "#E2E3E3",
        "#FFD200",
        "#367ADD",
        "#8CE0FD",
        "#E52B26",
        "#F8623F",
        "#00838F",
        "#30C08F",
        "#8EA1FF",
    ],
    "figure_bg_color": "#FFFFFF",
    "vertical_line_color": "#E2E3E3",
    "ax_spine_color": "#C3C2C3",
    "title_font_path": pkg_resources.resource_filename(
        __name__, "fonts/Barlow-SemiBold.ttf"
    ),
    "body_font_path": pkg_resources.resource_filename(
        __name__, "fonts/Roboto-Regular.ttf"
    ),
    "chart_bg_color": "#161B21",
    "font_color": "#000",
    "logo_path": pkg_resources.resource_filename(__name__, "images/logo.png"),
}

SAMPLE_MODEL_PATH = {
    "drivers_path": pkg_resources.resource_filename(__name__, "sample_data/drivers"),
    "models_path": pkg_resources.resource_filename(__name__, "sample_data/models"),
}


class BulkDataKeys(Enum):
    FORECAST_DATA = "forecast-data"
    HISTORICAL_DATA = "historical-data"
    NAME_INDEX = "name-index"
    MODEL_INFO = "model-info"


class BulkDataFileType(Enum):
    CSV = "csv"
    PARQUET = "parquet"


# Setting it up as a local dict instead of based on the /bulk-data endpoint results
# to allow for faster generation (less queries to the  MDS),
# at the cost of duplication across the MDS & Candas.
BULK_DATA_KEY_FILE_TYPES = {
    BulkDataKeys.FORECAST_DATA: [
        BulkDataFileType.CSV.value,
        BulkDataFileType.PARQUET.value,
    ],
    BulkDataKeys.HISTORICAL_DATA: [
        BulkDataFileType.CSV.value,
        BulkDataFileType.PARQUET.value,
    ],
    BulkDataKeys.NAME_INDEX: [
        BulkDataFileType.CSV.value,
        BulkDataFileType.PARQUET.value,
    ],
    BulkDataKeys.MODEL_INFO: [
        # model info _only_ has CSV due to incompatibility with parquet
        BulkDataFileType.CSV.value
    ],
}


def set_api_key(key, key_file_path):
    """
    Loads API key from path
    """
    with open(key_file_path) as f:
        keys_json = json.load(f)

    keys_json["canalyst_api_key"] = key

    with open(key_file_path, "w") as f:
        json.dump(keys_json, f)


def get_api_headers(canalyst_api_key: Optional[str]) -> Dict[str, str]:
    """
    Return the authorization bearer header to use for API requests and user agent
    """
    return {
        "Authorization": f"Bearer {canalyst_api_key}",
        "User-Agent": f"canalyst-sdk-{version}",
    }


class LogFile:
    """
    Logging helper class
    """

    # to be refactored to log try: except: errors

    # the idea of this class is to help with user debug ...
    def __init__(self, default_dir: str = settings.DEFAULT_DIR, verbose: bool = False):
        self.default_dir = default_dir
        self.verbose = verbose
        tm = datetime.datetime.now()
        self.log_file_name = f"{default_dir}/candas_logfile.csv"

        if not os.path.isfile(self.log_file_name):
            rows: Iterable[Iterable[Any]] = [
                ["timestamp", "action"],
                [tm, "initiate logfile"],
            ]
            with open(self.log_file_name, "w", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(rows)

    def write(self, text):
        if self.verbose is True:
            print(text)
        tm = datetime.datetime.now()
        rows = [tm, text]
        with open(self.log_file_name, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(rows)

    def read(self):
        df = pd.read_csv(self.log_file_name)
        return df


# helper class to get data from S3
class Getter:
    """
    S3 Client to retreive data from S3
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.log = LogFile(default_dir=self.config.default_dir)
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.config.s3_access_key_id,
            aws_secret_access_key=self.config.s3_secret_key,
            verify=config.verify_ssl,
        )

    def get_s3_file(self, to_filename, from_filename):
        """
        Download file from S3
        """
        try:
            with open(to_filename, "wb+") as f:
                self.s3_client.download_fileobj("candas", from_filename, f)
            print(f"Downloaded to {to_filename}. ")
        except:
            self.log.write("Get S3 file failed")
        return

    def get_file_from_s3(self, file_name):
        """
        Get file from S3
        """
        try:
            csv_obj = self.s3_client.get_object(Bucket="candas", Key=file_name)
            body = csv_obj["Body"].read()
        except:
            self.log.write("Missing file from s3." + file_name)
            return None
        return body

    def get_zip_csv_from_s3(self, file_name):
        """
        Get zip from S3
        """
        d = self.s3_client.get_object(Bucket="candas", Key=file_name)
        import io

        buffer = io.BytesIO(d["Body"].read())
        import zipfile

        z = zipfile.ZipFile(buffer)
        with z.open(z.namelist()[0]) as f:
            body = f.read().decode("utf-8")
        return body

    def get_csv_from_s3(self, file_name):
        """
        Get CSV from S3
        """
        try:
            csv_obj = self.s3_client.get_object(Bucket="candas", Key=file_name)
            body = csv_obj["Body"]
            csv_string = body.read().decode("utf-8")
            df = pd.read_csv(StringIO(csv_string))
        except:
            self.log.write("Missing csv from s3." + file_name)
            return None
        return df

    def get_parquet_from_s3(self, file_name):

        """
        Get parquet from S3
        """
        try:
            import io

            csv_obj = self.s3_client.get_object(Bucket="candas", Key=file_name)
            df = pd.read_parquet(io.BytesIO(csv_obj["Body"].read()))
        except:
            return None
        return df

    def get_pickle_from_s3(self, file_name):

        """
        Get pickle from S3
        """
        try:
            csv_obj = self.s3_client.get_object(Bucket="candas", Key=file_name)
            body = csv_obj["Body"].read()
            df = pickle.loads(body)
        except:
            return None
        return df

    def get_file(self, ticker, type, file_type):  # returns a dataframe
        df = None
        """
        Gets ticker file from S3

        Uses local filesystem if no file found on S3
        """

        file_ticker = ticker.split(" ")[0]

        dict_file = {
            "parquet": "get_parquet_from_s3",
            "pickle": "get_pickle_from_s3",
            "csv": "get_csv_from_s3",
        }

        path_name = f"DATA/{file_ticker}/{ticker}_{type}.{file_type}"
        df = getattr(self, dict_file[file_type])(path_name)

        if df is None:
            if path.exists(f"{path_name}"):
                df = pd.read_csv(f"{path_name}")

        if df is None:
            return None

        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        return df


def get_test_data_set(csv_type: BulkDataFileType, ticker: str):
    """
    Fetches data from the test data set in the models folder
    """
    csv_type = csv_type.value.replace("-", "_")
    csv_types = ["historical_data", "forecast_data", "name_index"]
    if csv_type not in csv_types:
        raise BaseCandasException("CSV type is not one of the available options")
    model_path = SAMPLE_MODEL_PATH["models_path"]
    file_name = f"{model_path}/{ticker}_{csv_type}.csv"
    df = pd.read_csv(file_name)
    return df


def get_data_set_from_mds(
    bulk_data_key: BulkDataKeys,
    file_type: str,
    csin: str,
    model_version: str,
    auth_headers: Dict[str, str],
    log: LogFile,
    mds_host: str = settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
):
    """
    Pulls a bulk data file from the MDS at the equity-model endpoint and returns it in a dataframe.

    Different types are available, listed in the BulkDataKeys enum. Different formats are also
    available, listed in the BULK_DATA_KEY_FILE_TYPES based on the type.
    """

    available_file_types = BULK_DATA_KEY_FILE_TYPES[bulk_data_key]
    if file_type not in available_file_types:
        print(
            f"{bulk_data_key.value} is not available in the requested file type: {file_type}.",
            f"It is available in the following type(s): {available_file_types}.",
        )
        return

    url = _get_mds_bulk_data_url(
        bulk_data_key,
        file_type,
        csin,
        model_version,
        mds_host,
    )
    response = get_request(url, auth_headers, log, verify_ssl)
    # parquet not available: fallback to csv
    if response is None:
        url = _get_mds_bulk_data_url(
            bulk_data_key,
            "csv",  # file_type set to csv
            csin,
            model_version,
            mds_host,
        )
        response = get_request(url, auth_headers, log, verify_ssl)
        file_type = "csv"

    if file_type == BulkDataFileType.CSV.value:
        csv_string = response.content.decode("utf-8")
        df = pd.read_csv(StringIO(csv_string))
        return df
    elif file_type == BulkDataFileType.PARQUET.value:
        df = pd.read_parquet(BytesIO(response.content))

        if bulk_data_key in [BulkDataKeys.FORECAST_DATA, BulkDataKeys.HISTORICAL_DATA]:
            # parquet values are set as strings by default, manually set to floats/ints
            df["value"] = pd.to_numeric(df["value"])
            # parquet dates are set as datetime.date objects by default, set to strings
            df["period_end_date"] = df["period_end_date"].astype(str)
            df["period_start_date"] = df["period_start_date"].astype(str)

        return df
    else:
        print(
            f"Unable to retrieve {bulk_data_key} with file type {file_type} from"
            f"the following URL: {url}."
        )
        return


def _get_mds_bulk_data_url(
    bulk_data_key: BulkDataKeys,
    file_type,
    csin,
    model_version,
    mds_host: str = settings.MDS_HOST,
):
    return (
        f"{mds_host}/"
        f"{BULK_DATA_URL.format(csin=csin, version=model_version, bulk_data_key=bulk_data_key.value, file_type=file_type)}"
    )


def save_guidance_csv(df, ticker, filename):
    """
    Save guidance CSV
    """
    pathname = filename  # CWD + "\\DATA\\" + filename + "\\"  # .replace(r,'')

    for i, row in df.iterrows():
        for col in row.index:
            val = row[col]
            val = str(val)

            df = df.iloc[3:, 1:]
            df = df.dropna(axis=1, how="all")

            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
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


def send_scenario(
    ticker,
    data,
    auth_headers,
    logger: LogFile,
    csin="",
    latest_version="",
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
):
    """
    Makes a POST request for scenario to the MDS
    """
    if csin == "" or latest_version == "":
        csin, latest_version = get_company_info_from_ticker(
            ticker, auth_headers, logger, mds_host, verify_ssl
        )
    scenario_url = (
        f"{mds_host}/{SCENARIO_URL.format(csin=csin, version=latest_version)}"
    )
    # print("scenario_url: " + scenario_url)
    try:
        scenario_response = post_request(
            url=scenario_url, headers=auth_headers, json=data, verify_ssl=verify_ssl
        )
    except:
        return None
    # print(scenario_response.text)

    return scenario_response


def make_scenario_data(feature_name, feature_period, feature_value, scenario_name):
    """
    Create JSON structured data from scenario information
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


def get_name_index_from_csv(ticker, config):
    default_dir = config.default_dir

    # check if the model exists if so use it.
    file_ticker = ticker.split(" ")[0]

    get_excel_model(config, ticker, default_dir)

    path_name = f"{config.default_dir}/DATA/{file_ticker}/"

    cache_files = []
    _, files = scandir(path_name, [".xlsx"])
    for filename in files:
        if "data" not in filename:
            cache_files.append(filename)
    for f_path in cache_files:
        if "~" in f_path or ".csv" in f_path:
            continue
        workbook = load_workbook(f_path, data_only=True)
        model_sheet = workbook["Model"]
        list_names = [
            defined_name for defined_name in workbook.defined_names.definedName
        ]

        dict_names = {}
        for named_range in list_names:
            if "Model!" in named_range.attr_text and ":" in named_range.attr_text:
                x = str(named_range.attr_text).replace("Model!", "")
                x = x.split(":")[0]
                x = x.replace("$", "")
                x = x.replace("A", "")
                x = x.replace("C", "")
                x = x.replace("B", "")

                if x != "":
                    dict_names[named_range.name] = x

        df_index = pd.DataFrame(dict_names, index=[0]).T
        df_index = df_index.reset_index()
        df_index.columns = ["time_series_name", "index"]
        df_index["index"] = df_index["index"].astype(int)
        df_index = df_index.sort_values("index")

        workbook.close()
    return df_index


def get_scenarios(
    ticker,
    auth_headers,
    logger: LogFile,
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
):
    if ticker == "":
        ticker = ticker
    url = f"{mds_host}/api/scenarios/"
    csin, latest_version = get_company_info_from_ticker(
        ticker, auth_headers, logger, mds_host, verify_ssl
    )
    response_content = get_request_json_content(url, auth_headers, logger, verify_ssl)
    res = response_content.get("results")
    list_out = []
    for i in res:
        dict_out = {}
        dict_out["model_name"] = i["name"]
        dict_out["model_csin"] = i["equity_model"]["equity_model_series"]["csin"]
        df = pd.DataFrame.from_dict(dict_out, orient="index").T
        list_out.append(df)
    df_data = pd.concat(list_out)
    df_data = df_data.loc[df_data["model_csin"] == csin]
    return df_data


def post_request(url, headers, json, verify_ssl):
    """
    POST request helper function

    Uses timer to account for throttle limits.
    """
    response = requests.post(url=url, headers=headers, json=json, verify=verify_ssl)
    if check_throttle_limit(response):
        response = requests.post(url=url, headers=headers, json=json, verify=verify_ssl)

    return response


def get_company_info_from_ticker(
    ticker,
    auth_headers,
    logger: LogFile,
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
):
    """
    Get CSIN and latest version for a ticker
    """
    company_url = f"{mds_host}/{CSIN_URL.format(ticker=quote_plus(ticker))}"
    response_content = get_request_json_content(
        company_url, auth_headers, logger, verify_ssl
    )

    json = response_content.get("results")[0]
    csin: str = json.get("csin")
    latest_version: str = (
        json.get("latest_equity_model", {}).get("model_version", {}).get("name", {})
    )

    return (csin, latest_version)


def get_csin_from_ticker(
    ticker,
    auth_headers,
    logger: LogFile,
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
):
    """
    Get CSIN for ticker
    """
    company_url = f"{mds_host}/{CSIN_URL.format(ticker=quote_plus(ticker))}"
    response_content = get_request_json_content(
        company_url, auth_headers, logger, verify_ssl
    )
    json = response_content.get("results")[0]
    csin = json.get("csin")
    return csin


def get_excel_model(config, logger, ticker="", download_dir=""):
    """
    Retrieve Excel model from Web Portal
    """
    auth_headers = get_api_headers(config.canalyst_api_key)
    mds_host = config.mds_host
    wp_host = config.wp_host
    if download_dir == "":
        default_dir = config.default_dir
    else:
        default_dir = download_dir

    csin = get_csin_from_ticker(
        ticker, auth_headers, logger, mds_host, config.verify_ssl
    )

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
        query=query, variables=variables, headers=auth_headers, verify=config.verify_ssl
    )
    url = data["data"]["modelSeries"]["latestModel"]["variantsByDimensions"][0][
        "downloadUrl"
    ]
    file_ticker = ticker.split(" ")[0]
    file_name = data["data"]["modelSeries"]["latestModel"]["name"]
    backup_name = file_name.replace(".xlsx", "")

    file_name = f"{default_dir}/DATA/{file_ticker}/{file_name}.xlsx"

    ext = str(datetime.datetime.now())
    ext = ext.replace(" ", "_")
    ext = ext.replace(":", "_")
    ext = ext.replace("-", "_")

    backup_name = f"{default_dir}/DATA/{file_ticker}/{backup_name}.{ext}"

    os.makedirs(f"{default_dir}/DATA/{file_ticker}/", exist_ok=True)

    if path.exists(file_name):
        os.rename(file_name, backup_name)
        print("Backup to: " + backup_name)

    r = requests.get(url, headers=auth_headers, verify=config.verify_ssl)

    with open(file_name, "wb") as f:
        f.write(r.content)
    print("Saved to: " + file_name)

    return


def create_pdf_from_dot(dot_file):
    """
    Create a PDF file from a Dot file
    """
    pdf_file = dot_file.replace(".dot", ".pdf")
    import graphviz

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


def get_forecast_url_data(
    res_dict, ticker, auth_headers, log, verify_ssl=settings.VERIFY_SSL
):
    list_out = []
    url = res_dict["self"]

    res_loop = get_request(url, auth_headers, log, verify_ssl)
    url = res_loop.json()["data_points"]

    res_loop = get_request(url, auth_headers, log, verify_ssl)
    # try:
    next_url = res_loop.json()["next"]
    # except:
    #    next_url = None

    while url is not None:

        res_data = res_loop.json()["results"]
        dict_out = {}

        for res_data_dict in res_data:
            dict_out["time_series_slug"] = res_data_dict["time_series"]["slug"]
            dict_out["time_series_name"] = res_data_dict["time_series"]["names"][0]
            dict_out["time_series_description"] = res_data_dict["time_series"][
                "description"
            ]
            dict_out["category_slug"] = res_data_dict["time_series"]["category"]["slug"]
            dict_out["category"] = res_data_dict["time_series"]["category"][
                "description"
            ]  # ?
            dict_out["category_type_slug"] = res_data_dict["time_series"]["category"][
                "type"
            ]["slug"]
            dict_out["category_type_name"] = res_data_dict["time_series"]["category"][
                "type"
            ]["name"]
            dict_out["unit_description"] = res_data_dict["time_series"]["unit"][
                "description"
            ]
            dict_out["unit_type"] = res_data_dict["time_series"]["unit"][
                "unit_type"
            ]  # ?
            dict_out["unit_symbol"] = res_data_dict["time_series"]["unit"]["symbol"]
            dict_out["period_name"] = res_data_dict["period"]["name"]
            dict_out["period_duration_type"] = res_data_dict["period"][
                "period_duration_type"
            ]
            dict_out["period_start_date"] = res_data_dict["period"]["start_date"]
            dict_out["period_end_date"] = res_data_dict["period"]["end_date"]
            dict_out["value"] = res_data_dict["value"]
            dict_out["ticker"] = ticker
            df = pd.DataFrame.from_dict(dict_out, orient="index").T
            list_out.append(df)
        url = next_url
        try:
            res_loop = get_request(url, auth_headers, log, verify_ssl)
            next_url = res_loop.json()["next"]
        except:
            url = None
    return pd.concat(list_out)


def get_scenario_url_data(
    res_dict,
    ticker,
    auth_headers,
    default_dir=settings.DEFAULT_DIR,
    verify_ssl=settings.VERIFY_SSL,
):
    log = LogFile(default_dir=default_dir)
    url = res_dict["self"]

    res_loop = get_request(url, auth_headers, log, verify_ssl)
    url = res_loop.json()["data_points"]

    res_loop = get_request(url, auth_headers, log, verify_ssl)

    try:
        res_data = res_loop.json()["results"]
    except:
        log.write("Scenario timeout: " + url)
        return

    url = res_loop.json()["next"]
    list_out = []
    while url is not None:
        dict_out = {}
        for res_data_dict in res_data:
            dict_out["time_series_slug"] = res_data_dict["time_series"]["slug"]
            dict_out["time_series_name"] = res_data_dict["time_series"]["names"][0]
            dict_out["time_series_description"] = res_data_dict["time_series"][
                "description"
            ]
            dict_out["category_slug"] = res_data_dict["time_series"]["category"]["slug"]
            dict_out["category_type_slug"] = res_data_dict["time_series"]["category"][
                "type"
            ]["slug"]
            dict_out["category_type_name"] = res_data_dict["time_series"]["category"][
                "type"
            ]["name"]
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
            dict_out["ticker"] = ticker
            df = pd.DataFrame.from_dict(dict_out, orient="index").T
            list_out.append(df)
        try:
            res_loop = get_request(url, auth_headers, log, verify_ssl)
            res_data = res_loop.json()["results"]
            url = res_loop.json()["next"]
        except:
            url = None
    return pd.concat(list_out)


def get_request(url, headers, logger, verify_ssl=settings.VERIFY_SSL):
    """
    Wrapper for a GET request

    Handles throttle limits by using a timer
    """
    response = requests.get(url=url, headers=headers, verify=verify_ssl)
    if check_throttle_limit(response, logger):
        response = requests.get(url=url, headers=headers, verify=verify_ssl)

    try:
        response.raise_for_status()
    except requests.RequestException as ex:
        logger.write(
            f"Candas: Error when making a request to '{url}'. " f"Exception: {ex}."
        )
        return

    return response


def check_throttle_limit(response, logger):
    """
    Checks to see if a throttle limit has been hit, waits until throttle is lifted.
    """
    if response.status_code == 429:
        timer = int(response.headers.get("Retry-After"))
        err_message = (
            f"Request limit hit. Please wait...Retrying request in {timer} seconds."
        )
        print(err_message)
        logger.write(err_message)
        time.sleep(timer)
        return True
    return False


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


def select_distinct(df, col):
    d = {}
    d[col] = df[col].unique()
    d = pd.DataFrame(d)
    return d


def create_drivers_dot(
    time_series, auth_headers, ticker, logger: LogFile, config=None, s3_client=None
):
    if time_series == "":
        time_series = "net-revenue"

    file_ticker = ticker.split(" ")[0]

    path_name = f"DATA/{file_ticker}"

    if not config:
        config = settings.CONFIG or resolve_config()
    if not s3_client:
        s3_client = Getter(config)

    if time_series == "net-revenue":

        dot_file = s3_client.get_file_from_s3(f"DATA/{file_ticker}/drivers.dot")
        if dot_file is not None:
            if not os.path.exists(f"{config.default_dir}/{path_name}"):
                os.makedirs(f"{config.default_dir}/{path_name}")
            file_name = f"{config.default_dir}/{path_name}/drivers.dot"
            with open(file_name, "wb") as f:
                f.write(dot_file)
            return file_name

    if not os.path.exists(f"{config.default_dir}/{path_name}"):
        os.makedirs(f"{config.default_dir}/{path_name}")

    path_name = f"{config.default_dir}/{path_name}/drivers.dot"

    if os.path.exists(path_name) and time_series == "net-revenue":
        return path_name
    else:
        csin, version = get_company_info_from_ticker(
            ticker, auth_headers, logger, config.mds_host, config.verify_ssl
        )
        url = (
            f"{config.mds_host}/api/equity-model-series/{csin}/equity-models/"
            f"{version}/time-series/{time_series}/forecast-data-points/"
        )

        r = requests.get(url, headers=auth_headers, verify=config.verify_ssl)
        json = r.json().get("results")[0]
        name = json["period"]["name"]
        url = (
            f"{config.mds_host}/api/equity-model-series/{csin}/equity-models/"
            f"{version}/time-series/{time_series}/forecast-data-points/{name}/"
            f"drivers/?format=dot"
        )

        print(url)
        r = requests.get(url, headers=auth_headers, verify=config.verify_ssl)
        with open(path_name, "wb") as f:
            f.write(r.content)
        return path_name


def get_model_url(ticker, wp_host=settings.WP_HOST):
    """
    Return MDS endpoint URL for model
    """
    file_ticker = ticker.split(" ")[0]
    url = f"{wp_host}/files/search?query={file_ticker}"
    return url


# retrieve latest equity model from API
def get_model_info(
    ticker,
    auth_headers,
    log: LogFile,
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
):
    """
    Retrieve latest equity model information and historical datapoints endpoint url
    """
    model_info = {}

    company_url = f"{mds_host}/{CSIN_URL.format(ticker=quote_plus(ticker))}"

    company_response = get_request_json_content(
        company_url, auth_headers, log, verify_ssl
    )
    json = company_response.get("results")[0]
    csin = json.get("csin")
    company = json.get("company", {}).get("name")
    latest_version = (
        json.get("latest_equity_model", {}).get("model_version").get("name")
    )
    earnings_update_type = json.get("latest_equity_model", {}).get(
        "earnings_update_type"
    )
    publish_date = json.get("latest_equity_model", {}).get("published_at")

    periods = get_historical_periods(
        csin,
        latest_version,
        auth_headers,
        mds_host,
        log,
        HISTORICAL_PERIODS_TO_FETCH,
        verify_ssl,
    )

    model_info[ticker] = (
        csin,
        company,
        latest_version,
        periods,
        earnings_update_type,
        publish_date,
    )

    return_string = (
        f"{mds_host}/api/equity-model-series/{csin}/equity-models/"
        f"{latest_version}/historical-data-points/?page_size=500"
    )

    return return_string, model_info


def get_historical_periods(
    csin,
    version,
    auth_headers,
    mds_host,
    log,
    num_of_periods_to_fetch=HISTORICAL_PERIODS_TO_FETCH,
    verify_ssl=settings.VERIFY_SSL,
):
    periods_url = f"{mds_host}/{PERIODS_URL.format(csin=csin, version=version)}"
    periods_response = get_request_json_content(
        periods_url, auth_headers, log, verify_ssl
    )
    periods_json = periods_response.get("results")
    periods = [period.get("name") for period in periods_json[:num_of_periods_to_fetch]]

    return periods


def write_json(json_data, file_name):
    """
    Write JSON to file
    """
    with open(file_name, "w") as f:
        json.dump(json_data, f)


def crawl_company_pages(
    next_url,
    ticker,
    key_name,
    next_name,
    auth_headers,
    logger: LogFile,
    default_dir=settings.DEFAULT_DIR,
    verify_ssl=settings.VERIFY_SSL,
):
    """
    Retreive data from url and subsequent pages.
    """
    file_ticker = ticker.split(" ")[0]

    page_number = 1
    while next_url is not None:
        response_content = get_request_json_content(
            next_url, auth_headers, logger, verify_ssl
        )
        response = response_content.get(key_name)
        next_url = response_content.get(next_name)
        files_path = f"{default_dir}/DATA/{file_ticker}"
        os.makedirs(files_path, exist_ok=True)
        file_name = f"{files_path}/{page_number}.json"
        write_json(response, file_name)
        page_number += 1
    return

    # get api json data


def read_json(ticker, default_dir=settings.DEFAULT_DIR):
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


def json_to_df(dict_json, ticker):
    """
    Creates a dataframe out of JSON data
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


def refresh_cache(
    config: Config = None, log: LogFile = None, s3_client: Getter = None
) -> None:
    """
    Refresh the cache by deleting and pulling files from S3
    """
    if not config:
        config = settings.CONFIG or resolve_config()
    if not log:
        log = LogFile(default_dir=config.default_dir)
    if not s3_client:
        s3_client = Getter(config=config)

    folder = "DATA"
    subfolders, files = scandir(folder, [".csv"])
    for filename in files:
        try:
            print("Candas: refresh " + filename)
            os.remove(filename)
            s3_client.get_csv_from_s3(filename)
        except:
            log.write("Candas: get error for " + filename)
    return


def get_cache_tickers(default_dir=settings.DEFAULT_DIR):

    folder = "DATA"
    folder = f"{default_dir}/{folder}"
    subfolders, files = scandir(folder, [".xlsx"])
    ticker_list = []
    for filename in files:
        if "data" not in filename:
            ticker_list.append(filename)
    return ticker_list


def get_candas_ticker_list(ticker="", config=None):
    list_files = []
    ticker = ticker.split(" ")[0]

    if not config:
        config = settings.CONFIG or resolve_config()

    session = Session(
        aws_access_key_id=config.s3_access_key_id,
        aws_secret_access_key=config.s3_secret_key,
    )
    s3 = session.resource("s3")
    your_bucket = s3.Bucket("candas")
    for s3_file in your_bucket.objects.all():
        str_key = s3_file.key
        try:
            str_key = str_key.split("/")[1]
            if str_key != "DATA":
                list_files.append(str_key)
        except:
            continue
    list_files = list(set(list_files))
    if ticker != "":
        return list_files.count(ticker)
    else:
        return list_files


def find_in_list(feature_list, search_term):
    """
    Finds search term in items from passed in list
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


def get_sample_drivers(ticker: str):
    """
    Return the drivers for the sample models
    """
    driver_path = SAMPLE_MODEL_PATH["drivers_path"]
    path_name = f"{driver_path}/{ticker}_drivers.json"
    with open(path_name, "r") as file:
        json_data = json.load(file)
    return json_data["results"]


def get_drivers_from_api(
    mds_host, csin, model_version, api_headers, logger, verify_ssl
):
    """
    Returns a list of drivers for the model with the specified CSIN and model version
    """
    driver_list = []

    forecast_period_name = get_forecast_period_name(
        mds_host, csin, model_version, api_headers, logger, verify_ssl
    )

    driver_url = "api/equity-model-series/{csin}/equity-models/{model_version}/forecast-periods/{period}/data-points/?is_driver=true&page_size=200"
    driver_endpoint = f"{mds_host}/{driver_url.format(csin=quote_plus(csin), model_version=quote_plus(model_version), period=quote_plus(forecast_period_name))}"
    next_url: Optional[str] = driver_endpoint

    while next_url is not None:
        json = get_request_json_content(
            next_url, headers=api_headers, logger=logger, verify_ssl=verify_ssl
        )
        drivers = json.get("results")
        next_url = json.get("next")
        driver_list.extend(drivers)

    return driver_list


def get_forecast_period_name(
    mds_host,
    csin,
    model_version,
    api_headers,
    logger,
    verify_ssl=settings.VERIFY_SSL,
):
    """
    Retrieve the first forecast period name
    """
    forecast_url = (
        "api/equity-model-series/{csin}/equity-models/{model_version}/forecast-periods/"
    )
    forecast_endpoint = f"{mds_host}/{forecast_url.format(csin=quote_plus(csin), model_version=quote_plus(model_version))}"
    try:
        response = requests.get(forecast_endpoint, headers=api_headers)
    except:
        print(f"Candas: Error with getting forecast period.")
        return

    response_content = get_request_json_content(
        forecast_endpoint, api_headers, logger, verify_ssl
    )

    forecast_period_name = response_content["results"][0]["name"]

    return forecast_period_name


def get_request_json_content(url, headers, logger, verify_ssl=settings.VERIFY_SSL):
    """
    Make a GET request to a URL that responds with JSON content

    Returns the deserialized response content
    """
    return get_request(url, headers, logger, verify_ssl).json()


def get_chart_brand_config(
    chart_colors: List[str] = BRAND_CONFIG_DEFAULTS["chart_plot_colors"],
    figure_bg_color: str = BRAND_CONFIG_DEFAULTS["figure_bg_color"],
    vertical_line_color: str = BRAND_CONFIG_DEFAULTS["vertical_line_color"],
    ax_spine_color: str = BRAND_CONFIG_DEFAULTS["ax_spine_color"],
    title_font_path: str = BRAND_CONFIG_DEFAULTS["title_font_path"],
    body_font_path: str = BRAND_CONFIG_DEFAULTS["body_font_path"],
    chart_bg_color: str = BRAND_CONFIG_DEFAULTS["chart_bg_color"],
    font_color: str = BRAND_CONFIG_DEFAULTS["font_color"],
    logo_path: str = BRAND_CONFIG_DEFAULTS["logo_path"],
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


def _validate_colors(
    colors,
    figure_bg_color,
    vertical_line_color,
    ax_spine_color,
    chart_bg_color,
):
    """
    Validate color arguments
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
