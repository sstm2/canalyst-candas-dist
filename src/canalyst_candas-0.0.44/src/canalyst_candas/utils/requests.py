"""
Utilities for making resource requests 
"""
import botocore
import json
import pickle
import os
import datetime
import time
from io import StringIO, BytesIO
from urllib.parse import quote_plus
from typing import Dict, List, Optional, Union

import pandas as pd
import boto3
import requests
from boto3.session import Session
from botocore.exceptions import ClientError
from itertools import product
from python_graphql_client import GraphqlClient
from openpyxl import load_workbook

from canalyst_candas.exceptions import (
    BaseCandasException,
    AWSCredentialException,
    HTTPClientException,
)
from canalyst_candas.configuration.config import Config, resolve_config
from canalyst_candas import settings
from canalyst_candas.utils.logger import LogFile
from canalyst_candas.utils.transformations import (
    _get_mds_bulk_data_url,
    _get_mds_time_series_url,
    write_json,
    scandir,
    get_api_headers,
)


class Getter:
    """
    S3 Client to retreive data from S3
    NOTE: Separate s3 connection from getters. Consider moving s3 connection to utils/connections.py
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.log = LogFile(default_dir=self.config.default_dir)
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.config.s3_access_key_id,
            aws_secret_access_key=self.config.s3_secret_key,
            verify=config.verify_ssl,
            config=botocore.config.Config(proxies=self.config.proxies),
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
        try:
            d = self.s3_client.get_object(Bucket="candas", Key=file_name)
        except ClientError as e:
            raise AWSCredentialException(
                "Please check that the s3_access_key and s3_secret_key are correct"
            ) from e

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
            if os.path.exists(f"{path_name}"):
                df = pd.read_csv(f"{path_name}")

        if df is None:
            return None

        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        return df


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


def get_request(
    url, headers, logger, verify_ssl=settings.VERIFY_SSL, proxies=settings.PROXIES
):
    """
    Wrapper for a GET request

    Handles throttle limits by using a timer
    """
    response = requests.get(
        url=url, headers=headers, verify=verify_ssl, proxies=proxies
    )
    if check_throttle_limit(response, logger):
        response = requests.get(url=url, headers=headers, verify=False, proxies=proxies)
    if response.status_code == 401:
        raise HTTPClientException(
            f"HTTP Error code : {response.status_code}. Check Canalyst API key"
        )
    try:
        response.raise_for_status()
    except requests.RequestException as ex:
        logger.write(
            f"Candas: Error when making a request to '{url}'. " f"Exception: {ex}."
        )
        return

    return response


def post_request(url, headers, json, verify_ssl, proxies):
    """
    POST request helper function

    Uses timer to account for throttle limits.
    """
    response = requests.post(
        url=url, headers=headers, json=json, verify=verify_ssl, proxies=proxies
    )
    if check_throttle_limit(response):
        response = requests.post(
            url=url, headers=headers, json=json, verify=verify_ssl, proxies=proxies
        )

    return response


def get_request_json_content(
    url, headers, logger, verify_ssl=settings.VERIFY_SSL, proxies=settings.PROXIES
):
    """
    Make a GET request to a URL that responds with JSON content

    Returns the deserialized response content
    """
    return get_request(url, headers, logger, verify_ssl, proxies).json()


def set_api_key(key, key_file_path):
    """
    Loads API key from path

    TODO: possible unused function: delete if necessary
    """
    with open(key_file_path) as f:
        keys_json = json.load(f)

    keys_json["canalyst_api_key"] = key

    with open(key_file_path, "w") as f:
        json.dump(keys_json, f)


def get_test_data_set(csv_type: settings.BulkDataFileType, ticker: str):
    """
    Fetches data from the test data set in the models folder
    """
    csv_type = csv_type.value.replace("-", "_")
    csv_types = ["historical_data", "forecast_data", "name_index"]
    if csv_type not in csv_types:
        raise BaseCandasException("CSV type is not one of the available options")
    model_path = settings.SAMPLE_MODEL_PATH["models_path"]
    file_name = f"{model_path}/{ticker}_{csv_type}.csv"
    df = pd.read_csv(file_name)
    return df


def get_data_set_from_mds(
    bulk_data_key: settings.BulkDataKeys,
    file_type: str,
    csin: str,
    model_version: str,
    auth_headers: Dict[str, str],
    log: LogFile,
    mds_host: str = settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    """
    Pulls a bulk data file from the MDS at the equity-model endpoint and returns it in a dataframe.

    Different types are available, listed in the settings.BulkDataKeys enum. Different formats are also
    available, listed in the settings.BULK_DATA_KEY_FILE_TYPES based on the type.
    """

    available_file_types = settings.BULK_DATA_KEY_FILE_TYPES[bulk_data_key]
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
    response = get_request(url, auth_headers, log, verify_ssl, proxies)
    # parquet not available: fallback to csv
    if response is None:
        url = _get_mds_bulk_data_url(
            bulk_data_key,
            "csv",  # file_type set to csv
            csin,
            model_version,
            mds_host,
        )
        response = get_request(url, auth_headers, log, verify_ssl, proxies)
        file_type = "csv"

    if file_type == settings.BulkDataFileType.CSV.value:
        csv_string = response.content.decode("utf-8")
        df = pd.read_csv(StringIO(csv_string))
        return df
    elif file_type == settings.BulkDataFileType.PARQUET.value:
        df = pd.read_parquet(BytesIO(response.content))

        if bulk_data_key in [
            settings.BulkDataKeys.FORECAST_DATA,
            settings.BulkDataKeys.HISTORICAL_DATA,
        ]:
            # parquet values are set as strings by default, manually set to floats/ints
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
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


def check_time_series_name(
    df,
    time_series_name: Union[str, List[str]],
    auth_headers: Dict[str, str],
    log: LogFile,
    verify_ssl=settings.VERIFY_SSL,
) -> List[str]:
    """
    Checks for additional defined names for a time series

    The user has requested a time series name that exists but is not the first one in the names list
    """
    # convert both to list for easier processing
    csin = df.CSIN.unique().tolist()
    time_series_name = (
        [time_series_name] if type(time_series_name) is str else time_series_name
    )
    for csin_, ts in product(csin, time_series_name):
        time_series_url = _get_mds_time_series_url(csin_, ts)
        try:
            names = get_request_json_content(
                time_series_url, auth_headers, log, verify_ssl
            )["names"]
            if names[0] != ts and ts in names:  # valid name that is not the first name
                df.loc[
                    (df["time_series_name"] == names[0]) & (df["CSIN"] == csin_),
                    "time_series_name",
                ] = ts
        except AttributeError:
            continue

    return df


def get_company_info_from_ticker(
    ticker,
    auth_headers,
    logger: LogFile,
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    """
    Get CSIN and latest version for a ticker
    """
    company_url = f"{mds_host}/{settings.CSIN_URL.format(ticker=quote_plus(ticker))}"
    response_content = get_request_json_content(
        url=company_url,
        headers=auth_headers,
        logger=logger,
        verify_ssl=verify_ssl,
        proxies=proxies,
    )

    try:
        json = response_content.get("results")[0]
    except IndexError as e:
        raise BaseCandasException("No response returned. Invalid ticker input") from e

    csin: str = json.get("csin")
    latest_version: str = (
        json.get("latest_equity_model", {}).get("model_version", {}).get("name", {})
    )
    return (csin, latest_version)


def send_scenario(
    ticker,
    data,
    auth_headers,
    logger: LogFile,
    csin="",
    latest_version="",
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    """
    Makes a POST request for scenario to the MDS
    """
    if csin == "" or latest_version == "":
        csin, latest_version = get_company_info_from_ticker(
            ticker, auth_headers, logger, mds_host, verify_ssl, proxies
        )
    scenario_url = (
        f"{mds_host}/{settings.SCENARIO_URL.format(csin=csin, version=latest_version)}"
    )
    # print("scenario_url: " + scenario_url)
    try:
        scenario_response = post_request(
            url=scenario_url,
            headers=auth_headers,
            json=data,
            verify_ssl=verify_ssl,
            proxies=proxies,
        )
    except:
        return None
    # print(scenario_response.text)

    return scenario_response


def get_scenarios(
    ticker,
    auth_headers,
    logger: LogFile,
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    # TODO: possible unused function: delete if necessary
    if ticker == "":
        ticker = ticker
    url = f"{mds_host}/api/scenarios/"
    csin, latest_version = get_company_info_from_ticker(
        ticker,
        auth_headers,
        logger,
        mds_host,
        verify_ssl,
        proxies,
    )
    response_content = get_request_json_content(
        url, auth_headers, logger, verify_ssl, proxies
    )
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


def get_csin_from_ticker(
    ticker,
    auth_headers,
    logger: LogFile,
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    """
    Get CSIN for ticker
    """
    company_url = f"{mds_host}/{settings.CSIN_URL.format(ticker=quote_plus(ticker))}"
    response_content = get_request_json_content(
        company_url,
        auth_headers,
        logger,
        verify_ssl,
        proxies,
    )
    json = response_content.get("results")[0]
    csin = json.get("csin")
    return csin


def get_excel_model(config, ticker="", download_dir="", logger=None):
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
        ticker,
        auth_headers,
        logger,
        mds_host,
        config.verify_ssl,
        config.proxies,
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

    if os.path.exists(file_name):
        os.rename(file_name, backup_name)
        print("Backup to: " + backup_name)

    r = requests.get(
        url, headers=auth_headers, verify=config.verify_ssl, proxies=config.proxies
    )

    with open(file_name, "wb") as f:
        f.write(r.content)
    print("Saved to: " + file_name)

    return


def get_forecast_url_data(
    res_dict,
    ticker,
    auth_headers,
    log,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    list_out = []
    url = res_dict["self"]

    res_loop = get_request(url, auth_headers, log, verify_ssl, proxies)
    url = res_loop.json()["data_points"]

    res_loop = get_request(url, auth_headers, log, verify_ssl, proxies)
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
            res_loop = get_request(url, auth_headers, log, verify_ssl, proxies)
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
    proxies=settings.PROXIES,
):
    log = LogFile(default_dir=default_dir)
    url = res_dict["self"]

    res_loop = get_request(url, auth_headers, log, verify_ssl, proxies)
    url = res_loop.json()["data_points"]

    res_loop = get_request(url, auth_headers, log, verify_ssl, proxies)

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
            res_loop = get_request(url, auth_headers, log, verify_ssl, proxies)
            res_data = res_loop.json()["results"]
            url = res_loop.json()["next"]
        except:
            url = None
    return pd.concat(list_out)


def create_drivers_dot(
    time_series, auth_headers, ticker, logger: LogFile, config=None, s3_client=None
):
    if time_series == "":
        time_series = "net-revenue"

    file_ticker = ticker.split(" ")[0]

    path_name = f"DATA/{file_ticker}"

    if not config:
        config = resolve_config()
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
            ticker,
            auth_headers,
            logger,
            config.mds_host,
            config.verify_ssl,
            config.proxies,
        )
        url = (
            f"{config.mds_host}/api/equity-model-series/{csin}/equity-models/"
            f"{version}/time-series/{time_series}/forecast-data-points/"
        )

        r = requests.get(
            url, headers=auth_headers, verify=config.verify_ssl, proxies=config.proxies
        )
        json = r.json().get("results")[0]
        name = json["period"]["name"]
        url = (
            f"{config.mds_host}/api/equity-model-series/{csin}/equity-models/"
            f"{version}/time-series/{time_series}/forecast-data-points/{name}/"
            f"drivers/?format=dot"
        )

        print(url)
        r = requests.get(
            url, headers=auth_headers, verify=config.verify_ssl, proxies=config.proxies
        )
        with open(path_name, "wb") as f:
            f.write(r.content)
        return path_name


def get_historical_periods(
    csin,
    version,
    auth_headers,
    mds_host,
    log,
    num_of_periods_to_fetch=settings.HISTORICAL_PERIODS_TO_FETCH,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    periods_url = (
        f"{mds_host}/{settings.PERIODS_URL.format(csin=csin, version=version)}"
    )
    periods_response = get_request_json_content(
        periods_url,
        auth_headers,
        log,
        verify_ssl,
        proxies,
    )
    periods_json = periods_response.get("results")
    periods = [period.get("name") for period in periods_json[:num_of_periods_to_fetch]]

    return periods


def get_model_info(
    ticker,
    auth_headers,
    log: LogFile,
    mds_host=settings.MDS_HOST,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    """
    Retrieve latest equity model information and historical datapoints endpoint url
    """
    model_info = {}

    company_url = f"{mds_host}/{settings.CSIN_URL.format(ticker=quote_plus(ticker))}"

    company_response = get_request_json_content(
        company_url,
        auth_headers,
        log,
        verify_ssl,
        proxies,
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
        settings.HISTORICAL_PERIODS_TO_FETCH,
        verify_ssl,
        proxies,
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


def crawl_company_pages(
    next_url,
    ticker,
    key_name,
    next_name,
    auth_headers,
    logger: LogFile,
    default_dir=settings.DEFAULT_DIR,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    """
    Retreive data from url and subsequent pages.

    TODO: possible unused function: delete if necessary
    """
    file_ticker = ticker.split(" ")[0]

    page_number = 1
    while next_url is not None:
        response_content = get_request_json_content(
            next_url,
            auth_headers,
            logger,
            verify_ssl,
            proxies,
        )
        response = response_content.get(key_name)
        next_url = response_content.get(next_name)
        files_path = f"{default_dir}/DATA/{file_ticker}"
        os.makedirs(files_path, exist_ok=True)
        file_name = f"{files_path}/{page_number}.json"
        write_json(response, file_name)
        page_number += 1
    return


def refresh_cache(
    config: Config = None, log: LogFile = None, s3_client: Getter = None
) -> None:
    """
    Refresh the cache by deleting and pulling files from S3

    TODO: possible unused function: delete if necessary
    """
    if not config:
        config = resolve_config()
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


def get_candas_ticker_list(ticker="", config=None):
    # TODO: possible unused function: delete if necessary
    list_files = []
    ticker = ticker.split(" ")[0]

    if not config:
        config = resolve_config()

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


def get_forecast_period_name(
    mds_host,
    csin,
    model_version,
    api_headers,
    logger,
    verify_ssl=settings.VERIFY_SSL,
    proxies=settings.PROXIES,
):
    """
    Retrieve the first forecast period name
    """
    forecast_url = (
        "api/equity-model-series/{csin}/equity-models/{model_version}/forecast-periods/"
    )
    forecast_endpoint = f"{mds_host}/{forecast_url.format(csin=quote_plus(csin), model_version=quote_plus(model_version))}"
    try:
        response = requests.get(
            forecast_endpoint, headers=api_headers, verify=verify_ssl, proxies=proxies
        )
    except:
        print(f"Candas: Error with getting forecast period.")
        return

    response_content = get_request_json_content(
        forecast_endpoint, api_headers, logger, verify_ssl, proxies
    )

    forecast_period_name = response_content["results"][0]["name"]

    return forecast_period_name


def get_sample_drivers(ticker: str):
    """
    Return the drivers for the sample models
    """
    driver_path = settings.SAMPLE_MODEL_PATH["drivers_path"]
    path_name = f"{driver_path}/{ticker}_drivers.json"
    with open(path_name, "r") as file:
        json_data = json.load(file)
    return json_data["results"]


def get_drivers_from_api(
    mds_host,
    csin,
    model_version,
    api_headers,
    logger,
    verify_ssl,
    proxies,
):
    """
    Returns a list of drivers for the model with the specified CSIN and model version
    """
    driver_list = []

    forecast_period_name = get_forecast_period_name(
        mds_host, csin, model_version, api_headers, logger, verify_ssl, proxies
    )

    driver_url = "api/equity-model-series/{csin}/equity-models/{model_version}/forecast-periods/{period}/data-points/?is_driver=true&page_size=200"
    driver_endpoint = f"{mds_host}/{driver_url.format(csin=quote_plus(csin), model_version=quote_plus(model_version), period=quote_plus(forecast_period_name))}"
    next_url: Optional[str] = driver_endpoint

    while next_url is not None:
        json = get_request_json_content(
            next_url,
            headers=api_headers,
            logger=logger,
            verify_ssl=verify_ssl,
            proxies=proxies,
        )
        drivers = json.get("results")
        next_url = json.get("next")
        driver_list.extend(drivers)

    return driver_list


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
