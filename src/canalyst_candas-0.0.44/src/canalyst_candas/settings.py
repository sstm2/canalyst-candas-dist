from pathlib import Path
from enum import Enum
import pkg_resources
import tempfile

EQUITY_MODEL_URL = "api/equity-model-series/{csin}/equity-models/{version}/"  # TODO: possible unused setting: delete if necessary
CSIN_URL = "api/equity-model-series/?company_ticker_bloomberg={ticker}"
PERIODS_URL = (
    "api/equity-model-series/{csin}/equity-models/{version}/historical-periods/"
)
SCENARIO_URL = "api/equity-model-series/{csin}/equity-models/{version}/scenarios/"
SCENARIO_URL_FORECAST = (  # TODO: possible unused setting: delete if necessary
    "api/equity-model-series/{csin}/equity-models/"
    "{version}/scenarios/{scenario_id}/forecast-periods/"
)
BULK_DATA_URL = "api/equity-model-series/{csin}/equity-models/{version}/bulk-data/{bulk_data_key}.{file_type}"

TIME_SERIES_URL = "api/equity-model-series/{csin}/equity-models/latest/time-series/{time_series_name}/"

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

SAMPLE_MODEL_PATH = {
    "drivers_path": pkg_resources.resource_filename(__name__, "sample_data/drivers"),
    "models_path": pkg_resources.resource_filename(__name__, "sample_data/models"),
}

DEFAULT_DIR = tempfile.gettempdir()
ROOT_DIR = Path().resolve()
MDS_HOST = "https://mds.canalyst.com"
WP_HOST = "https://app.canalyst.com"
VERIFY_SSL = True
PROXIES = None
