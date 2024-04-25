# Canalyst Data Science Library

Canalyst's Python SDK is intended as a full featured data science library facilitating dataframe manipulation, data model mapping, and interactive scenario analysis of core Model, ModelSet, and ModelMap objects.

# Python Version Requirement
This package requires Python version `>=3.7`. Recommended Python version `>=3.8`.

# Usage

## Install Required Packages
Run `pip install -r requirements.txt` to install the requried packages.

## To Preview Sample Data
The Python SDK supports a preview of sample data for three tickers: DIS US, TSLA US, or NFLX US.  
```
import canalyst_candas as cd
model = cd.SampleModel(ticker="DIS US") 
df = model.model_frame()
```

## Configuration
The Python SDK supports three ways of providing configuration, in order of precedence:
   1. [Creating a `Config` instance](#using-config)
   2. [Using environment variables](#using-environment-variables)
   3. [Using a configuration file](#using-configuration-file)

`Config` is used to store a user's configuration information. On initial `canalyst_candas` import, it will attempt to retrieve the configuration.

### Using `Config`
A `Config` instance can be created with the desired configurations.

In Python/iPython or Jupyter Notebook. Replace `<..>` with values

```
import sys
sys.path.append('<path to sdk-python>/src')
import canalyst_candas as cd
from canalyst_candas.configuration.config import Config

# if you want to set a default directory, otherwise `default_dir` will be a temporary directory
# from pathlib import Path
# home = Path.home()
# default_dir = f"{home}/<any place under your home directory>"

config_info = {
  "canalyst_api_key": "<Canalyst API key>",
  "s3_access_key_id": "<S3 Access Key ID>",
  "s3_secret_key": "<S3 Secret Key>",
  "fred_key": "<Fred key>",
  "default_dir": "",
  "mds_host": "",
  "wp_host": "",
  "verify_ssl": "True",
}

config = Config(config=config_info)

ticker = "<ticker you want test with>"
model = cd.Model(ticker=ticker, config=config)
model_set = cd.ModelSet(ticker_list=[ticker], config=config)
cd.ModelMap(
  ticker=ticker,
  config=config,
  time_series_name=time_series,
  col_for_labels=<label>,
  common_size_tree=True,
  notebook=False,
  auto_download=False,
)
```

### Using Environment Variables
Environment variables can be set in your terminal. The SDK looks for
- CANALYST_API_KEY
- S3_ACCESS_KEY_ID
- S3_SECRET_KEY
- FRED_KEY (optional)
- DEFAULT_DIR (optional; default is a temporary directory)
- MDS_HOST (optional; default is production Model Data System)
- WP_HOST (optional; default is production Web Portal)
- VERIFY_SSL (optional: default is "True")

In Python/iPython or Jupyter Notebook. Replace `<..>` with values

```
import sys
sys.path.append('<path to sdk-python>/src')
import canalyst_candas as cd

ticker = "<ticker to test with>"
time_series = "<time series to test with>"
label = "<label of interest>
model = cd.Model(ticker=ticker)
model_set = cd.ModelSet(ticker_list=[ticker])
cd.ModelMap(
  ticker=ticker,
  time_series_name=time_series,
  col_for_labels=<label>,
  common_size_tree=True,
  notebook=False,
  auto_download=False,
)
```

### Using a Configuration File
When `canalyst_candas` is imported and if there is not already an existing configuration file, a configuration file, `keys.json`, is created in the user's home directory, `C:\Users\<username>\canalyst\keys.json` on Windows or `/User/<username>/canalyst/keys.json` on a Unix-based OS (e.g. Mac, Linux). The configuration file can be moved but must be under a user's home directory and must be in directly under a 'canalyst' folder. Examples of valid locations:

Windows
- `C:\Users\<username>\canalyst\keys.json`
- `C:\Users\<username\Downloads\canalyst\keys.json'`
  
Unix-based OS
- `/Users/<username/canalyst/keys.json`
- `/Users/<username>/Downloads/canalyst/keys.json`

The contents of `keys.json`:
```
{
    "canalyst_api_key": "",
    "canalyst_s3_id": "",
    "canalyst_s3_key": "",
    "fred_key": "",
    "default_dir": "",
    "mds_host": "",
    "wp_host": "",
    "verify_ssl": "",
    "proxies": ""
}
```

Fill in 
1. `canalyst_api_key`: Canalyst API token. Get it at https://app.canalyst.com/settings/api-tokens
2. `canalyst_s3_id`:  AWS S3 Access Key ID
3. `canalyst_s3_key`:  AWS S3 Secret Key
4. (Optional) `default_dir`: your chosen directory for where downloads will go. The default is a temporary directory.

In Python/iPython or Jupyter Notebook. Replace `<..>` with values

```
import sys
sys.path.append('<path to sdk-python>/src')
import canalyst_candas as cd

ticker = "<ticker to test with>"
time_series = "<time series to test with>"
label = "<label of interest>
model = cd.Model(ticker=ticker)
model_set = cd.ModelSet(ticker_list=[ticker])
cd.ModelMap(
  ticker=ticker,
  time_series_name=time_series,
  col_for_labels=<label>,
  common_size_tree=True,
  notebook=False,
  auto_download=False,
)
```

## Proxy Support
The Candas library allows for proxies to be passed in as part of the request. The proxies are to be formatted as a python dictionary:

`{"http": "http://http_proxy.com:8000", "https": "https://https_proxy.com:8001"}`

If you only have one proxy, the `http` and `https` keys can have the same proxy link.
This dictionary is to be passed in as part of the config object: 

```
    "proxies": {
        "http": "http://18.206.117.131:3128",
        "https": "http://18.206.117.131:3128"
      }
```