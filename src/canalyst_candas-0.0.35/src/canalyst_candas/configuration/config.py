"""
Configs module
"""
import os
import json
import re
from typing import Dict, List, Optional
from pathlib import Path, PosixPath
from pydantic import BaseModel
from canalyst_candas import settings
from canalyst_candas.configuration.exceptions import ConfigException


def resolve_config(config: "Config" = None) -> "Config":
    """
    Extract configuration information

    Extracts the information in three ways:
    1. From a passed in Config object
    2. From environment variables
    3. From a configuration file found in the user's home directory
    """
    if config:
        return config

    envar_provider = EnvarProvider()
    try:
        envar_config = envar_provider.extract_config()
    except ConfigException as envar_config_err:
        raise ConfigException(
            "Environment variable configuration incorrectly defined"
        ) from envar_config_err

    if envar_config:
        return Config(config=envar_config)

    file_provider = FileProvider()
    try:
        file_config = file_provider.extract_config()
    except ConfigException as file_config_err:
        raise ConfigException(
            "Configuration file incorrectly defined"
        ) from file_config_err

    if file_config:
        return Config(config=file_config)

    raise ConfigException("No configuration was provided")


class SdkConfig(BaseModel):
    """
    This class is to establish the possible configurations with their types
    """

    canalyst_api_key: str
    s3_access_key_id: str
    s3_secret_key: str
    fred_key: Optional[str]
    default_dir: Optional[str]
    mds_host: Optional[str]
    wp_host: Optional[str]
    verify_ssl: Optional[bool]


class Config(object):
    """
    Configuration to use with the SDK
    """

    def __init__(self, config: Dict[str, str]) -> None:
        self.canalyst_api_key: Optional[str] = config.get("canalyst_api_key")
        self.s3_access_key_id: Optional[str] = config.get("s3_access_key_id")
        self.s3_secret_key: Optional[str] = config.get("s3_secret_key")
        self.fred_key = config.get("fred_key") or ""
        self.default_dir = config.get("default_dir") or settings.DEFAULT_DIR
        self.mds_host = config.get("mds_host") or settings.MDS_HOST
        self.wp_host = config.get("wp_host") or settings.WP_HOST
        self.verify_ssl = config.get("verify_ssl", True)


class ConfigProvider(object):
    """
    A base class for the various configuration providers

    Not to be implemented directly
    """

    def extract_config(self) -> Optional[Dict[str, str]]:
        """
        To be implemented by each provider that inherits from this class
        """
        raise NotImplementedError


class EnvarProvider(ConfigProvider):
    """
    Environment variables configuration provider
    """

    CANALYST_API_KEY = "CANALYST_API_KEY"
    S3_ACCESS_KEY_ID = "S3_ACCESS_KEY_ID"
    S3_SECRET_KEY = "S3_SECRET_KEY"
    FRED_KEY = "FRED_KEY"
    MDS_HOST = "MDS_HOST"
    WP_HOST = "WP_HOST"
    DEFAULT_DIR = "DEFAULT_DIR"
    VERIFY_SSL = "VERIFY_SSL"

    def __init__(
        self, environ: os._Environ = None, mapping: Dict[str, str] = None
    ) -> None:
        self.environ = environ or os.environ
        self._mapping = self._build_mapping(mapping)

    def _build_mapping(self, mapping: Optional[Dict[str, str]]) -> Dict[str, str]:
        """
        Map variable names to environment variables
        """
        if mapping is None:
            return {
                "canalyst_api_key": self.CANALYST_API_KEY,
                "s3_access_key_id": self.S3_ACCESS_KEY_ID,
                "s3_secret_key": self.S3_SECRET_KEY,
                "fred_key": self.FRED_KEY,
                "mds_host": self.MDS_HOST,
                "wp_host": self.WP_HOST,
                "default_dir": self.DEFAULT_DIR,
                "verify_ssl": self.VERIFY_SSL,
            }

        return {
            "canalyst_api_key": mapping.get("canalyst_api_key", self.CANALYST_API_KEY),
            "s3_access_key_id": mapping.get("s3_access_key_id", self.S3_ACCESS_KEY_ID),
            "s3_secret_key": mapping.get("s3_secret_key", self.S3_SECRET_KEY),
            "fred_key": mapping.get("fred_key", self.FRED_KEY),
            "mds_host": mapping.get("mds_host", self.MDS_HOST),
            "wp_host": mapping.get("wp_host", self.WP_HOST),
            "default_dir": mapping.get("default_dir", self.DEFAULT_DIR),
            "verify_ssl": mapping.get("verify_ssl", self.VERIFY_SSL),
        }

    def extract_config(self) -> Optional[Dict[str, str]]:
        """
        Return the configuration information from environment variables

        Check first to see if an API key exists before checking other variables.

        Raises ConfigException if
        - S3 access key ID not provided
        - S3 secret key not provided
        """
        canalyst_api_key = self.environ.get(self._mapping["canalyst_api_key"])
        if not canalyst_api_key:
            return None

        s3_access_key_id = self.environ.get(self._mapping["s3_access_key_id"])
        if not s3_access_key_id:
            raise ConfigException("S3 access key ID was not provided")

        s3_secret_key = self.environ.get(self._mapping["s3_secret_key"])
        if not s3_secret_key:
            raise ConfigException("S3 secret key was not provided")

        fred_key = self.environ.get(self._mapping["fred_key"], "")
        default_dir = self.environ.get(self._mapping["default_dir"], "")
        mds_host = self.environ.get(self._mapping["mds_host"], "")
        wp_host = self.environ.get(self._mapping["wp_host"], "")

        verify_ssl = self.environ.get(self._mapping["verify_ssl"], "True") in [
            "True",
            "true",
            "",
        ]

        return {
            "canalyst_api_key": canalyst_api_key,
            "s3_access_key_id": s3_access_key_id,
            "s3_secret_key": s3_secret_key,
            "fred_key": fred_key,
            "default_dir": default_dir,
            "mds_host": mds_host,
            "wp_host": wp_host,
            "verify_ssl": verify_ssl,
        }


class FileProvider(ConfigProvider):
    """
    Configuration file provider
    """

    def extract_config(self) -> Optional[Dict[str, str]]:
        """
        Extract configuration from the configuration file

        If the config file is not in the default location, search the
        user's home directory for it. If there are multiple matching
        files, use the first one found.

        Raises ConfigException if
        - the file is not found
        - the file is not formatted properly in JSON
        - S3 access ID not provided
        - S3 secret key not provided
        """
        config_file = f"{Path.home()}/canalyst/keys.json"
        possible_configs: List[Path] = []

        if len(possible_configs) > 1:
            new_line_point = "\n - "
            files = [str(_file) for _file in possible_configs]
            print(
                f"Found multiple configuration files:\n - "
                f"{new_line_point.join(files)}\n\n"
                f"Using {possible_configs[0]}.\n\n"
                "If this not the desired configuration file, remove it. If using a \n"
                "Jupyter Notebook, stop and restart the notebook for the changes to \n"
                "take effect. If using a Python/iPython session, quit the current \n"
                "session and start a new one."
            )

        try:
            with open(config_file) as _file:
                config = json.load(_file)
        except (json.JSONDecodeError, PermissionError, FileNotFoundError):
            raise ConfigException("Configuration file unreadable or unreachable")

        if not config.get("canalyst_api_key"):
            return None

        # "canalyst_s3_id" is the old configuration variable
        if not config.get("s3_access_key_id") and not config.get("canalyst_s3_id"):
            raise ConfigException("S3 access key ID was not provided")

        # "canalyst_s3_key" is the old configuration variable
        if not config.get("s3_secret_key") and not config.get("canalyst_s3_key"):
            raise ConfigException("S3 secret key was not provided")

        verify_ssl = config.get("verify_ssl", "True") in ["true", "True", ""]

        return {
            "canalyst_api_key": config.get("canalyst_api_key"),
            "s3_access_key_id": config.get("s3_access_key_id")
            or config.get("canalyst_s3_id"),
            "s3_secret_key": config.get("s3_secret_key")
            or config.get("canalyst_s3_key"),
            "fred_key": config.get("fred_key", ""),
            "mds_host": config.get("mds_host", ""),
            "wp_host": config.get("wp_host", ""),
            "default_dir": config.get("default_dir", ""),
            "verify_ssl": verify_ssl,
        }
