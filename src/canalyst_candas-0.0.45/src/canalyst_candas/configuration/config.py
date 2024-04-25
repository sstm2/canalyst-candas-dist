"""
Configs module
"""
import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError, root_validator
from canalyst_candas import settings
from canalyst_candas.configuration.exceptions import ConfigException


def create_config():
    """
    Create configuration and set to CONFIG for Candas use
    """
    try:
        return resolve_config()
    except ConfigException as config_err:
        config_path = Path.home() / "canalyst"
        config_file = config_path / "keys.json"
        if not os.path.isfile(config_file):
            Path(config_path).mkdir(exist_ok=True)
            config_file_json = {
                "canalyst_api_key": "",
                "s3_access_key_id": "",
                "s3_secret_key": "",
                "fred_key": "",
                "default_dir": "",
                "mds_host": "",
                "wp_host": "",
                "verify_ssl": True,
                "proxies": "",
            }
            try:
                config_file.write_text(json.dumps(config_file_json, indent=2))
                print(
                    "A configuration file has been created for you here -> \n"
                    f"{config_file}."
                )
            except:
                print("Failed to create config file in \n" f"{config_file}.")

        raise config_err


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
    envar_config = envar_provider.extract_config()
    if envar_config:
        try:
            return Config(**envar_config)
        except ConfigException as envar_config_err:
            raise ConfigException(
                "Environment variable(s) configuration incorrectly defined"
            ) from envar_config_err

    file_provider = FileProvider()
    file_path = file_provider.extract_config()
    if file_path:
        try:
            return Config.parse_file(file_path)
        except ConfigException as file_config_err:
            raise ConfigException(
                "Configuration file incorrectly defined"
            ) from file_config_err
        except Exception as other_err:
            raise ConfigException from other_err

    raise ConfigException("No configuration was provided")


class Config(BaseModel):
    """
    Configuration validation model
    """

    def __init__(self, config: Dict[str, Any] = {}, **data) -> None:
        """
        Change validation to raise ConfigException
        """
        try:
            super().__init__(**config, **data)
        except ValidationError as error:
            raise ConfigException(error) from None

    canalyst_api_key: str
    s3_access_key_id: str = Field(..., alias="canalyst_s3_id")
    s3_secret_key: str = Field(..., alias="canalyst_s3_key")
    fred_key: str = ""
    default_dir: str = settings.DEFAULT_DIR
    mds_host: str = settings.MDS_HOST
    wp_host: str = settings.WP_HOST
    verify_ssl: Union[bool, str] = True
    proxies: Optional[Dict[str, str]] = None

    class Config:
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def remove_empty_values(cls, values):
        """
        Remove fields with empty values before validation to ensure
        the default values are set for preset fields
        """
        return {k: v for k, v in values.items() if v not in ["", None]}


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
    PROXIES = "PROXIES"

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
                "proxies": self.PROXIES,
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
            "proxies": mapping.get("proxies", self.PROXIES),
        }

    def extract_config(self) -> Optional[Dict[str, str]]:
        """
        Return the configuration information from environment variables
        """
        config = {}

        # only extract set configurations
        for config_var in self._mapping.keys():
            value = self.environ.get(self._mapping[config_var])
            if value is not None:
                config[config_var] = value

        return config


class FileProvider(ConfigProvider):
    """
    Configuration file provider
    """

    def extract_config(self) -> Optional[str]:
        """
        Get configuration file path

        If the config file is not in the default location, search the
        user's home directory for it. If there are multiple matching
        files, use the first one found.
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

        if os.path.isfile(config_file):
            return config_file
