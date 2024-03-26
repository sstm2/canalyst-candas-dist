from pathlib import Path
from canalyst_candas.configuration.config import resolve_config, ConfigException
import json
import tempfile

DEFAULT_DIR = tempfile.gettempdir()
ROOT_DIR = Path().resolve()
MDS_HOST = "https://mds.canalyst.com"
WP_HOST = "https://app.canalyst.com"
VERIFY_SSL = True


def create_config():
    """
    Create configuration and set to CONFIG for Candas use
    """
    try:
        return resolve_config()
    except ConfigException:
        new_path = Path.home() / "canalyst"
        if not Path(new_path).exists():
            Path(new_path).mkdir()
        config_file = new_path / "keys.json"

        config_file_json = {
            "canalyst_api_key": "",
            "s3_access_key_id": "",
            "s3_secret_key": "",
            "fred_key": "",
            "default_dir": "",
            "mds_host": "",
            "wp_host": "",
            "verify_ssl": True,
        }
        try:
            config_file.write_text(json.dumps(config_file_json, indent=2))
            print(
                "A configuration file has been created for you in \n" f"{config_file}."
            )
        except:
            print("Failed to create config file in \n" f"{config_file}.")


CONFIG = create_config()
