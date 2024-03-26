"""
Logging utilities
"""

import os
import datetime
import csv
from typing import Any, Iterable

import pandas as pd

from canalyst_candas import settings


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
