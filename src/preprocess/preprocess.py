import os
from datetime import datetime

import pandas as pd

from utils.utils import init_dirs


class WatchLogPreprocessor:
    def __init__(self, args):
        self.args = args
        self.base_date = datetime.strptime(args.base_date, "%Y-%m-%d")

        self.dst = os.path.join(self.args.dataset_dir, "train")
        init_dirs(self.dst)

    @property
    def filename(self) -> str:
        return f"prepared_{self.args.dataset_name}_train_{self.args.model_name}.snappy.parquet"

    def load_dataset(self):
        return pd.read_csv(os.path.join(self.args.dataset_dir, f"{self.args.dataset_name}.csv"))

    def run(self):
        df = self.load_dataset()
        df.to_parquet(os.path.join(self.dst, self.filename), index=False)


class WatchLogNCFPreprocessor(WatchLogPreprocessor):
    def __init__(self, args):
        super().__init__(args)
