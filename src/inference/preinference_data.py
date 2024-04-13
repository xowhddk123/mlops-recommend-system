import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np

from utils.utils import init_dirs


class PreInferenceData:
    def __init__(self, args):
        self.args = args
        self.base_date = datetime.strptime(args.base_date, "%Y-%m-%d")
        self.src = os.path.join(self.args.dataset_dir, "train")
        self.dst = os.path.join(self.args.dataset_dir, "inference")
        init_dirs(self.src, self.dst)
        logging.info(f"dataset_dir : {self.args.dataset_dir}")
        logging.info(f"src : {self.src}")
        logging.info(f"dst : {self.dst}")

    def load_dataset(self):
        return pd.read_parquet(
            os.path.join(
                self.src,
                f"{self.args.dataset_name}_train_{self.args.model_name}.snappy.parquet"
            )
        )

    def process(self, df):
        raise NotImplementedError

    def run(self):
        df = self.load_dataset()        
        self.process(df)
        logging.info("success all process!")


class PreInferenceNCFData(PreInferenceData):
    def __init__(self, args):
        super().__init__(args)

    def process(self, df):
        df = pd.DataFrame(df, columns=["user_id", "contents_code"])
        grouped_df = df["contents_code"].groupby(df["user_id"])

        test_data = [
            [user_id, s]
            for user_id, item_id_list in grouped_df
            for s in np.random.choice(
                list(set(df["contents_code"].unique()) - set(item_id_list.values)),
                self.args.test_num_ng,
                replace=False,
            )
        ]

        test_df = pd.DataFrame(test_data, columns=["user_id", "contents_code"])
        test_df.to_parquet(
            os.path.join(self.dst, f"{self.args.dataset_name}_inference_{self.args.model_name}.snappy.parquet"),
            index=False
        )