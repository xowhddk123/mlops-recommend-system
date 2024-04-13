import os
import logging
import pprint
import random
import tarfile
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd

from model.ncf import NCFDataset
from utils.utils import init_dirs


class Inference:
    def __init__(self, args):
        self.args = args
        self.base_date = datetime.strptime(self.args.base_date, "%Y-%m-%d")
        self.device = torch.device("cuda:0" if self.args.use_cuda else "cpu")
        self.dataset_src = os.path.join(self.args.dataset_dir, "inference")
        self.dst = os.path.join(self.args.output_dir, "inference")
        init_dirs(self.dataset_src, self.dst)
        self._init_config()

    def _init_config(self):
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        cudnn.benchmark = True

    def load_inference_dataset(self, item_num):
        df = pd.read_parquet(
            os.path.join(
                self.dataset_src,
                f"{self.args.dataset_name}_inference_{self.args.model_name}.snappy.parquet"
            )
        )
        return NCFDataset(df.values.tolist(), item_num, None, 0, False)

    def load_model(self):
        gz = os.path.join(self.args.model_dir, "model.tar.gz")
        if os.path.exists(gz):
            with tarfile.open(gz) as f:
                f.extractall(self.args.model_dir)

        model = torch.load(
            os.path.join(self.args.model_dir, f"{self.args.model_name}_best_VAR.pth"),
            map_location=torch.device(self.device) if self.args.use_cuda else None,
        )

        if self.args.use_cuda:
            model.to(device=self.device)

        return model


class InferenceNCF(Inference):
    def __init__(self, args):
        super().__init__(args)
        self.user_index = None
        self.item_index = None
        self.reverse_user_index = None
        self.reverse_item_index = None
        self.index_src = os.path.join(self.args.output_dir, "data", "index")
        init_dirs(self.index_src)
        self._init_index()

    def _init_index(self):
        gz = os.path.join(self.index_src, "output.tar.gz")
        if os.path.exists(gz):
            with tarfile.open(gz) as f:
                f.extractall(os.path.join(self.args.output_dir, "data"))

        self.user_index = self.load_user_index()
        self.item_index = self.load_item_index()

        self.reverse_user_index = self.user_index.reset_index().set_index("user", inplace=False)
        self.reverse_item_index = self.item_index.reset_index().set_index("item", inplace=False)

    def load_item_index(self):
        return pd.read_csv(
            os.path.join(self.index_src, "item_index.csv"),
            index_col=0
        )

    def load_user_index(self):
        return pd.read_csv(
            os.path.join(self.index_src, "user_index.csv"),
            index_col=0
        )

    def inference(self, model, test_dataloader):
        user_item_list = dict()
        weight_activation = nn.Sigmoid()

        for user, item in test_dataloader:
            if self.args.use_cuda:
                user = user.to(device=self.device)
                item = item.to(device=self.device)
            user.apply_(lambda x: self.reverse_user_index.loc[x].item())
            item.apply_(lambda x: self.reverse_item_index.loc[x].item())
            predictions = model(user, item)
            predictions = weight_activation(predictions)
            _, indices = torch.topk(predictions, self.args.top_k)
            recommends = torch.take(item, indices)
            prediction_weights = (
                torch.take(predictions, indices).detach().cpu().numpy().tolist()
            )

            recommends.apply_(lambda x: self.item_index.loc[x].item())
            user_no = user[0].item()

            user_item_list[user_no] = [
                {"code": code.cpu().numpy().item(), "score": score}
                for code, score in zip(recommends, prediction_weights)
            ]

        logging.info("Inference Completed")
        logging.info(len(user_item_list.keys()))
        logging.info(pprint.pformat(list(user_item_list.items())[:10]))
        return user_item_list

    def run(self):
        logging.info(f"Base Date: {self.base_date}")
        warnings.filterwarnings("ignore")

        item_num = len(self.item_index)
        model = self.load_model()
        model.eval()
        logging.info(f"model : {model}")

        test_dataset = self.load_inference_dataset(item_num)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        logging.info("Data Load Completed")

        user_item_list = self.inference(
            model=model,
            test_dataloader=test_loader,
        )
        logging.info("Inference Completed")

        df = pd.DataFrame(data={"user_id": user_item_list.keys(), "items": user_item_list.values()})
        df.to_parquet(os.path.join(self.dst, "inference_result.snappy.parquet"), index=False)