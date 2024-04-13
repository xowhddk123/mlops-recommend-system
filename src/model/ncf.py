import os
import logging
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from utils.utils import init_dirs
from dataset.data_utils import DefaultDataset, DefaultDataGenerator


class NCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num, num_layers, dropout):
        logging.info(f"MODEL Item num : {item_num}")
        super(NCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors;
        num_layers: the number of layers in MLP model;
        dropout: dropout rate between fully connected layers;
        """
        self.dropout = dropout

        self.embed_user_GMF = nn.Embedding(user_num, factor_num)
        self.embed_item_GMF = nn.Embedding(item_num, factor_num)
        self.embed_user_MLP = nn.Embedding(
            user_num, factor_num * (2 ** (num_layers - 1)))
        self.embed_item_MLP = nn.Embedding(
            item_num, factor_num * (2 ** (num_layers - 1)))

        mlp_modules = []
        for i in range(num_layers):
            input_size = factor_num * (2 ** (num_layers - i))
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, input_size // 2))
            mlp_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*mlp_modules)
        predict_size = factor_num * 2
        self.predict_layer = nn.Linear(predict_size, 1)
        self.final_act = nn.Sigmoid()
        self._init_weight_()

    def _init_weight_(self):
        """We leave the weights initialization here."""
        nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
        nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
        nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity="sigmoid")

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user, item):
        embed_user_gmf = self.embed_user_GMF(user)
        embed_item_gmf = self.embed_item_GMF(item)
        output_gmf = embed_user_gmf * embed_item_gmf
        embed_user_mlp = self.embed_user_MLP(user)
        embed_item_mlp = self.embed_item_MLP(item)
        interaction = torch.cat((embed_user_mlp, embed_item_mlp), -1)
        output_mlp = self.MLP_layers(interaction)

        concat = torch.cat((output_gmf, output_mlp), -1)

        prediction = self.predict_layer(concat)
        return prediction.view(-1)


class NCFDataset(DefaultDataset):
    def __init__(self, features, num_item, user_items, num_ng=0, is_training=None):
        super(NCFDataset, self).__init__()
        self.features_ps = features
        self.num_item = num_item
        self.total_items = np.arange(num_item)
        self.user_items = user_items
        self.num_ng = num_ng
        self.is_training = is_training

    def __len__(self):
        return len(self.features_ps)

    def __getitem__(self, idx):
        user = self.features_ps[idx][0]
        item = self.features_ps[idx][1]
        return user, item

    def _get_batch_ng_sample(self, user, size):
        idx = 0
        indices = []
        ng_sample = np.random.choice(self.total_items, size=size)

        while len(indices) != self.num_ng and idx < size:
            if ng_sample[idx] in self.user_items[user][:-1]:  # if watched then pass
                idx += 1
                continue

            indices.append(idx)
            idx += 1

        ng_sample = ng_sample[np.array(indices, dtype=np.int32)]
        return ng_sample

    def _get_ng_sample(self, user, tries=5):
        ng_item = np.random.choice(self.total_items, size=1)
        while ng_item in self.user_items[user][:-1] and tries > 0:
            ng_item = np.random.choice(self.total_items, size=1)
            tries -= 1
        return ng_item

    def collate_fn(self, batch):
        batch_size = len(batch)
        collate_size = batch_size * (1 + self.num_ng)

        users = np.empty(collate_size, dtype=np.int32)
        items = np.empty(collate_size, dtype=np.int32)
        labels = np.empty(collate_size, dtype=np.int32)

        for i, (user, item) in enumerate(batch):
            ng_sample = self._get_batch_ng_sample(user, size=self.num_ng * 3)

            while len(ng_sample) != self.num_ng:
                np.append(ng_sample, self._get_ng_sample(user))

            s_idx = i * (1 + self.num_ng)
            e_idx = (i + 1) * (1 + self.num_ng)

            users[s_idx:e_idx] = np.array([user] * (1 + self.num_ng))

            items[s_idx] = item
            items[s_idx + 1 : e_idx] = ng_sample

            labels[s_idx] = 1
            labels[s_idx + 1 : e_idx] = np.array([0] * self.num_ng)

        users = torch.tensor(users, dtype=torch.long)
        items = torch.tensor(items, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return users, items, labels


class NCFDataGenerator(DefaultDataGenerator):
    def __init__(self, args):
        super(NCFDataGenerator, self).__init__(args)

        self._base_date = datetime.strptime(args.base_date, "%Y-%m-%d")

        self._train_data = None
        self._valid_data = None
        self._user_index = None
        self._item_index = None
        self._user_num = None
        self._user_items = None
        self._item_num = None

        self._src = os.path.join(self._args.dataset_dir, "train")

        self._dst = os.path.join(self._args.output_dir, "data", "index")

        init_dirs(self._src, self._dst)

    @property
    def train_data(self):
        return self._train_data

    @property
    def valid_data(self):
        return self._valid_data

    @property
    def user_num(self):
        return self._user_num

    @property
    def user_items(self):
        return self._user_items

    @property
    def item_num(self):
        return self._item_num

    def download(self):
        logging.info("Download NCF data")
        self._train_data = pd.read_parquet(
            os.path.join(
                self._src,
                f"{self._args.dataset_name}_train_{self._args.model_name}.snappy.parquet"
            ),
        )
        logging.info("Download Success!")
        logging.info(self._train_data.head())

    def save_index(self):
        logging.info("Save Index")
        user_index_dst = os.path.join(self._dst, "user_index.csv")
        item_index_dst = os.path.join(self._dst, "item_index.csv")
        self._user_index.to_csv(user_index_dst, index=True)
        self._item_index.to_csv(item_index_dst, index=True)

        logging.info(f"user_index_dst : {user_index_dst}")
        logging.info(f"user_index_dst : {item_index_dst}")
        logging.info("Save Success!")

    def _preprocess_idx(self):
        logging.info("Preprocess Index")

        self._train_data = self._train_data[["user_id", "contents_code"]]

        logging.info(self._train_data.head())

        user_index = pd.DataFrame(self._train_data["user_id"].unique(), columns=["user"])
        item_index = pd.DataFrame(self._train_data["contents_code"].unique(), columns=["item"])

        reverse_user_index = user_index.reset_index().set_index("user")
        reverse_item_index = item_index.reset_index().set_index("item")

        self._train_data["user_id"] = self._train_data.user_id.apply(
            lambda user: reverse_user_index.loc[user]
        )
        self._train_data["contents_code"] = self._train_data.contents_code.apply(
            lambda item: reverse_item_index.loc[item]
        )

        self._train_data.columns = ["user", "item"]

        self._user_index = user_index
        self._item_index = item_index

    def preprocess(self):
        logging.info("Run NCF Data Preprocess")
        self._preprocess_idx()
        self.save_index()

        user_items = self._train_data.groupby("user")["item"].apply(list).to_dict()
        self._user_items = user_items

        self._user_num = self._train_data["user"].max() + 1
        self._item_num = self._train_data["item"].max() + 1

        logging.info("Data Split ...")
        valid_data = self._train_data.sample(frac=0.2)
        self._train_data.drop(valid_data.index, inplace=True)

        self._train_data = self._train_data.values.tolist()
        self._valid_data = valid_data.values.tolist()

        logging.info(f"user_num: {self._user_num}")
        logging.info(f"item_num: {self._item_num}")
        logging.info(f"Length of train_data: {len(self._train_data)}")
        logging.info(f"Length of valid_data: {len(self._valid_data)}")

    def get_top_k(self, top_k: int = 500):
        top_k_result, _ = zip(*Counter(self._train_data).most_common(n=top_k))
        return top_k_result

    def get_train_dataset(self):
        return NCFDataset(
            self._train_data, self._item_num, self._user_items, self._args.train_num_ng, True
        )

    def get_valid_dataset(self):
        return NCFDataset(
            self._valid_data, self._item_num, self._user_items, self._args.train_num_ng, True
        )

    def get_valid_metric_dataset(self):
        return NCFDataset(self._valid_data, self._item_num, self._user_items, 99, False)

    def get_train_loader(self, dataset, batch_size):
        batch_size = batch_size * self._args.num_gpus if self._args.num_gpus > 0 else batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            sampler=None,
            num_workers=self._args.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
        )

    def get_valid_loader(self, dataset, batch_size):
        batch_size = batch_size * self._args.num_gpus if self._args.num_gpus > 0 else batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            sampler=None,
            num_workers=self._args.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
        )

    def get_valid_metric_loader(self, dataset, batch_size):
        batch_size = batch_size * self._args.num_gpus if self._args.num_gpus > 0 else batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            sampler=None,
            num_workers=self._args.num_workers,
            collate_fn=dataset.collate_fn,
            pin_memory=True,
            drop_last=True,
        )
