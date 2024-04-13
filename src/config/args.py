import os
import argparse
from typing import List
from datetime import datetime

import torch

from utils.utils import get_root_dir


def subclasses_to_classnames(base_class) -> List[str]:
    return list(set(map(lambda x: x.__name__, base_class.__subclasses__())))


def parse_common_arguments(parser):
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--base_date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0", help="gpu card ID")
    parser.add_argument("--seed", type=int, default=0, help="seed number")
    parser.add_argument("--log_level", type=int, default=os.environ.get("SM_LOG_LEVEL", 20))
    parser.add_argument("--serve_contents_type", type=str, default="movie")
    parser.add_argument("--serve_recommend_type", type=str, default="like")


def parse_model_arguments(parser):
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--factor_num", type=int, default=64, help="predictive factors numbers in the model")
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers in MLP model")
    parser.add_argument("--model_name", type=str, default="NCF", help="save model dir name")
    parser.add_argument("--model_dir", type=str, default=os.path.join(get_root_dir(), "local", "model"))


def parse_train_arguments(parser):
    loss_function_names = subclasses_to_classnames(torch.nn.modules.loss._Loss)
    optimizer_names = subclasses_to_classnames(torch.optim.Optimizer)
    scheduler_names = subclasses_to_classnames(torch.optim.lr_scheduler._LRScheduler)
    scheduler_params = '{"milestones":[15,25,32,40,45],"gamma":0.5}'

    parser.add_argument("--loss_function_name", type=str, default="BCEWithLogitsLoss", choices=loss_function_names)
    parser.add_argument("--loss_function_params", type=str, default="{}")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--optimizer_name", type=str, default="Adam", choices=optimizer_names)
    parser.add_argument("--optimizer_params", type=str, default='{"lr":0.001}')
    parser.add_argument("--scheduler_name", type=str, default="MultiStepLR", choices=scheduler_names)
    parser.add_argument("--scheduler_params", type=str, default=scheduler_params)
    parser.add_argument("--epochs", type=int, default=2, help="training epochs")
    parser.add_argument("--checkpoint_path", type=str, default=os.path.join(get_root_dir(), "local", "checkpoints"))


def parse_data_arguments(parser):
    parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
    parser.add_argument("--train_num_ng", type=int, default=6, help="sample negative items for training")
    parser.add_argument("--valid_num_ng", type=int, default=99, help="sample part of negative items for valid")
    parser.add_argument("--test_num_ng", type=int, default=50, help="sample part of negative items for testing")
    parser.add_argument("--top_k", type=int, default=5, help="compute metrics@top_k")
    parser.add_argument("--dataset_version", type=int, default=1)
    parser.add_argument("--dataset_dir", type=str, default=os.path.join(get_root_dir(), "local", "input", "data"))
    parser.add_argument("--dataset_name", type=str, default="watch_log")
    parser.add_argument("--output_dir", type=str, default=os.path.join(get_root_dir(), "local", "output"))


def parse_args():
    parser = argparse.ArgumentParser()
    
    parse_common_arguments(parser)
    parse_data_arguments(parser)
    parse_train_arguments(parser)
    parse_model_arguments(parser)

    args = parser.parse_args()
    return args
