import json
import logging

from torch import optim


def load_optimizer(args, model):
    logging.info(f"Load optimizer : {args.optimizer_name}")
    logging.info(f"Optimizer params : {args.optimizer_params}")
    return getattr(optim, args.optimizer_name)(model.parameters(), **json.loads(args.optimizer_params))
