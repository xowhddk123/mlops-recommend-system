import json
import logging

from torch import optim


def load_scheduler(args, optimizer):
    logging.info(f"Load scheduler : {args.scheduler_name}")
    logging.info(f"Scheduler params : {repr(args.scheduler_params)}")

    return getattr(optim.lr_scheduler, args.scheduler_name)(optimizer, **json.loads(args.scheduler_params))
