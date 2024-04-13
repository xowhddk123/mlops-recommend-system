import os
import json
import random
import logging

import numpy as np
from torch import nn, cuda, manual_seed, save, load
from torch.backends import cudnn

from utils.utils import init_dirs


def save_checkpoint(model, optimizer, epoch, train_loss, valid_loss, args):
    init_dirs(args.checkpoint_path)

    logging.info(f"Saving the Checkpoint: {args.checkpoint_path}")
    logging.debug(f"epoch: {epoch + 1}")
    logging.debug(f"train_loss: {train_loss}")
    logging.debug(f"valid_loss: {valid_loss}")

    save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        },
        os.path.join(args.checkpoint_path, f"{args.model_name}_e{epoch+1}.pth"),
    )


def load_checkpoint(model, optimizer, checkpoint_path):
    logging.debug("Checkpoint file found!")
    checkpoint_path = checkpoint_path + "/checkpoint.pth"
    logging.debug(f"Rank({checkpoint_path}) Loading Checkpoint")
    checkpoint = load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_number = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    valid_loss = checkpoint["valid_loss"]
    logging.debug(
        f"Checkpoint File Loaded - epoch_number: {epoch_number} "
        f"- train_loss: {train_loss} "
        f"- valid_loss: {valid_loss}"
    )
    logging.debug(f"Resuming training from epoch: {epoch_number+1}")
    return model, optimizer, epoch_number


def set_seed(args):
    logging.info(f"Set seed : {args.seed}")
    manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.use_cuda:
        cudnn.benchmark = True
        cuda.manual_seed(args.seed)


def load_loss_function(args):
    logging.info(f"Load loss function function : {args.loss_function_name}")
    logging.info(f"Loss function params : {args.loss_function_params}")
    return getattr(nn, args.loss_function_name)(**json.loads(args.loss_function_params))
