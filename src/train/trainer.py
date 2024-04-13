import os
import time
import pprint
import logging
import datetime

import torch

from model.functions import load_checkpoint, save_checkpoint, set_seed, load_loss_function
from model.scheduler import load_scheduler
from model.optimizer import load_optimizer
from train.train import run as run_train
from train.valid import run as run_validation
from train.evaluate import evaluate
from utils.utils import init_dirs


def initialize_executor(args, model):
    logging.debug(f"Number of gpus available - {args.num_gpus}")
    logging.debug(f"Model Save Dir - {args.model_dir}")

    args.save_path = args.model_dir
    init_dirs(args.save_path)

    set_seed(args)

    if args.use_cuda:
        model.cuda()

    return args


def load_datasets(dataset_generator):
    logging.info("Load datasets")
    train_dataset = dataset_generator.get_train_dataset()
    valid_dataset = dataset_generator.get_valid_dataset()
    valid_metric_dataset = dataset_generator.get_valid_metric_dataset()
    return train_dataset, valid_dataset, valid_metric_dataset


def load_dataloaders(args, dataset_generator, train_dataset, valid_dataset, valid_metric_dataset):
    logging.info("Load dataloaders")
    train_dataloader = dataset_generator.get_train_loader(train_dataset, args.batch_size)
    valid_dataloader = dataset_generator.get_valid_loader(valid_dataset, args.batch_size)
    valid_metric_dataloader = dataset_generator.get_valid_metric_loader(valid_metric_dataset, args.valid_num_ng)
    return train_dataloader, valid_dataloader, valid_metric_dataloader


def run(
        args,
        model,
        criterion,
        optimizer,
        scheduler,
        resume_from_epoch,
        train_loader,
        valid_loader,
        valid_metric_loader,
):
    base_metric_name = "VAR"
    best_metric_value, best_epoch = 0, 0

    logging.info(model)
    for epoch in range(resume_from_epoch, args.epochs):
        logging.info(f"Epoch: {epoch + 1}")

        epoch_start_time = time.time()
        start_time = time.time()

        train_loss = run_train(args.use_cuda, model, criterion, optimizer, train_loader)

        train_duration = str(datetime.timedelta(seconds=time.time() - start_time))
        logging.info(f"Train loss: {train_loss:.4f};")
        logging.info(f"End Train | Time: {train_duration}")

        start_time = time.time()

        valid_loss = run_validation(args.use_cuda, model, criterion, valid_loader)

        valid_duration = str(datetime.timedelta(seconds=time.time() - start_time))
        logging.info(f"Valid loss: {valid_loss:.4f};")
        logging.info(f"End Valid | Time: {valid_duration}")

        start_time = time.time()

        metric_result = evaluate(args.use_cuda, model, valid_metric_loader, args.top_k)

        metric_eval_duration = str(datetime.timedelta(seconds=time.time() - start_time))
        logging.info(f"End Metric | Time: {metric_eval_duration}")

        for key, val in metric_result.items():
            logging.info(f"{key.upper()}: {val:.3f};")

        base_metric_value = metric_result.get(base_metric_name.upper(), 0)
        if base_metric_value > best_metric_value:
            best_metric_value = base_metric_value
            torch.save(
                model, os.path.join(args.save_path, f"{args.model_name}_best_{base_metric_name.upper()}.pth")
            )
            logging.info(f"Saved best {base_metric_name.upper()} model")
            save_checkpoint(model, optimizer, epoch, train_loss, valid_loss, args)

        scheduler.step()

        epoch_duration = str(datetime.timedelta(seconds=time.time() - epoch_start_time))
        logging.info(f"Epoch: {epoch + 1} | Time: {epoch_duration}")


def start(args, model, dataset_generator):
    logging.info(f"Arguments : {pprint.pformat(vars(args))}")
    args = initialize_executor(args, model)

    train_dataset, valid_dataset, valid_metric_dataset = load_datasets(dataset_generator)

    train_loader, valid_loader, valid_metric_loader = load_dataloaders(
        args,
        dataset_generator,
        train_dataset,
        valid_dataset,
        valid_metric_dataset,
    )

    criterion = load_loss_function(args)
    optimizer = load_optimizer(args, model)

    if not os.path.isfile(os.path.join(args.checkpoint_path, "checkpoint.pth")):
        resume_from_epoch = 0
    else:
        model, optimizer, resume_from_epoch = load_checkpoint(
            model, optimizer, args.checkpoint_path
        )

    scheduler = load_scheduler(args, optimizer)

    run(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        resume_from_epoch=resume_from_epoch,
        train_loader=train_loader,
        valid_loader=valid_loader,
        valid_metric_loader=valid_metric_loader,
    )
