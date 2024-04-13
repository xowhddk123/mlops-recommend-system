import pprint
import logging
import warnings

from config.args import parse_args
from config.meta import Services
from utils.utils import init_dirs


def get_data_generator(args, dataset_generator):
    data_generator = dataset_generator(args)
    data_generator.download()
    data_generator.preprocess()
    return data_generator


def load_model(args, model, dataset_generator):
    return model(
        dataset_generator.user_num,
        dataset_generator.item_num,
        args.factor_num,
        args.num_layers,
        args.dropout,
    )


def run_prepare_train_data_task(args, tasks):
    processor = tasks.get_process(args.task)
    preprocessor = processor(args)
    preprocessor.run()


def run_train_task(args, tasks):
    processor = tasks.get_process(args.task)

    init_dirs(args.model_dir, args.checkpoint_path)

    logging.info("Generating data...")
    dataset_generator = get_data_generator(args, tasks.DATA_GENERATOR)

    logging.info(f"Load {args.model_name.upper()} model...")
    model = load_model(args, tasks.MODEL, dataset_generator)

    logging.info("Start Training Process...")
    processor.start(args, model, dataset_generator)


def run_prepare_inference_data_task(args, tasks):
    processor = tasks.get_process(args.task)

    pre_inference_data = processor(args)
    pre_inference_data.run()


def run_inference_task(args, tasks):
    processor = tasks.get_process(args.task)

    inference = processor(args)
    inference.run()


def main(args):
    args.use_cuda = args.num_gpus > 0
    service_name = f"{args.serve_recommend_type}-{args.serve_contents_type}"
    tasks = Services.get_tasks(service_name)
    func_map = {
        tasks.PREPARE_TRAIN_DATA: run_prepare_train_data_task,
        tasks.TRAIN: run_train_task,
        tasks.PREPARE_INFERENCE_DATA: run_prepare_inference_data_task,
        tasks.INFERENCE: run_inference_task,
    }

    func = func_map.get(args.task)

    if not func:
        raise KeyError(f"Not found target function : {args.task}")

    logging.info(f"[TASK-START] {args.task}")
    func(args, tasks)
    logging.info(f"[TASK-END] {args.task}")


if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger()
    logger.setLevel(args.log_level)
    warnings.filterwarnings("ignore")
    logging.debug(f"ncf_run args : {pprint.pformat(vars(args))}")
    main(args)
