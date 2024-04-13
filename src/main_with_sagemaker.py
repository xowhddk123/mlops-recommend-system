import sys
import logging
import datetime


from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

from config.args import parse_args
from config.meta import Tasks
from utils.utils import make_s3_dataset_path, json_to_str
from config.meta import SageMakerMeta

from sagemaker.pytorch import PyTorch

import boto3
from botocore.exceptions import ClientError
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial

from sagemaker.debugger import ProfilerRule, rule_configs, ProfilerConfig


def get_profiler():
    return ProfilerConfig(system_monitor_interval_millis=500)


def get_rules():
    return [
        ProfilerRule.sagemaker(rule_configs.ProfilerReport())
    ]


def get_metric_definitions():
    return [
        {'Name': 'NDCG', 'Regex': "NDCG: (.*?);"},
        {'Name': 'train:loss', 'Regex': "Train loss: (.*?);"},
        {'Name': 'valid:loss', 'Regex': "Valid loss: (.*?);"},
    ]

def init_experiments(experiment_name, trial_name):
    sagemaker_boto3_client = boto3.client("sagemaker")
    try:
        Experiment.create(
            experiment_name=experiment_name,
            sagemaker_boto_client=sagemaker_boto3_client
        )
    except ClientError as e:
        logging.info(e)

    try:
        Trial.create(
            trial_name=trial_name,
            experiment_name=experiment_name,
            sagemaker_boto_client=sagemaker_boto3_client
        )
    except ClientError as e:
        logging.info(e)


def run_prepare_train_data_task(args, sagemaker_meta):
    output_dst = make_s3_dataset_path(
        base_dir=sagemaker_meta.s3_input_dir,
        dataset_name=f"prepared_{args.dataset_name}",
        dataset_version=args.dataset_version,
        base_date=sagemaker_meta.base_datetime
    ).replace('\\', '/')

    logging.info(f"input_src : {sagemaker_meta.s3_input_src}")
    logging.info(f"output_src : {sagemaker_meta.train_dataset_dir}")
    logging.info(f"output_dst : {output_dst}")
    pytorch_processor = PyTorchProcessor(
        framework_version=args.framework_version,
        py_version=args.py_version,
        code_location=sagemaker_meta.s3_output_dst,
        role=sagemaker_meta.sagemaker_role,
        instance_type=args.instance_type,
        instance_count=1,
        max_runtime_in_seconds=1 * 60 * 60,
    )

    pytorch_processor.run(
        code="main.py",
        source_dir=".",
        arguments=sys.argv[1:] + [
            "--dataset_dir", args.dataset_dir, 
            "--job_name", args.job_name
        ],
        inputs=[
            ProcessingInput(
                source=f"{sagemaker_meta.s3_input_src}/{args.dataset_name}.csv",
                destination=args.dataset_dir
            )
        ],
        outputs=[
            ProcessingOutput(
                source=sagemaker_meta.train_dataset_dir,
                destination=output_dst,
            )
        ],
        job_name=args.job_name,
        wait=True,
        experiment_config=sagemaker_meta.experiment_config, # wait=True 부분 찾아서 추가
    )


def run_prepare_inference_data_task(args, sagemaker_meta):
    logging.info(f"input_src : {sagemaker_meta.s3_input_src}")
    logging.info(f"input_dst : {sagemaker_meta.train_dataset_dir}")
    logging.info(f"output_src : {sagemaker_meta.inference_dataset_dir}")
    logging.info(f"output_dst : {sagemaker_meta.s3_input_src}")

    pytorch_processor = PyTorchProcessor(
        framework_version=args.framework_version,
        py_version=args.py_version,
        code_location=sagemaker_meta.s3_output_dst,
        role=sagemaker_meta.sagemaker_role,
        instance_type=args.instance_type,
        instance_count=1,
        max_runtime_in_seconds=1 * 60 * 60, # 최대 구동 시간
    )
    
    pytorch_processor.run(
        code="main.py",
        source_dir=".",
        arguments=sys.argv[1:] + [
            "--dataset_dir", args.dataset_dir, 
            "--job_name", args.job_name
        ],
        inputs=[
            ProcessingInput(
                source=(
                    f"{sagemaker_meta.s3_input_src}/"
                    f"{args.dataset_name}_train_{args.model_name}.snappy.parquet"
                ),
                destination=sagemaker_meta.train_dataset_dir
            )
        ],
        outputs=[
            ProcessingOutput(
                source=sagemaker_meta.inference_dataset_dir,
                destination=sagemaker_meta.s3_input_src,
            )
        ],
        job_name=args.job_name,
        wait=True,
        experiment_config=sagemaker_meta.experiment_config,
    )


def run_train_task(args, sagemaker_meta):
    checkpoint_s3_uri = (
        None if sagemaker_meta.is_local_mode 
        else f"{sagemaker_meta.s3_output_dst}/checkpoints"
    )
    checkpoint_local_path = (
	      None if sagemaker_meta.is_local_mode 
	      else args.checkpoint_path
	  )
    use_spot_instances = (
        False if sagemaker_meta.is_local_mode 
        else args.use_spot
    )
    
    profiler_config = None if sagemaker_meta.is_local_mode else get_profiler()
    rules = None if sagemaker_meta.is_local_mode else get_rules()
    
    hyperparameters = {key: json_to_str(val) for key, val in vars(args).items()}
    metric_definitions = get_metric_definitions()

    logging.info(f"input_src : {sagemaker_meta.s3_input_src}")
    logging.info(f"output_dst : {sagemaker_meta.s3_output_dst}")

    estimator = PyTorch(
        sagemaker_session=sagemaker_meta.sagemaker_session,
        entry_point="main.py",
        source_dir=".",
        output_path=sagemaker_meta.s3_output_dst,
        code_location=sagemaker_meta.s3_output_dst,
        role=sagemaker_meta.sagemaker_role,
        framework_version=args.framework_version,
        py_version=args.py_version,
        instance_type=args.instance_type,
        instance_count=1,
        hyperparameters=hyperparameters,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        max_run=1 * 60 * 60,
        use_spot_instances=use_spot_instances,
        max_wait=3 * 60 * 60,
        max_retry_attempts=3,
        metric_definitions=metric_definitions,
        # rule output을 만드는 코드
        disable_profiler=False,
        profiler_config=profiler_config,
        rules=rules,
    )

    estimator.fit(
        inputs={
            "train": (
                f"{sagemaker_meta.s3_input_src}/"
                f"{args.dataset_name}_train_{args.model_name}.snappy.parquet"
            )
        },
        job_name=args.job_name,
        wait=True,
        experiment_config=sagemaker_meta.experiment_config,
        
    )

def run_inference_task(args, sagemaker_meta):
    logging.info(f"input_src : {sagemaker_meta.s3_input_src}")
    logging.info(f"input_dst : {sagemaker_meta.inference_dataset_dir}")
    logging.info(f"output_src : {sagemaker_meta.inference_output_dir}")
    logging.info(f"output_dst : {sagemaker_meta.s3_output_dst}")

    if args.dependency_job_name is None:
        raise AttributeError("Dependency job name is required!")

    pytorch_processor = PyTorchProcessor(
        framework_version=args.framework_version,
        py_version=args.py_version,
        code_location=sagemaker_meta.s3_output_dst,
        role=sagemaker_meta.sagemaker_role,
        instance_type=args.instance_type,
        instance_count=1,
    )

    pytorch_processor.run(
        code="main.py",
        source_dir=".",
        arguments=sys.argv[1:] + [
            "--dataset_dir", args.dataset_dir,
            "--output_dir", args.output_dir,
            "--model_dir", args.model_dir,
            "--job_name", args.job_name,
        ],
        inputs=[
            ProcessingInput(
                source=(
                    f"{sagemaker_meta.s3_output_dst}/"
                    f"{args.dependency_job_name}/output/model.tar.gz"
                ),
                destination=f"{args.model_dir}"
            ),
            ProcessingInput(
                source=(
                    f"{sagemaker_meta.s3_output_dst}/"
                    f"{args.dependency_job_name}/output/output.tar.gz"
                ),
                destination=f"{args.output_dir}/data/index"
            ),
            ProcessingInput(
                source=(
                    f"{sagemaker_meta.s3_input_src}/"
                    f"{args.dataset_name}_inference_{args.model_name}.snappy.parquet"
                ),
                destination=sagemaker_meta.inference_dataset_dir
            ),
        ],
        outputs=[
            ProcessingOutput(
                source=f"{sagemaker_meta.inference_output_dir}/",
                destination=sagemaker_meta.s3_output_dst,
            )
        ],
        job_name=args.job_name,
        wait=True,
        experiment_config=sagemaker_meta.experiment_config,
    )


def get_default_local_dir(args):
    if args.task == "train":
        return "/opt/ml"
    else:
        return "/opt/ml/processing"

if __name__ == '__main__':
    args = parse_args()
    str_datetime = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    if args.job_name == "NoAssigned":
        args.job_name = f"{args.namespace}-{args.model_name}-{args.task}-{str_datetime}"
    default_local_dir = get_default_local_dir(args)
    args.dataset_dir = f"{default_local_dir}/input/data"
    args.output_dir = f"{default_local_dir}/output"
    args.model_dir = f"{default_local_dir}/model"
    args.checkpoint_path = "/opt/ml/checkpoints"

    sagemaker_meta = SageMakerMeta(args)

    init_experiments(
        experiment_name=sagemaker_meta.experiment_name,
        trial_name=sagemaker_meta.trial_name
    )

    task_map = {
        Tasks.PREPARE_TRAIN_DATA: run_prepare_train_data_task,
        Tasks.PREPARE_INFERENCE_DATA: run_prepare_inference_data_task,
        Tasks.TRAIN: run_train_task,
        Tasks.INFERENCE: run_inference_task,        
    }


    task = task_map.get(args.task)

    if not task:
        raise KeyError(f"작업을 찾을 수 없습니다 : {args.task}")

    task(args, sagemaker_meta)
