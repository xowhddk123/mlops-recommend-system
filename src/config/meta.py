from datetime import datetime

from sagemaker.session import Session
from sagemaker.local.local_session import LocalSession

from utils.utils import make_s3_dataset_path, make_s3_model_output_path

class ArgumentsMeta:
    TYPE = str
    NAME = "default"

    def __init__(self):
        super().__init__()

    @classmethod
    def contains(cls, item):
        return item in cls.values()

    @classmethod
    def values(cls):
        return [
            cls.__dict__[key]
            for key in cls.__dict__.keys()
            if key[:1] != "_" and isinstance(cls.__dict__[key], cls.TYPE)
        ]

    @staticmethod
    def help():
        return ""


class Tasks(ArgumentsMeta):
    PREPARE_TRAIN_DATA = "prepare-train-data"
    PREPARE_DATA = "prepare-data"
    PREPARE_INFERENCE_DATA = "prepare-inference-data"
    TRAIN = "train"
    INFERENCE = "inference"

    MODEL = None
    DATA_GENERATOR = None


class LikeMovieTasks(Tasks):
    from train import trainer
    from model.ncf import NCF, NCFDataGenerator
    from preprocess.preprocess import WatchLogNCFPreprocessor
    from inference.preinference_data import PreInferenceNCFData
    from inference.inference_model import InferenceNCF

    PROCESS_MAP = {
        Tasks.PREPARE_TRAIN_DATA: WatchLogNCFPreprocessor,
        Tasks.PREPARE_INFERENCE_DATA: PreInferenceNCFData,
        Tasks.TRAIN: trainer,
        Tasks.INFERENCE: InferenceNCF,
    }

    MODEL = NCF
    DATA_GENERATOR = NCFDataGenerator

    @classmethod
    def get_process(cls, task_name):
        if task_name not in cls.PROCESS_MAP.keys():
            raise KeyError(f"Not found mapped process : {task_name}")

        return cls.PROCESS_MAP[task_name]


class Services:
    TASK_MAP = {
        "like-movie": LikeMovieTasks()
    }

    @classmethod
    def get_tasks(cls, service_name):
        if service_name not in cls.TASK_MAP.keys():
            raise KeyError(f"Not found mapped task : {service_name}")

        return cls.TASK_MAP[service_name]


class SageMakerMeta:
    def __init__(self, args):
        self.is_local_mode = "local" in args.instance_type.lower()
        self.sagemaker_role = \
            "arn:aws:iam::602570579971:role/MLOpsSageMakerExecutionRole"  # <ACCOUNT> 수정
        self.sagemaker_session = LocalSession() if self.is_local_mode else Session()
        self.base_datetime = datetime.strptime(args.base_date, "%Y-%m-%d")
        self.str_datetime = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        self.train_dataset_dir = f"{args.dataset_dir}/train"
        self.inference_dataset_dir = f"{args.dataset_dir}/inference"
        self.inference_output_dir = f"{args.output_dir}/inference"
        self.s3_base_path = \
            f"s3://mlops-recommend-system-user03/ns={args.namespace}"  # <유저명> 수정
        self.s3_input_dir = f"{self.s3_base_path}/input/data"
        self.s3_input_src = make_s3_dataset_path(
            base_dir=self.s3_input_dir,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            base_date=self.base_datetime
        ).replace('\\', '/')
        self.s3_output_dir = f"{self.s3_base_path}/output"
        self.s3_output_dst = make_s3_model_output_path(
            base_dir=self.s3_output_dir,
            model_name=args.model_name,
            base_date=self.base_datetime
        ).replace('\\', '/')
        
        self.experiment_name = f"{args.namespace}-{args.model_name}"
        self.trial_name = f"{args.task}-{self.str_datetime}"
        self.experiment_config = None if self.is_local_mode else {
            "ExperimentName": self.experiment_name,
            "TrialName": self.trial_name,
        }
