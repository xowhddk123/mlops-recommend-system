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
