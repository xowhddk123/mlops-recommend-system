import sys
import json
sys.path.append("./src")

from config.meta import Services, LikeMovieTasks
from utils.utils import json_to_str


def test_sample():
    assert True


def test_json_to_str():
    case_1 = json_to_str('{"a": 1, "b": "2"}')
    case_2 = json_to_str('plain text')
    case_3 = json_to_str(12345)

    assert case_1 == json.dumps('{"a": 1, "b": "2"}')
    assert case_2 == "plain text"
    assert case_3 == 12345


def test_service_meta():
    services = Services()
    assert 'like-movie' in services.TASK_MAP.keys()


def test_task_meta():
    tasks = LikeMovieTasks()
    assert tasks.MODEL.__name__ == "NCF"
