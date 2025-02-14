import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import interplote_pose, interplote_qpos, np_get_delta_qpos
from task.eval_func.base import BaseEval


class TableTopArmEval(BaseEval):
    def _simulate_under_extforce_details(self):
        raise NotImplementedError
