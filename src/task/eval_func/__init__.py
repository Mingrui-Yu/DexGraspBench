import os

os.environ["MUJOCO_GL"] = "osmesa"

from .fc_mocap import FCMocapEval
from .tabletop_mocap import TableTopMocapEval
from .tabletop_arm import TableTopArmEval
