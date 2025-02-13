import os
import sys
import pdb
from copy import deepcopy

import numpy as np
import mujoco
import mujoco.viewer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import interplote_pose, interplote_qpos, np_get_delta_qpos
from task.eval_sim.basic import BasicEval


class TableTopMocapEval(BasicEval):
    def _eval_external_force_details(self, pre_obj_qpos):
        # 1. Set object gravity
        external_force_direction = np.array([0.0, 0, -1, 0, 0, 0])
        self.mj_data.qfrc_applied[:] = 0.0
        self.mj_data.xfrc_applied[-1] = (
            10 * external_force_direction * self.configs.task.obj_mass
        )

        # 2. Move hand to grasp pose
        self._control_hand_with_interp(
            self.grasp_data["pregrasp_pose"],
            self.grasp_data["grasp_pose"],
            self.grasp_data["pregrasp_ctrl"],
            self.grasp_data["grasp_ctrl"],
        )

        # 3. Move hand to squeeze pose.
        # NOTE step 2 and 3 are seperate because pre -> grasp -> squeeze are stage-wise linear.
        # If step 2 and 3 are merged to one linear interpolation, the performance will drop a lot.
        self._control_hand_with_interp(
            self.grasp_data["grasp_pose"],
            self.grasp_data["squeeze_pose"],
            self.grasp_data["grasp_ctrl"],
            self.grasp_data["squeeze_ctrl"],
        )

        # 5. Lift the object
        lift_pose = deepcopy(self.grasp_data["squeeze_pose"])
        lift_pose[2] += 0.1
        self._control_hand_with_interp(
            self.grasp_data["squeeze_pose"],
            lift_pose,
            self.grasp_data["squeeze_ctrl"],
            self.grasp_data["squeeze_ctrl"],
        )

        return
