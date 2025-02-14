import os
import sys
import pdb

import numpy as np
import mujoco

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import np_get_delta_qpos
from task.eval_func.base import BaseEval


class FCMocapEval(BaseEval):
    def _simulate_under_extforce_details(self, pre_obj_qpos):
        external_force_direction = np.array(
            [
                [-1.0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, -1, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )

        for i in range(len(external_force_direction)):
            # 1. Reset to pre-grasp pose
            mujoco.mj_resetDataKeyframe(
                self.mj_model, self.mj_data, self.mj_model.nkey - 2
            )
            self.mj_data.qfrc_applied[:] = 0.0
            self.mj_data.xfrc_applied[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)

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

            # 4. Add external force on the object
            self.mj_data.xfrc_applied[-1] = (
                10 * external_force_direction[i] * self.configs.task.obj_mass
            )

            # 5. Wait for 2 seconds
            for j in range(10):
                for _ in range(50):
                    mujoco.mj_step(self.mj_model, self.mj_data)

                if self.configs.debug_viewer:
                    self.debug_viewer.sync()
                    pdb.set_trace()

                # Early stop
                _, _, latter_obj_qpos = self.hospec.split_qpos_pose(self.mj_data.qpos)
                delta_pos, delta_angle = np_get_delta_qpos(
                    pre_obj_qpos, latter_obj_qpos
                )
                succ_flag = (
                    delta_pos < self.configs.task.simulation_metrics.trans_thre
                ) & (delta_angle < self.configs.task.simulation_metrics.angle_thre)
                if not succ_flag:
                    break
            if not succ_flag:
                break

        return
