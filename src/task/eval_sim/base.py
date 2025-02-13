import os
import sys
import pdb
from copy import deepcopy

import numpy as np
import imageio
import mujoco
import mujoco.viewer


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from util.rot_util import interplote_pose, interplote_qpos, np_get_delta_qpos
from util.hand_util import HOMjSpec
from util.file_util import load_json
from task.fc_metric import *


class BaseEval:
    def __init__(self, input_npy_path, configs):
        self.input_npy_path = input_npy_path
        self.configs = configs
        self.grasp_data = np.load(input_npy_path, allow_pickle=True).item()

        # Fix object mass by setting density
        obj_info = load_json(
            os.path.join(self.grasp_data["obj_path"], "info/simplified.json")
        )
        obj_coef = obj_info["mass"] / (obj_info["density"] * (obj_info["scale"] ** 3))
        new_obj_density = configs.task.obj_mass / (
            obj_coef * (self.grasp_data["obj_scale"] ** 3)
        )

        # Build mj_spec
        self.hospec = HOMjSpec(
            obj_path=self.grasp_data["obj_path"],
            obj_pose=self.grasp_data["obj_pose"],
            obj_scale=self.grasp_data["obj_scale"],
            has_floor_z0="TableTop" in configs.setting,
            obj_density=new_obj_density,
            hand_xml_path=configs.hand.xml_path,
            hand_tendon_cfg=configs.hand.tendon,
            friction_coef=configs.task.miu_coef,
            grasp_pose=self.grasp_data["grasp_pose"],
            grasp_qpos=self.grasp_data["grasp_qpos"],
            pregrasp_pose=self.grasp_data["pregrasp_pose"],
            pregrasp_qpos=self.grasp_data["pregrasp_qpos"],
        )

        # Get ready for simulation
        self.mj_model = self.hospec.spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        if self.configs.debug:
            with open("debug.xml", "w") as f:
                f.write(self.hospec.spec.to_xml())

        return

    def _eval_pene_and_contact(self):
        # Change contact margin
        for i in range(self.mj_model.ngeom):
            if "object_collision" in self.mj_model.geom(i).name:
                self.mj_model.geom_margin[i] = self.mj_model.geom_gap[i] = (
                    self.configs.task.contact_margin
                )

        # Set to grasp pose
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.nkey - 1)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self_pene = 0
        ho_pene = 0
        contact_dist_dict = {
            name: self.configs.task.contact_margin
            for name in self.configs.hand.finger_prefix
        }
        contact_link_set = set()
        for jj, contact in enumerate(self.mj_data.contact):
            body1_id = self.mj_model.geom(contact.geom1).bodyid
            body2_id = self.mj_model.geom(contact.geom2).bodyid
            body1_name = self.mj_model.body(
                self.mj_model.geom(contact.geom1).bodyid
            ).name
            body2_name = self.mj_model.body(
                self.mj_model.geom(contact.geom2).bodyid
            ).name
            total_body_num = self.mj_model.nbody
            object_id = total_body_num - 1
            hand_id = total_body_num - 2
            if (body1_id < hand_id and body2_id == object_id) or (
                body2_id < hand_id and body1_id == object_id
            ):
                ho_pene = min(ho_pene, contact.dist)
                body_name = (
                    body1_name.removeprefix(self.hospec.hand_prefix)
                    if body2_id == object_id
                    else body2_name.removeprefix(self.hospec.hand_prefix)
                )
                for finger_prefix in contact_dist_dict:
                    if body_name.startswith(finger_prefix):
                        contact_dist_dict[finger_prefix] = min(
                            contact_dist_dict[finger_prefix], contact.dist
                        )
                        break
                if (
                    np.abs(contact.dist) < self.configs.task.contact_threshold
                    and body_name in self.configs.hand.valid_body_name
                ):
                    contact_link_set.add(body_name)
            elif body1_id < hand_id and body2_id < hand_id:
                self_pene = min(self_pene, contact.dist)
        self_pene, ho_pene = -self_pene, -ho_pene
        contact_dist_lst = list(contact_dist_dict.values())
        contact_distance = np.mean(contact_dist_lst)
        contact_consistency = np.max(contact_dist_lst) - np.min(contact_dist_lst)
        contact_number = len(contact_link_set)
        self.mj_model.geom_margin = 0
        self.mj_model.geom_gap = 0
        return ho_pene, self_pene, contact_number, contact_distance, contact_consistency

    def _check_large_penetration(self):
        for jj, contact in enumerate(self.mj_data.contact):
            body1_id = self.mj_model.geom(contact.geom1).bodyid
            body2_id = self.mj_model.geom(contact.geom2).bodyid
            total_body_num = self.mj_model.nbody
            object_id = total_body_num - 1
            hand_id = total_body_num - 2
            if contact.dist < -self.configs.task.max_pene and (
                (body1_id < hand_id and body2_id == object_id)
                or (body2_id < hand_id and body1_id == object_id)
            ):
                return False
        return True

    def _control_hand_with_interp(
        self, pose1, pose2, ctrl1, ctrl2, step_outer=10, step_inner=10
    ):
        pose_interp = interplote_pose(pose1, pose2, step_outer)
        qpos_interp = interplote_qpos(ctrl1, ctrl2, step_outer)
        for j in range(step_outer):
            self.mj_data.mocap_pos[0] = pose_interp[j, :3]
            self.mj_data.mocap_quat[0] = pose_interp[j, 3:7]
            self.mj_data.ctrl[:] = qpos_interp[j]
            mujoco.mj_forward(self.mj_model, self.mj_data)
            for _ in range(step_inner):
                mujoco.mj_step(self.mj_model, self.mj_data)

            if self.configs.debug_render:
                self.debug_renderer.update_scene(
                    self.mj_data, "closeup", self.debug_options
                )
                pixels = self.debug_renderer.render()
                self.debug_images.append(pixels)
        return

    def _eval_external_force(self):
        if self.configs.debug_viewer:
            viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            viewer.sync()
            pdb.set_trace()

        if self.configs.debug_render:
            self.debug_renderer = mujoco.Renderer(self.mj_model, 480, 640)
            self.debug_options = mujoco.MjvOption()
            mujoco.mjv_defaultOption(self.debug_options)
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self.debug_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            self.debug_images = []

        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.nkey - 2)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        _, _, pre_obj_pose = self.hospec.split_qpos_pose(self.mj_data.qpos)
        pre_obj_qpos = deepcopy(pre_obj_pose)
        if "TableTop" in self.configs.setting:
            pre_obj_qpos[2] += 0.1

        dense_actu_moment = np.zeros((self.mj_model.nu, self.mj_model.nv))
        mujoco.mju_sparse2dense(
            dense_actu_moment,
            self.mj_data.actuator_moment,
            self.mj_data.moment_rownnz,
            self.mj_data.moment_rowadr,
            self.mj_data.moment_colind,
        )
        self.grasp_data["pregrasp_ctrl"] = (
            dense_actu_moment[:, 6:-6] @ self.grasp_data["pregrasp_qpos"]
        )
        self.grasp_data["grasp_ctrl"] = (
            dense_actu_moment[:, 6:-6] @ self.grasp_data["grasp_qpos"]
        )
        self.grasp_data["squeeze_ctrl"] = (
            dense_actu_moment[:, 6:-6] @ self.grasp_data["squeeze_qpos"]
        )

        self._eval_external_force_details(pre_obj_qpos)

        _, _, latter_obj_qpos = self.hospec.split_qpos_pose(self.mj_data.qpos)
        delta_pos, delta_angle = np_get_delta_qpos(pre_obj_qpos, latter_obj_qpos)
        succ_flag = (delta_pos < self.configs.task.trans_thre) & (
            delta_angle < self.configs.task.angle_thre
        )

        if self.configs.debug or self.configs.debug_viewer or self.configs.debug_render:
            print(succ_flag, delta_pos, delta_angle)

        if self.configs.debug_render:
            debug_path = self.input_npy_path.replace(
                self.configs.grasp_dir, self.configs.debug_dir
            ).replace(".npy", ".gif")
            os.makedirs(os.path.dirname(debug_path), exist_ok=True)
            imageio.mimsave(debug_path, self.debug_images)
            print("save to ", debug_path)

        return succ_flag, delta_pos, delta_angle

    def _eval_external_force_details(self, pre_obj_qpos):
        raise NotImplementedError

    def _eval_analytic_fc_metric(self):
        # Change contact margin
        for i in range(self.mj_model.ngeom):
            if "object_collision" in self.mj_model.geom(i).name:
                self.mj_model.geom_margin[i] = self.mj_model.geom_gap[i] = (
                    self.configs.task.contact_threshold
                )

        # Set to grasp pose
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.nkey - 1)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        contact_point_lst = []
        contact_normal_lst = []
        for jj, contact in enumerate(self.mj_data.contact):
            body1_id = self.mj_model.geom(contact.geom1).bodyid
            body2_id = self.mj_model.geom(contact.geom2).bodyid
            body1_name = self.mj_model.body(
                self.mj_model.geom(contact.geom1).bodyid
            ).name
            body2_name = self.mj_model.body(
                self.mj_model.geom(contact.geom2).bodyid
            ).name
            total_body_num = self.mj_model.nbody
            object_id = total_body_num - 1
            hand_id = total_body_num - 2
            if (body1_id < hand_id and body2_id == object_id) or (
                body2_id < hand_id and body1_id == object_id
            ):
                body_name = (
                    body1_name.removeprefix(self.hospec.hand_prefix)
                    if body2_id == object_id
                    else body2_name.removeprefix(self.hospec.hand_prefix)
                )
                if (
                    np.abs(contact.dist) < self.configs.task.contact_threshold
                    and body_name in self.configs.hand.valid_body_name
                ):
                    contact_point_lst.append(contact.pos)
                    contact_normal_lst.append(contact.frame[0:3])

        self.mj_model.geom_margin = 0
        self.mj_model.geom_gap = 0

        if len(contact_point_lst) == 0:
            return 1, 1, 1, 1

        contact_points = np.stack(contact_point_lst)
        contact_normals = np.stack(contact_normal_lst)
        qp_metric = calcu_qp_metric(
            contact_points, contact_normals, self.configs.task.miu_coef
        )
        qp_dfc_metric = calcu_qp_dfc_metric(
            contact_points, contact_normals, self.configs.task.miu_coef
        )
        q1_metric = calcu_q1_metric(
            contact_points, contact_normals, self.configs.task.miu_coef
        )
        dfc_metric = calcu_dfc_metric(contact_points, contact_normals)
        return qp_metric, qp_dfc_metric, dfc_metric, q1_metric

    def run(self):
        ho_pene, self_pene, contact_num, contact_dist, contact_consis = (
            self._eval_pene_and_contact()
        )
        succ_flag, delta_pos, delta_angle = self._eval_external_force()
        qp_metric, qp_dfc_metric, dfc_metric, q1_metric = (
            self._eval_analytic_fc_metric()
        )
        # Save success data
        succ_npy_path = self.input_npy_path.replace(
            self.configs.grasp_dir, self.configs.succ_dir
        )
        if succ_flag and not os.path.exists(succ_npy_path):
            os.makedirs(os.path.dirname(succ_npy_path), exist_ok=True)
            os.system(
                f"ln -s {os.path.relpath(self.input_npy_path, os.path.dirname(succ_npy_path))} {succ_npy_path}"
            )

        # Save evaluation information
        eval_npy_path = self.input_npy_path.replace(
            self.configs.grasp_dir, self.configs.eval_dir
        )
        if not os.path.exists(eval_npy_path):
            os.makedirs(os.path.dirname(eval_npy_path), exist_ok=True)
            eval_dict = {
                "ho_pene": ho_pene,
                "self_pene": self_pene,
                "contact_num": contact_num,
                "contact_dist": contact_dist,
                "contact_consis": contact_consis,
                "succ_flag": succ_flag,
                "delta_pos": delta_pos,
                "delta_angle": delta_angle,
                "qp_metric": qp_metric,
                "qp_dfc_metric": qp_dfc_metric,
                "q1_metric": q1_metric,
                "dfc_metric": dfc_metric,
            }
            for key in [
                "grasp_pose",
                "grasp_qpos",
                "pregrasp_pose",
                "pregrasp_qpos",
                "squeeze_pose",
                "squeeze_qpos",
                "obj_scale",
                "obj_path",
                "obj_pose",
            ]:
                if key in self.grasp_data.keys():
                    eval_dict[key] = self.grasp_data[key]
            np.save(eval_npy_path, eval_dict)
        return
