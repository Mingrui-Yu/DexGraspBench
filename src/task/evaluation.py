import os
import multiprocessing
import logging
import pdb
import glob
import traceback
from copy import deepcopy

import numpy as np
import mujoco
import mujoco.viewer

from util.rot_utils import np_get_delta_qpos
from util.mujoco_utils import build_spec_for_test
from util.file_utils import load_json


class EvalOne:
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
        self.hospec = build_spec_for_test(
            obj_path=self.grasp_data["obj_path"],
            obj_pose=self.grasp_data["obj_pose"],
            obj_scale=self.grasp_data["obj_scale"],
            has_floor_z0=False,
            obj_density=new_obj_density,
            hand_xml_path=configs.hand.xml_path,
            hand_pose=self.grasp_data["hand_pose"],
            hand_qpos=self.grasp_data["hand_qpos"],
            hand_tendon=configs.hand.tendon,
            friction_coef=configs.task.miu_coef,
        )

        # Get ready for simulation
        self.mj_model = self.hospec.spec.compile()
        self.mj_data = mujoco.MjData(self.mj_model)

        if self.configs.debug:
            with open("debug.xml", "w") as f:
                f.write(self.hospec.spec.to_xml())

        # Initialize hand pose by setting keyframe
        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.nkey - 1)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return

    def _penetration_and_contact(self):
        for i in range(self.mj_model.ngeom):
            if "object_collision" in self.mj_model.geom(i).name:
                self.mj_model.geom_margin[i] = self.mj_model.geom_gap[i] = (
                    self.configs.task.contact_margin
                )

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
                body_name = body1_name if body2_id == object_id else body2_name
                for name in contact_dist_dict:
                    if body_name.startswith(name):
                        ho_pene = min(ho_pene, contact.dist)
                        contact_dist_dict[name] = min(
                            contact_dist_dict[name], contact.dist
                        )
                if np.abs(contact.dist) < self.configs.task.contact_threshold:
                    contact_link_set.add(body_name)
            elif body1_id < hand_id and body2_id < hand_id:
                self_pene = min(self_pene, contact.dist)
        self_pene, ho_pene = -self_pene, -ho_pene
        contact_dist_lst = list(contact_dist_dict.values())
        contact_distance = np.mean(contact_dist_lst)
        contact_consistency = np.max(contact_dist_lst) - np.min(contact_dist_lst)
        self.mj_model.geom_margin = 0
        self.mj_model.geom_gap = 0
        return ho_pene, self_pene, contact_distance, contact_consistency

    def _resist_obj_gravity(self):
        if self.configs.debug_viewer:
            viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            viewer.sync()
            pdb.set_trace()

        _, _, pre_obj_pose = self.hospec.split_qpos_pose(self.mj_data.qpos)
        pre_obj_qpos = deepcopy(pre_obj_pose)

        grasp_hand_qpos = self.grasp_data["hand_qpos"]
        if (
            self.configs.task.pregrasp_type == "real"
            and "pregrasp_qpos" in self.grasp_data
        ):
            pregrasp_hand_qpos = self.grasp_data["pregrasp_qpos"]
        elif self.configs.task.pregrasp_type == "minus":
            pregrasp_hand_qpos = (
                deepcopy(grasp_hand_qpos) - self.configs.task.pregrasp_minus
            )
        elif self.configs.task.pregrasp_type == "multiply":
            pregrasp_hand_qpos = grasp_hand_qpos * np.where(
                grasp_hand_qpos > 0,
                self.configs.task.pregrasp_multiply,
                2 - self.configs.task.pregrasp_multiply,
            )

        if (
            self.configs.task.squeeze_type == "real"
            and "squeeze_qpos" in self.grasp_data
        ):
            squeeze_hand_qpos = self.grasp_data["squeeze_qpos"]
            pregrasp_hand_qpos = self.grasp_data["hand_qpos"]
        else:
            squeeze_hand_qpos = (
                grasp_hand_qpos - pregrasp_hand_qpos
            ) * self.configs.task.squeeze_ratio + grasp_hand_qpos

        dense_actuator_moment = np.zeros((self.mj_model.nu, self.mj_model.nv))
        mujoco.mju_sparse2dense(
            dense_actuator_moment,
            self.mj_data.actuator_moment,
            self.mj_data.moment_rownnz,
            self.mj_data.moment_rowadr,
            self.mj_data.moment_colind,
        )
        squeeze_hand_ctrl = dense_actuator_moment[:, 6:-6] @ squeeze_hand_qpos
        grasp_hand_ctrl = dense_actuator_moment[:, 6:-6] @ grasp_hand_qpos

        if self.configs.task.force_closure:
            external_force_direction = np.array(
                [
                    [1.0, 0, 0, 0, 0, 0],
                    [-1.0, 0, 0, 0, 0, 0],
                    [0.0, 1, 0, 0, 0, 0],
                    [0.0, -1, 0, 0, 0, 0],
                    [0.0, 0, 1, 0, 0, 0],
                    [0.0, 0, -1, 0, 0, 0],
                ]
            )
        else:
            external_force_direction = [self.grasp_data["obj_gravity_direction"]]

        for i in range(len(external_force_direction)):
            mujoco.mj_resetDataKeyframe(
                self.mj_model, self.mj_data, self.mj_model.nkey - 1
            )
            self.mj_data.qfrc_applied[:] = 0.0
            self.mj_data.xfrc_applied[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
            for j in range(10):
                self.mj_data.ctrl[:] = (j + 1) / 10 * (
                    squeeze_hand_ctrl - grasp_hand_ctrl
                ) + grasp_hand_ctrl
                mujoco.mj_forward(self.mj_model, self.mj_data)
                for _ in range(10):
                    mujoco.mj_step(self.mj_model, self.mj_data)
            self.mj_data.xfrc_applied[-1] = (
                10 * external_force_direction[i] * self.configs.task.obj_mass
            )
            for j in range(10):
                for _ in range(50):
                    mujoco.mj_step(self.mj_model, self.mj_data)

                if self.configs.debug_viewer:
                    viewer.sync()
                    pdb.set_trace()

                _, _, latter_obj_qpos = self.hospec.split_qpos_pose(self.mj_data.qpos)
                delta_pos, delta_angle = np_get_delta_qpos(
                    pre_obj_qpos, latter_obj_qpos
                )
                succ_flag = (delta_pos < 0.05) & (delta_angle < 15)
                if not succ_flag:
                    break
            if not succ_flag:
                break

        if self.configs.debug or self.configs.debug_viewer:
            print(succ_flag, delta_pos, delta_angle)

        return succ_flag, delta_pos, delta_angle

    def _force_closure_metric(self):
        return

    def run(self):
        ho_pene, self_pene, cont_dist, cont_consis = self._penetration_and_contact()
        succ_flag, delta_pos, delta_angle = self._resist_obj_gravity()
        succ_npy_path = self.input_npy_path.replace(
            self.configs.grasp_dir, self.configs.succ_dir
        )
        if succ_flag and not os.path.exists(succ_npy_path):
            os.makedirs(os.path.dirname(succ_npy_path), exist_ok=True)
            os.system(
                f"ln -s {os.path.relpath(self.input_npy_path, os.path.dirname(succ_npy_path))} {succ_npy_path}"
            )
        eval_npy_path = self.input_npy_path.replace(
            self.configs.grasp_dir, self.configs.eval_dir
        )
        if not os.path.exists(eval_npy_path):
            os.makedirs(os.path.dirname(eval_npy_path), exist_ok=True)
            np.save(
                eval_npy_path,
                {
                    "ho_pene": ho_pene,
                    "self_pene": self_pene,
                    "cont_dist": cont_dist,
                    "cont_consis": cont_consis,
                    "succ_flag": succ_flag,
                },
            )
        return


def safe_eval_one(params):
    input_npy_path, configs = params[0], params[1]
    try:
        return EvalOne(input_npy_path, configs).run()
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return input_npy_path


def task_eval(configs):
    input_path_lst = glob.glob(
        os.path.join(configs.grasp_dir, *list(configs.data_struct))
    )
    if configs.skip:
        eval_path_lst = glob.glob(
            os.path.join(configs.eval_dir, *list(configs.data_struct))
        )
        eval_path_lst = [
            p.replace(configs.eval_dir, configs.grasp_dir) for p in eval_path_lst
        ]
        input_path_lst = list(set(input_path_lst).difference(set(eval_path_lst)))

    if configs.task.mini_set > 0:
        input_path_lst = input_path_lst[: configs.task.mini_set]
    logging.info(f"Find {len(input_path_lst)} graspdata")

    if len(input_path_lst) == 0:
        return

    iterable_params = zip(input_path_lst, [configs] * len(input_path_lst))
    if configs.debug or configs.debug_viewer:
        for ip in iterable_params:
            safe_eval_one(ip)
    else:
        with multiprocessing.Pool(processes=configs.n_worker) as pool:
            result_iter = pool.imap_unordered(safe_eval_one, iterable_params)
            results = list(result_iter)

    logging.info(f"Finish")

    return
