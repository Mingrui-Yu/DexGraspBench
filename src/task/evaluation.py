import os
import multiprocessing
import logging
import pdb
from glob import glob
import traceback
from copy import deepcopy

import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from util.rot_util import np_get_delta_qpos
from util.hand_util import HOMjSpec
from util.file_util import load_json


def interplote_pose(pose1: np.array, pose2: np.array, step: int) -> np.array:
    trans1, quat1 = pose1[:3], pose1[3:7]
    trans2, quat2 = pose2[:3], pose2[3:7]
    slerp = Slerp([0, 1], R.from_quat([quat1, quat2], scalar_first=True))
    trans_interp = np.linspace(trans1, trans2, step + 1)[1:]
    quat_interp = slerp(np.linspace(0, 1, step + 1))[1:].as_quat(scalar_first=True)
    return np.concatenate([trans_interp, quat_interp], axis=1)


def interplote_qpos(qpos1: np.array, qpos2: np.array, step: int) -> np.array:
    return np.linspace(qpos1, qpos2, step + 1)[1:]


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
        self.hospec = HOMjSpec(
            obj_path=self.grasp_data["obj_path"],
            obj_pose=self.grasp_data["obj_pose"],
            obj_scale=self.grasp_data["obj_scale"],
            has_floor_z0=False,
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
        for i in range(self.mj_model.ngeom):
            if "object_collision" in self.mj_model.geom(i).name:
                self.mj_model.geom_margin[i] = self.mj_model.geom_gap[i] = (
                    self.configs.task.contact_margin
                )

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

    def _resist_obj_gravity(self):
        if self.configs.debug_viewer:
            viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            viewer.sync()
            pdb.set_trace()

        _, _, pre_obj_pose = self.hospec.split_qpos_pose(self.mj_data.qpos)
        pre_obj_qpos = deepcopy(pre_obj_pose)

        dense_actu_moment = np.zeros((self.mj_model.nu, self.mj_model.nv))
        mujoco.mju_sparse2dense(
            dense_actu_moment,
            self.mj_data.actuator_moment,
            self.mj_data.moment_rownnz,
            self.mj_data.moment_rowadr,
            self.mj_data.moment_colind,
        )
        pregrasp_ctrl = dense_actu_moment[:, 6:-6] @ self.grasp_data["pregrasp_qpos"]
        grasp_ctrl = dense_actu_moment[:, 6:-6] @ self.grasp_data["grasp_qpos"]
        squeeze_ctrl = dense_actu_moment[:, 6:-6] @ self.grasp_data["squeeze_qpos"]

        external_force_direction = np.array(self.configs.task.external_force_direction)

        for i in range(len(external_force_direction)):
            # 1. Reset to pre-grasp pose
            mujoco.mj_resetDataKeyframe(
                self.mj_model, self.mj_data, self.mj_model.nkey - 2
            )
            self.mj_data.qfrc_applied[:] = 0.0
            self.mj_data.xfrc_applied[:] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)

            # 2. Check large penetration
            if not self._check_large_penetration():
                succ_flag = False
                delta_pos = 100
                delta_angle = 100
                break

            # 3. Move hand to grasp pose
            step_num = 10
            pose_interp = interplote_pose(
                self.grasp_data["pregrasp_pose"],
                self.grasp_data["grasp_pose"],
                step_num,
            )
            qpos_interp = interplote_qpos(pregrasp_ctrl, grasp_ctrl, step_num)
            for j in range(step_num):
                self.mj_data.mocap_pos[0] = pose_interp[j, :3]
                self.mj_data.mocap_quat[0] = pose_interp[j, 3:7]
                self.mj_data.ctrl[:] = qpos_interp[j]
                mujoco.mj_forward(self.mj_model, self.mj_data)
                for _ in range(10):
                    mujoco.mj_step(self.mj_model, self.mj_data)

            # 4. Move hand to squeeze pose.
            # NOTE step 3 and 4 are seperate because pre -> grasp -> squeeze are stage-wise linear.
            # If step 3 and 4 are merged to one linear interpolation, the performance will drop a lot.
            step_num = 10
            pose_interp = interplote_pose(
                self.grasp_data["grasp_pose"], self.grasp_data["squeeze_pose"], step_num
            )
            qpos_interp = interplote_qpos(grasp_ctrl, squeeze_ctrl, step_num)
            for j in range(step_num):
                self.mj_data.mocap_pos[0] = pose_interp[j, :3]
                self.mj_data.mocap_quat[0] = pose_interp[j, 3:7]
                self.mj_data.ctrl[:] = qpos_interp[j]
                mujoco.mj_forward(self.mj_model, self.mj_data)
                for _ in range(10):
                    mujoco.mj_step(self.mj_model, self.mj_data)

            # 5. Add external force on the object
            self.mj_data.xfrc_applied[-1] = (
                10 * external_force_direction[i] * self.configs.task.obj_mass
            )

            # 6. Wait for 2 seconds
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
        ho_pene, self_pene, contact_num, contact_dist, contact_consis = (
            self._eval_pene_and_contact()
        )
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
            eval_dict = {
                "ho_pene": ho_pene,
                "self_pene": self_pene,
                "contact_num": contact_num,
                "contact_dist": contact_dist,
                "contact_consis": contact_consis,
                "succ_flag": succ_flag,
                "delta_pos": delta_pos,
                "delta_angle": delta_angle,
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


def safe_eval_one(params):
    input_npy_path, configs = params[0], params[1]
    try:
        EvalOne(input_npy_path, configs).run()
        return
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.warning(f"{error_traceback}")
        return


def task_eval(configs):
    input_path_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    init_num = len(input_path_lst)

    if configs.skip:
        eval_path_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
        eval_path_lst = [
            p.replace(configs.eval_dir, configs.grasp_dir) for p in eval_path_lst
        ]
        input_path_lst = list(set(input_path_lst).difference(set(eval_path_lst)))
    skip_num = init_num - len(input_path_lst)
    input_path_lst = sorted(input_path_lst)
    if configs.task.max_num > 0:
        input_path_lst = np.random.permutation(input_path_lst)[: configs.task.max_num]

    logging.info(
        f"Find {init_num} grasp data, skip {skip_num}, and use {len(input_path_lst)}."
    )

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

    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
    logging.info(
        f"Get {len(grasp_lst)} grasp data, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )
    logging.info(f"Finish evaluation")

    return
