import os
from glob import glob
import logging
import multiprocessing

import numpy as np
import transforms3d


def BODex(params):
    data_file, configs = params[0], params[1]

    raw_data = np.load(data_file, allow_pickle=True).item()
    robot_pose = raw_data["robot_pose"][0]

    if configs.hand_name == "shadow":
        robot_pose = np.concatenate(
            [robot_pose[:, :, :7], robot_pose[:, :, 12:], robot_pose[:, :, 7:12]],
            axis=-1,
        )
    else:
        raise NotImplementedError
    pregrasp_pose = robot_pose[:, 0, :7]
    pregrasp_qpos = robot_pose[:, 0, 7:]
    hand_qpos = robot_pose[:, 1, 7:]
    hand_pose = robot_pose[:, 1, :7]
    new_data = {}
    new_data["obj_pose"] = raw_data["obj_pose"][0]
    new_data["obj_scale"] = raw_data["obj_scale"][0]
    new_data["obj_path"] = f"assets/DGNObj/{data_file.split('/')[-2]}"

    for i in range(len(robot_pose)):
        if configs.hand_name == "shadow":
            tmp_hand_pose_rot = transforms3d.quaternions.quat2mat(hand_pose[i][3:])
            hand_pose[i][:3] -= tmp_hand_pose_rot @ np.array([0, 0, 0.034])
            tmp_hand_pose_rot = transforms3d.quaternions.quat2mat(pregrasp_pose[i][3:])
            pregrasp_pose[i][:3] -= tmp_hand_pose_rot @ np.array([0, 0, 0.034])
        new_data["hand_pose"] = hand_pose[i]
        new_data["hand_qpos"] = hand_qpos[i]
        new_data["pregrasp_pose"] = pregrasp_pose[i]
        new_data["pregrasp_qpos"] = pregrasp_qpos[i]
        new_data["squeeze_qpos"] = hand_qpos[i] * 2.5 - pregrasp_qpos[i] * 1.5
        save_path = data_file.replace(
            configs.task.data_path, configs.grasp_dir
        ).replace("_grasp.npy", f"/{i}.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, new_data)
    return


def FRoGGeR():
    return


def SpringGrasp():
    return


def DGN():
    return


def task_format(configs):
    if configs.task.data_name == "BODex":
        raw_data_struct = ["**", "**_grasp.npy"]
    else:
        raise NotImplementedError

    raw_data_path_lst = glob(os.path.join(configs.task.data_path, *raw_data_struct))

    if configs.task.max_num > 0:
        raw_data_path_lst = np.random.permutation(sorted(raw_data_path_lst))[
            : configs.task.max_num
        ]
    logging.info(f"Find {len(raw_data_path_lst)} raw files")

    if len(raw_data_path_lst) == 0:
        return

    iterable_params = zip(raw_data_path_lst, [configs] * len(raw_data_path_lst))
    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(eval(configs.task.data_name), iterable_params)
        results = list(result_iter)

    logging.info(f"Finish")
    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    logging.info(f"Get {len(grasp_lst)} grasp data in {configs.save_dir}")
    return
