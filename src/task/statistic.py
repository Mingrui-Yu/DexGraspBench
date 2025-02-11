import numpy as np
import os
from glob import glob
import multiprocessing
import logging
import matplotlib.pyplot as plt
import torch

from util.rot_utils import torch_quaternion_to_matrix, torch_matrix_to_axis_angle


def draw_obj_scale_fig(data_lst, save_path):
    obj_scale_lst = [float(d["obj_scale"]) for d in data_lst]

    bins = np.linspace(0.05, 0.2, 11)

    # Create the histogram
    plt.hist(
        np.array(obj_scale_lst),
        bins=bins,
        color="skyblue",
        edgecolor="black",
        rwidth=0.8,
    )

    # Add labels and title
    plt.xlabel("Scale")
    plt.ylabel("Frequency")
    plt.title("Distribution of Object Scales")

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    return


def read_data(npy_path):
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def get_diversity(data_lst):
    from sklearn.decomposition import PCA

    hand_poses = torch.tensor(
        np.stack([d["hand_pose"] for d in data_lst], axis=0)
    ).float()
    hand_qpos = torch.tensor(
        np.stack([d["hand_qpos"] for d in data_lst], axis=0)
    ).float()
    obj_poses = torch.tensor(
        np.stack([d["obj_pose"] for d in data_lst], axis=0)
    ).float()

    obj_rot = torch_quaternion_to_matrix(obj_poses[:, 3:])
    hand_rot = torch_quaternion_to_matrix(hand_poses[:, 3:])
    hand_real_trans = (
        obj_rot.transpose(-1, -2) @ (hand_poses[:, :3] - obj_poses[:, :3]).unsqueeze(-1)
    ).squeeze(-1)
    hand_real_rot = obj_rot.transpose(-1, -2) @ hand_rot
    hand_final_pose = torch.cat(
        [
            hand_real_trans,
            torch_matrix_to_axis_angle(hand_real_rot),
            hand_qpos,
        ],
        dim=-1,
    )

    pca = PCA()
    pca.fit(hand_final_pose.numpy())
    explained_variance = []
    for i in range(5):
        explained_variance.append(np.sum(pca.explained_variance_ratio_[: i + 1]))
    return explained_variance


def task_stat(configs):
    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
    logging.info(
        f"Find {len(grasp_lst)} grasp data, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )

    # Grasp success rate
    logging.info(f"Grasp success rate: {len(succ_lst)/len(eval_lst)}")

    # Object success rate
    obj_eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct[:-1])))
    obj_succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct[:-1])))
    logging.info(f"Object success rate: {len(obj_succ_lst)/len(obj_eval_lst)}")

    if len(eval_lst) == 0:
        logging.error("No evaluated grasp!")

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(read_data, eval_lst)
        data_lst = list(result_iter)

    if configs.task.scale_fig:
        save_path = os.path.join(os.path.dirname(configs.log_path), "objscale.png")
        draw_obj_scale_fig(data_lst, save_path)

    if configs.task.diversity:
        pca_eigenvalue = get_diversity(data_lst)
        logging.info(f"Diversity: {pca_eigenvalue}")

    average_penetration_depth = np.mean([d["ho_pene"] for d in data_lst])
    logging.info(f"Penetration depth: {average_penetration_depth}")
    average_self_penetration_depth = np.mean([d["self_pene"] for d in data_lst])
    logging.info(f"Self-penetration depth: {average_self_penetration_depth}")
    average_contact_distance = np.mean([d["contact_dist"] for d in data_lst])
    logging.info(f"Contact distance: {average_contact_distance}")
    average_contact_number = np.mean([d["contact_num"] for d in data_lst])
    logging.info(f"Contact number: {average_contact_number}")
    average_contact_consis = np.mean([d["contact_consis"] for d in data_lst])
    logging.info(f"Contact consistency: {average_contact_consis}")
