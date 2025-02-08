import trimesh
import numpy as np
import os
from glob import glob
import logging
from copy import deepcopy
import multiprocessing

from util.file_utils import load_yaml


def _single_visd(params):

    data_path, data_folder, check_folder, configs = (
        params[0],
        params[1],
        params[2],
        params[3],
    )
    task_config = configs.task

    check_path = data_path.replace(data_folder, check_folder)
    out_path = (
        data_path.replace(data_folder, configs.visd_dir)
        .replace(".npy", "")
        .replace(".yaml", "")
    )

    succ = os.path.exists(check_path)
    if task_config.data_type in ["grasp", "init"] and succ != task_config.check_succ:
        return

    mesh_data = kinematic.get_mesh(
        data_path,
        vis_sk=task_config.vis_hand_sk,
        vis_contact=task_config.vis_hand_contact,
        vis_init=task_config.vis_hand_init,
    )
    if data_path.endswith(".npy"):
        grasp_data = np.load(data_path, allow_pickle=True).item()
    elif data_path.endswith(".yaml"):
        grasp_data = load_yaml(data_path)
    else:
        print(f"Wrong input format: {data_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    visual_mesh = deepcopy(mesh_data["visual_mesh"])
    rotation_matrix = trimesh.transformations.quaternion_matrix(
        grasp_data["hand_pose"][3:]
    )
    rotation_matrix[:3, 3] = grasp_data["hand_pose"][:3]
    visual_mesh.apply_transform(rotation_matrix)
    visual_mesh.export(out_path + "_hand.obj")

    if task_config.vis_obj_mesh:
        obj_path = os.path.join(grasp_data["obj_path"], "mesh/coacd.obj")
        obj_tm = trimesh.load(obj_path, force="mesh")
        obj_tm.vertices *= grasp_data["obj_scale"]
        rotation_matrix = trimesh.transformations.quaternion_matrix(
            grasp_data["obj_pose"][3:]
        )
        rotation_matrix[:3, 3] = grasp_data["obj_pose"][:3]
        obj_tm.apply_transform(rotation_matrix)
        obj_tm.export(out_path + "_obj.obj")

    if task_config.vis_hand_sk:
        mesh_data["skeleton_mesh"].export(out_path + "_skeleton.obj")

    if task_config.vis_hand_contact:
        mesh_data["contact_point_mesh"].export(out_path + "_handcp.obj")
        mesh_data["contact_normal_mesh"].export(out_path + "_handcn.obj")

    if task_config.vis_hand_init:
        os.makedirs(
            os.path.join(os.path.dirname(out_path), "hand_raw_mesh"), exist_ok=True
        )
        for init_name, init_mesh in zip(
            mesh_data["init_name_lst"], mesh_data["init_mesh_lst"]
        ):
            init_mesh.export(
                os.path.join(
                    os.path.dirname(out_path), f"hand_raw_mesh/{init_name}.obj"
                )
            )

    if task_config.vis_obj_contact:
        np.savetxt(out_path + "_objcontact.txt", grasp_data["obj_worldframe_contacts"])

    logging.info(f"save to {os.path.dirname(os.path.abspath(out_path))}")

    return


def task_vis3d(configs):
    task_config = configs.task
    output_folder = configs.visd_dir
    data_folder = configs.grasp_dir
    check_folder = configs.succ_dir
    input_path_lst = glob(
        os.path.join(data_folder, *list(configs.data_struct))
    )
   

    if task_config.one_for_each_obj:
        final_path_lst = []
        data_dict = {}
        for p in input_path_lst:
            folder_name = os.path.dirname(p)
            if folder_name not in data_dict.keys():
                data_dict[folder_name] = True
                final_path_lst.append(p)
        input_path_lst = final_path_lst

    iterable_params = [
        (inp, data_folder, check_folder, configs) for inp in input_path_lst
    ]

    with multiprocessing.Pool(processes=configs.n_worker) as pool:
        result_iter = pool.imap_unordered(_single_visd, iterable_params)
        results = list(result_iter)

    return
