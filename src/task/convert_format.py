import os
from glob import glob
import logging
import multiprocessing

import numpy as np


def BODex():
    return


def FRoGGeR():
    return


def SpringGrasp():
    return


def DGN():
    return


def task_format(configs):
    input_path_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))

    if configs.task.max_num > 0:
        input_path_lst = np.random.permutation(sorted(input_path_lst))[
            : configs.task.max_num
        ]
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
    grasp_lst = glob(os.path.join(configs.grasp_dir, *list(configs.data_struct)))
    succ_lst = glob(os.path.join(configs.succ_dir, *list(configs.data_struct)))
    eval_lst = glob(os.path.join(configs.eval_dir, *list(configs.data_struct)))
    logging.info(
        f"Find {len(grasp_lst)} grasp data, {len(eval_lst)} evaluated, and {len(succ_lst)} succeeded in {configs.save_dir}"
    )
    return
