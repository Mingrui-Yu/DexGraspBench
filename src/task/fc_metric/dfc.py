import os
import sys

import numpy as np

from .base import build_grasp_matrix


def calcu_dfc_metric(contact_pos, contact_normal):
    grasp_matrix = build_grasp_matrix(
        contact_pos, contact_normal, contact_pos.mean(axis=0)
    )
    dfc_metric = np.linalg.norm(np.sum(grasp_matrix[..., 0], axis=0))
    return dfc_metric
