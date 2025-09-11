import numpy as np


def compare_npy_dicts(file1, file2):
    # 载入字典
    d1 = np.load(file1, allow_pickle=True).item()
    d2 = np.load(file2, allow_pickle=True).item()

    # 比较 keys
    keys1, keys2 = set(d1.keys()), set(d2.keys())
    if keys1 != keys2:
        print("⚠️ Keys 不同:")
        print("  仅在 file1:", keys1 - keys2)
        print("  仅在 file2:", keys2 - keys1)

    # 比较内容
    for k in keys1 & keys2:
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if np.array_equal(v1, v2):
                print(f"[OK] {k}: 两个数组完全相同, shape={v1.shape}")
            else:
                diff = np.abs(v1 - v2)
                print(f"[DIFF] {k}: 数组不同, shape={v1.shape}, max diff={diff.max()}")
        else:
            if v1 == v2:
                print(f"[OK] {k}: 值相同 ({v1})")
            else:
                print(f"[DIFF] {k}: 值不同, file1={v1}, file2={v2}")


if __name__ == "__main__":
    compare_npy_dicts(
        "output/learn_leap_tac3d/graspdata/core_bottle_547fa0085800c5c3846564a8a219239b/tabletop_ur10e/scale006_pose002_0/partial_pc_03_0.npy",
        "output/partial_pc_03_0.npy",
    )
