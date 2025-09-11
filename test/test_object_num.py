import os
import sys

def count_subdirs_and_files(path):
    level1_dirs = []
    level2_dirs = []
    level3_files = []

    for root, dirs, files in os.walk(path):
        relpath = os.path.relpath(root, path)
        depth = 0 if relpath == "." else relpath.count(os.sep) + 1

        if depth == 1:  # 一级目录下的文件夹
            for d in dirs:
                level1_dirs.append(os.path.join(root, d))
        elif depth == 2:  # 二级目录下的文件夹
            for d in dirs:
                level2_dirs.append(os.path.join(root, d))
        elif depth == 3:  # 三级目录下的文件
            for f in files:
                level3_files.append(os.path.join(root, f))

    return len(level1_dirs), len(level2_dirs), len(level3_files)

if __name__ == "__main__":
    folder = "output/learn_5k_dummy_arm_shadow/graspdata"
    n1, n2, n3 = count_subdirs_and_files(folder)
    print(f"{folder} 下：")
    print(f"  一级子文件夹数: {n1}")
    print(f"  二级子文件夹数: {n2}")
    print(f"  三级文件数: {n3}")

    folder = "output/learn_5k_dummy_arm_allegro/graspdata"
    n1, n2, n3 = count_subdirs_and_files(folder)
    print(f"{folder} 下：")
    print(f"  一级子文件夹数: {n1}")
    print(f"  二级子文件夹数: {n2}")
    print(f"  三级文件数: {n3}")


    folder = "output/learn_5k_dummy_arm_leap_tac3d/graspdata"
    n1, n2, n3 = count_subdirs_and_files(folder)
    print(f"{folder} 下：")
    print(f"  一级子文件夹数: {n1}")
    print(f"  二级子文件夹数: {n2}")
    print(f"  三级文件数: {n3}")


