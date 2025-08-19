import os


def count_files(path, recursive=False):
    dir_count = sum(1 for entry in os.scandir(path) if entry.is_dir())
    print(f"Number of objects: {dir_count}")

    file_count = 0
    for root, dirs, files in os.walk(path):
        file_count += sum(1 for f in files if f.startswith("partial"))
    print(f"Number of grasps: {file_count}")


if __name__ == "__main__":
    path = "/home/mingrui/mingrui/research/adaptive_grasping_2/DexLearn/output/bodex_tabletop_shadow_nflow_debug0/tests/step_045000/"  # replace with your directory path
    count_files(path, recursive=False)
