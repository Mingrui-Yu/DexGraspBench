import subprocess

# Parameters
exp_name = "learn"
max_num = -1
setting_names = ["dist_2"]
hands = ["dummy_arm_shadow"]
methods = ["ours"]
# methods = ["ours", "op", "bs1", "bs2", "bs3"]
ablation_names = ["ab2"]  # for "ours"


for setting in setting_names:
    if setting == "dist_0":
        offsets = "[0.00]"
    elif setting == "dist_2":
        offsets = "[0.02]"
    else:
        raise NotImplementedError

    for hand in hands:
        for method in methods:
            if method == "ours":
                ab_names = ablation_names
            else:
                ab_names = ["ab2"]

            for ab_name in ab_names:
                # Control eval
                cmd = [
                    "python",
                    "src/main.py",
                    "setting=tabletop",
                    f"hand={hand}",
                    "task=control_eval",
                    f"exp_name={exp_name}",
                    f"task.method={method}",
                    f"task.control.ablation_name={ab_name}",
                    f"task.offsets={offsets}",
                    "task.input_data=grasp_dir",
                    "task.debug_viewer=False",
                    "task.debug_render=True",
                    f"task.max_num={max_num}",
                    "n_worker=12",
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
