import subprocess

# Parameters
hands = ["dummy_arm_shadow", "dummy_arm_allegro", "dummy_arm_leap_tac3d"]
# hands = ["dummy_arm_leap_tac3d"]
methods = ["ours"]  # can only be 'ours'
ablation_names = ["ab0", "ab1", "ab2", "ab3", "ab4", "ab5"]

offsets = "[0.02]"  # unit: m
setting_names = ["dist_2"]

for hand in hands:
    for method in methods:
        for ab_name in ablation_names:
            # Control eval
            cmd = [
                "python",
                "src/main.py",
                "setting=tabletop",
                f"hand={hand}",
                "task=control_eval",
                "exp_name=learn",
                f"task.method={method}",
                f"task.control.ablation_name={ab_name}",
                f"task.offsets={offsets}",
                "task.input_data=grasp_dir",
                "task.debug_viewer=False",
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            # Control stat
            for setting in setting_names:
                cmd = [
                    "python",
                    "src/main.py",
                    "setting=tabletop",
                    f"hand={hand}",
                    "task=control_stat",
                    "exp_name=learn",
                    f"task.method={method}",
                    f"task.ablation_name={ab_name}",
                    f"task.setting_name={setting}",
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)
