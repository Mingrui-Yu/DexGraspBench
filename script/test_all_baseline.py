import subprocess

# Parameters
hands = ["dummy_arm_shadow", "dummy_arm_allegro", "dummy_arm_leap_tac3d"]
# hands = ["dummy_arm_leap_tac3d"]
methods = ["bs1"]

offsets = "[0.00]"  # unit: m
setting_names = ["dist_0"]


for hand in hands:
    for method in methods:
        # Control eval
        cmd = [
            "python",
            "src/main.py",
            "setting=tabletop",
            f"hand={hand}",
            "task=control_eval",
            "exp_name=learn",
            f"task.method={method}",
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
                f"task.setting_name={setting}",
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
