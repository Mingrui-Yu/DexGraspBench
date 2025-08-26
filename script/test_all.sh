#!/bin/bash


hands=("dummy_arm_shadow" "dummy_arm_allegro" "dummy_arm_leap_tac3d")
# hands=("dummy_arm_shadow")
methods=("ours" "op" "bs1" "bs2" "bs3" "bs4")
offsets="[0,0.02]" # unit: m; should be no space after comma
setting_names=("dist_0" "dist_2")

# control_eval
for hand in "${hands[@]}"; do
  for method in "${methods[@]}"; do
    python src/main.py setting=tabletop hand=$hand task=control_eval exp_name=learn task.method=$method task.offsets=$offsets task.input_data=grasp_dir task.debug_viewer=False
  done
done

# control_stat
for setting in "${setting_names[@]}"; do
  for hand in "${hands[@]}"; do
    for method in "${methods[@]}"; do
      python src/main.py setting=tabletop hand=$hand task=control_stat exp_name=learn task.method=$method task.setting_name=$setting
    done
  done
done
