# generate grasp dataset on the server

# shadow
python src/main.py setting=tabletop hand=shadow task=format exp_name=learn_5k task.data_name=Learning task.max_num=1000 task.data_path=../DexLearn/output/bodex_tabletop_shadow_nflow_debug0/tests/step_045000
python src/main.py setting=tabletop hand=shadow task=dummy_arm_qpos exp_name=learn_5k task.max_num=-1

# allegro
python src/main.py setting=tabletop hand=allegro task=format exp_name=learn_5k task.data_name=Learning task.max_num=1000 task.data_path=../DexLearn/output/bodex_tabletop_allegro_nflow_debug1/tests/step_050000
python src/main.py setting=tabletop hand=allegro task=dummy_arm_qpos exp_name=learn_5k task.max_num=-1

# leap_tac3d
python src/main.py setting=tabletop hand=leap_tac3d task=format exp_name=learn_5k task.data_name=Learning task.max_num=1000 task.data_path=../DexLearn/output/bodex_tabletop_leap_tac3d_nflow_debug0/tests/step_050000
python src/main.py setting=tabletop hand=leap_tac3d task=dummy_arm_qpos exp_name=learn_5k task.max_num=-1

















