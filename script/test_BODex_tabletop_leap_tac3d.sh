rm -r output/debug_leap_tac3d
python src/main.py task=format hand=leap_tac3d setting=tabletop exp_name=debug task.max_num=-1 task.data_path=../BODex/src/curobo/content/assets/output/sim_leap_tac3d/tabletop/debug/graspdata
python src/main.py task=eval hand=leap_tac3d setting=tabletop exp_name=debug task.max_num=-1
python src/main.py task=stat hand=leap_tac3d setting=tabletop exp_name=debug
python src/main.py task=vusd hand=leap_tac3d setting=tabletop exp_name=debug task.max_num=10
python src/main.py task=vobj hand=leap_tac3d setting=tabletop exp_name=debug task.max_num=10
python src/main.py task=collect hand=leap_tac3d setting=tabletop exp_name=debug