rm -r output/debug_shadow
python src/main.py setting=tabletop hand=shadow task=format exp_name=debug task.max_num=100 task.data_path=../BODex/src/curobo/content/assets/output/sim_shadow/tabletop/debug/graspdata
python src/main.py setting=tabletop hand=shadow task=eval exp_name=debug task.max_num=1000
python src/main.py setting=tabletop hand=shadow task=stat exp_name=debug
python src/main.py setting=tabletop hand=shadow task=vusd exp_name=debug task.max_num=10
python src/main.py setting=tabletop hand=shadow task=vobj exp_name=debug task.max_num=10
python src/main.py setting=tabletop hand=shadow task=collect exp_name=debug
