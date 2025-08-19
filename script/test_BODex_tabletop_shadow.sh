rm -r output/debug_shadow
python src/main.py task=format hand=shadow setting=tabletop exp_name=debug task.max_num=100 task.data_path=../BODex/src/curobo/content/assets/output/sim_shadow/tabletop/debug/graspdata
python src/main.py task=eval hand=shadow setting=tabletop exp_name=debug task.max_num=100
python src/main.py task=stat hand=shadow setting=tabletop exp_name=debug
python src/main.py task=vusd hand=shadow setting=tabletop exp_name=debug task.max_num=10
python src/main.py task=vobj hand=shadow setting=tabletop exp_name=debug task.max_num=10
python src/main.py task=collect hand=shadow setting=tabletop exp_name=debug