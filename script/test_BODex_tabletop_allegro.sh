rm -r output/debug_allegro
python src/main.py task=format hand=allegro setting=tabletop exp_name=debug task.max_num=-1 task.data_path=../BODex/src/curobo/content/assets/output/sim_allegro/tabletop/debug/graspdata
python src/main.py task=eval hand=allegro setting=tabletop exp_name=debug task.max_num=-1
python src/main.py task=stat hand=allegro setting=tabletop exp_name=debug
python src/main.py task=vusd hand=allegro setting=tabletop exp_name=debug task.max_num=10
python src/main.py task=vobj hand=allegro setting=tabletop exp_name=debug task.max_num=10
python src/main.py task=collect hand=allegro setting=tabletop exp_name=debug
