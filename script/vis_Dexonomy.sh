rm -r output/dexonomy_shadow
python src/main.py hand=shadow exp_name=dexonomy task=format task.max_num=10 task.data_name=Batched task.data_path=output/Dexonomy_GRASP_shadow/succ_collect/4_Adducted_Thumb
python src/main.py task=vusd hand=shadow exp_name=dexonomy
python src/main.py task=vobj hand=shadow exp_name=dexonomy