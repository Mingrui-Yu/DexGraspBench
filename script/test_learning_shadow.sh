rm -r output/learn_shadow
python src/main.py hand=shadow exp_name=learn task=format task.max_num=-1 task.data_name=Learning task.data_path=../DexLearn/output/dexonomy_shadow_nflow_type1/tests/step_015000
python src/main.py hand=shadow task=eval exp_name=learn task.max_num=-1
python src/main.py task=stat hand=shadow exp_name=learn