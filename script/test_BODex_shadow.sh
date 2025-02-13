rm -r output/bodex_shadow
python src/main.py task=format exp_name=bodex task.max_num=-1 task.data_path=/mnt/disk1/jiayichen/data/cuDex/sim_shadow/fc/debug/graspdata
python src/main.py task=eval exp_name=bodex task.max_num=-1 