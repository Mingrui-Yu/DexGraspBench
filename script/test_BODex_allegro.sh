rm -r output/bodex_allegro
python src/main.py hand=allegro task=format exp_name=bodex task.max_num=-1 task.data_path=/mnt/disk1/jiayichen/data/cuDex/sim_allegro/fc/DGNObj/graspdata
python src/main.py hand=allegro task=eval exp_name=bodex task.max_num=-1
# python src/main.py task=vusd exp_name=bodex hand=allegro 
