mkdir -r output/learn_step5_shadow
ln -s xxxxxx output/learn_step5_shadow/graspdata
python src/main.py task=eval exp_name=learn_step5 task.max_num=-1