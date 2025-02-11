# DexGraspBench



### Installation
```
git submodule update --init --recursive 
```

```
conda create -n DGBench python=3.10 
conda activate DGBench
pip install numpy==1.26.4
conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia 
pip install mujoco
pip install trimesh
pip install hydra-core
pip install transforms3d
pip install matplotlib
pip install scikit-learn
```

### Object Asset Preparation

### Running
```
# Evaluation   
python src/main.py task=eval       

# Calculate statistics
python src/main.py task=stat

# Visualization with OpenUSD
python src/main.py task=vusd task.max_num=10

# Visualization with OBJ file
python src/main.py task=vobj 'task.data_type=[grasp, pregrasp]'

```