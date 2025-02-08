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
pip install pyyaml
pip install hydra-core
pip install transforms3d
pip install matplotlib
```

### Object Asset Preparation

### Run Evaluation