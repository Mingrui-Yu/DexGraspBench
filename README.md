# DexGraspBench

A standard and unified simulation benchmark in [MuJoCo](https://github.com/google-deepmind/mujoco/) for dexterous grasping, aimed at **enabling a fair comparison across different grasp synthesis methods**, proposed in *BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization [ICRA 2025]*.

[Project page](https://pku-epic.github.io/BODex/) ｜ [Paper](https://arxiv.org/abs/2412.16490)

## Main Usage
Replay and test **open-loop** grasping poses/trajectories in parallel.

Each grasping data point should include:
- Object (pre-processed by [MeshProcess](https://github.com/JYChen18/MeshProcess)): `obj_scale`, `obj_pose`, `obj_path`.
- Hand: `approach_qpos` (optional), `pregrasp_qpos`, `grasp_qpos`, `squeeze_qpos`.


## Highlight

1. **Reproducible and high-quality physics simulation**, powered by [MuJoCo](https://github.com/google-deepmind/mujoco/).
2. **Comprehensive metrics**, including simulation success rate, analytic force closure metrics, penetration depth, contact quality, and data diversity, etc.
3. **Diverse settings**, including different robotic hands (Allegro, Shadow, Leap, UR10e+Shadow), data formats (motion sequences, static poses), and scenarios (tabletop lifting, force-closure testing).
4. **Multiple baselines**, including optimization-based grasp synthesis methods (e.g., [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet), [FRoGGeR](https://github.com/alberthli/frogger), [SpringGrasp](https://github.com/Stanford-TML/SpringGrasp_release), and [BODex](https://pku-epic.github.io/BODex/)) and data-driven baselines (e.g., CVAE, Diffusion Model, and Normalizing Flow).


## Installation
1. Clone the third-party library [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
```
git submodule update --init --recursive --progress
```
2. Install the python environment via [Anaconda](https://www.anaconda.com/). 
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
pip install usd-core
pip install imageio
pip install 'qpsolvers[clarabel]'
```
<!-- 
## Object Asset Preparation
1. Download the object assets of DexGraspNet from [TODO]().
2. Process the objects using [MeshProcess](https://github.com/JYChen18/MeshProcess).
3. Create soft link to `assets`.
```
ln -s ${YOUR_DATA_PATH} assets/DGNObj
ln -s ${YOUR_SPLIT_PATH} assets/DGNObj_splits
``` -->

## Running
1. (Optional) Download the object assets of DexGraspNet from [TODO]() and process with [MeshProcess](https://github.com/JYChen18/MeshProcess). Create soft link to `assets`.
```
ln -s ${YOUR_DATA_PATH} assets/DGNObj
```

2. (Optional) Synthesize new grasp data with [BODex]() and convert to our supported format. There are also some all-in-one scripts in `script` to test BODex's grasps.
```
python src/main.py task=format task.data_path=${YOUR_PATH_TO_BODEX_OUTPUT} exp_name=debug hand=allegro
```

3. Evaluate the synthesized grasps. For a quick start, some example data is provided in the `output/example_shadow` directory. 
```
python src/main.py task=eval exp_name=example hand=shadow
```

4. Calculate statistics after evaluation.
```
python src/main.py task=stat
```

## Visualization
We provide two methods to visualize the synthesized grasps. The first method is through [OpenUSD](https://github.com/PixarAnimationStudios/OpenUSD). 
```
python src/main.py task=vusd
```
The other method is to save OBJ files.
```
python src/main.py task=vobj
```

## Setting Changes
The `main` branch serves as our standard benchmark, with some adjustments to the settings compared to the [BODex](https://arxiv.org/abs/2412.16490) paper, aimed at improving the practicality. Key changes include increasing the object mass from 30g to 100g, raising the hand's kp from 1 to 5, and supporting more diverse object assets.

The original benchmark version is available in the `original` branch. This branch also includes code to test other grasp synthesis baselines, such as [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet), [FRoGGeR](https://github.com/alberthli/frogger), [SpringGrasp](https://github.com/Stanford-TML/SpringGrasp_release).


## Future Plans
1. Incorporate visual/tactile feedback to support **close-loop** evaluation.
2. Add support for other physics simulators, such as [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) and the [IPC-based simulator](https://dl.acm.org/doi/10.1145/3528223.3530064).

Contributions are welcome, and feel free to connect with me via [email](mailto:jiayichen@pku.edu/cn).


## Citation
If you find this project useful, please consider citing:
```
@article{chen2024bodex,
  title={BODex: Scalable and Efficient Robotic Dexterous Grasp Synthesis Using Bilevel Optimization},
  author={Chen, Jiayi and Ke, Yubin and Wang, He},
  journal={arXiv preprint arXiv:2412.16490},
  year={2024}
}
```
