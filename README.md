# DexGraspBench

The code base of our tactile-grasp work.

## Getting Started

### Installation
1. Clone the third-party library [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
  ```
  git submodule update --init --recursive --progress
  ```
2. Third party:
  ```bash
  cd third_party
  # pytorch_kinematics
  git clone git@github.com:DexGrasp-TH/pytorch_kinematics.git
  cd pytorch_kinematics
  git checkout v1.0.0
  pip install -e .

  # mr_utils
  cd third_party
  git clone git@github.com:Mingrui-Yu/utils_python.git
  cd utils_python
  pip install -e .
  ```
3. Install the python environment via [Anaconda](https://www.anaconda.com/). 
  ```bash
  conda create -n DGBench python=3.10 
  conda activate DGBench
  pip install numpy==1.26.4
  conda install pytorch==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia 
  pip install mujoco==3.3.2
  pip install trimesh
  pip install hydra-core
  pip install transforms3d
  pip install matplotlib
  pip install scikit-learn
  pip install usd-core
  pip install imageio
  pip install 'qpsolvers[clarabel]'
  pip install tqdm
  conda install pinocchio -c conda-forge
  ```

### Object Preparation
For the object assets used in [BODex](https://pku-epic.github.io/BODex/), please download our pre-processed object assets `DGN_2k_processed.zip` from [here](https://huggingface.co/datasets/JiayiChenPKU/BODex) and organize the unzipped folders as below. 
```
assets/object/DGN_2k
|- processed_data
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- scene_cfg
|  |- core_bottle_1a7ba1f4c892e2da30711cdbdbc73924
|  |_ ...
|- valid_split
|  |- all.json
|  |_ ...
```

## Usage

### Complete pipeline
1. Use BODex to synthesize grasp poses.
1. Use DexLearn to train a generative network.
1. Use DexLearn to sample grasp poses from single-view point clouds.
1. Use this repo to evaluate the grasping execution methods.

### Evaluation of the grasping execution methods
Complete procedures for each hand:
```bash
bash script/test_learning_dummy_arm_shadow.sh
bash script/test_learning_dummy_arm_allegro.sh
bash script/test_learning_dummy_arm_leap_tac3d.sh
```
Each includes the following tasks:
* format: convert the data format.
* dummy_arm_qpos: calculate the qpos of the dummy_arm via IK.
* control_eval: evaluate the execution method and save the manipulation data.
* control_stat: compute and save the statistic results.

Evaluation dataset:
* learn: 100 grasps for each hand.
* learn_large: 1k grasps for each hand.
* learn_5k: 5k grasps for each hand.

After generating the qpos of dummy arms, the control_eval and control_stat under different conditions can be conveniently run via the following scripts:
```bash
# local PC
python script/test_all_ablation_baseline_local.py # need internal modification of the settings
# server
python script/test_all_ablation_baseline.py # need internal modification of the settings
```

To run a specific case, change the case indices in `src/task/control_eval.py`
and the position indices in `src/task/control_eval_func/base.py`. 

We use `ab2` as the default parameters of `ours`.

### Video rendering
1. Change the viewpoint specified in `src/util/hand_util.py`.
1. Change the video type and save path in `src/task/control_eval_func/base.py`.
1. Run:
    ```bash
    python script/test_sim_render_local.py # need internal modification of the settings
    ```

### Visualze static grasps via trimesh
The scripts are at `script/quick_grasp_vis`.

## Tips
* The current code does not fully consider underactuated hands. Need improvement.








