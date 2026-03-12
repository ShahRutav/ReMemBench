# ReMemBench: Scaling Short-Term Memory of Visuomotor Policies for Long-Horizon Tasks
<!-- ![alt text](https://github.com/UT-Austin-RPL/maple/blob/web/src/overview.png) -->
<img src="docs/images/remembench-banner.png" width="100%" />

This is the official codebase of [ReMemBench](https://shahrutav.github.io/short-term-memory/), built upon [RoboCasa](https://robocasa.ai/), a benchmark for training and evaluating visuomotor policies with short-term memory. This guide contains information about installation and setup.

[**[Home page]**](https://shahrutav.github.io/short-term-memory/) &ensp; [**[Paper]**](https://shahrutav.github.io/short-term-memory/)
-------

### Task Categories

Tasks are organized by memory type. Each task is provided with 50 expert demonstrations for training.

| Task Name | Memory Category | Task Variants |
|-----------|----------------|---------------|
| **Retrieve Fruit**<br/>Remember fruit location (out of view). Varies: fruit type, placement position | **Spatial Memory**<br/>*Recall object locations* | `MemFruitInSinkLeftFar`<br/>`MemFruitInSinkRightFar` |
| **Retrieve Oil**<br/>Remember oil bottle location among distractors. Varies: bottle instance, counter position | **Spatial Memory**<br/>*Recall object locations* | `MemRetrieveOilsFromCounterLL`<br/>`MemRetrieveOilsFromCounterLR`<br/>`MemRetrieveOilsFromCounterRL`<br/>`MemRetrieveOilsFromCounterRR` |
| **Cook Meat**<br/>Remember cooking duration while waiting. Varies: duration, pan type, meat type | **Prospective Memory**<br/>*Retain intentions over delay* | `MemHeatPot` |
| **Cook Meat and Vegetable**<br/>Remember multiple timed actions (add vegetable, turn off). Varies: durations, pan/ingredient types, vegetable location | **Prospective Memory**<br/>*Retain intentions over delay* | `MemHeatPotMultiple` |
| **Wash and Return to Container**<br/>Remember which saucer (left/right) the fruit came from. Varies: fruit type, saucer type, side | **Object-Associative Memory**<br/>*Recall object-location associations* | `MemWashAndReturnLeft`<br/>`MemWashAndReturnRight` |
| **Wash and Return to Original Spot**<br/>Remember original countertop location. Varies: initial position, fruit type | **Object-Associative Memory**<br/>*Recall object-location associations* | `MemWashAndReturnSameLocation` |
| **Microwave Breadsticks**<br/>Remember count of breadsticks moved. Varies: count, bread type, positions | **Object-Set Memory**<br/>*Maintain and update sets of objects* | `MemPutKBreadInMicrowave` |
| **Relocate Bowls**<br/>Remember count of bowls among distractor plates. Varies: bowl types, counts, positions | **Object-Set Memory**<br/>*Maintain and update sets of objects* | `MemPutKBowlInCabinet` |

-------
## Installation
RoboCasa works across all major computing platforms. The easiest way to set up is through the [Anaconda](https://www.anaconda.com/) package management system. Follow the instructions below to install:
1. Set up conda environment:

   ```sh
   conda create -c conda-forge -n robocasa python=3.10
   ```
2. Activate conda environment:
   ```sh
   conda activate memory 
   ```
3. Clone and setup robosuite dependency (**important: use the master branch!**):

   ```sh
   git clone --branch=abs_robot https://github.com/ShahRutav/robosuite
   cd robosuite
   pip install -e .
   ```
4. Clone and setup this repo:

   ```sh
   cd ..
   git clone --branch=release https://github.com/ShahRutav/robocasa
   cd robocasa
   pip install -e .
   pip install pre-commit; pre-commit install           # Optional: set up code formatter.

   (optional: if running into issues with numba/numpy, run: conda install -c numba numba=0.56.4 -y)
   ```
5. Install the package and download assets:
   ```sh
   python robocasa/scripts/download_kitchen_assets.py   # Caution: Assets to be downloaded are around 5GB.
   python robocasa/scripts/setup_macros.py              # Set up system variables.
   ```

-------
## Dataset Overview

The memory dataset is provided at `Rutav/ReMemBench-Dataset` and contains expert teleoperated demonstrations for various manipulation tasks.

## Data Downloading 
```
huggingface-cli download Rutav/ReMemBench-Dataset \
  --repo-type dataset \
  --local-dir ReMemBench-Dataset \
  --local-dir-use-symlinks False
```

### File Structure

The dataset is organized by task name, with each task containing one or more demonstration sessions:

```
ReMemBench-Dataset/
├── MemFruitInSinkLeftFar/
├── MemFruitInSinkRightFar/
├── MemHeatPot/
│   ├── 2025-07-24-22-26-20/
│   │   ├── demo.hdf5
│   │   └── demo_im128.hdf5  # Image version
├── MemHeatPotMultiple/
├── MemPutKBowlInCabinet/
├── MemPutKBreadInMicrowave/
├── MemRetrieveOilsFromCounterLL/
├── MemRetrieveOilsFromCounterLR/
├── MemRetrieveOilsFromCounterRL/
├── MemRetrieveOilsFromCounterRR/
├── MemWashAndReturnLeft/
├── MemWashAndReturnRight/
├── MemWashAndReturnSameLocation/
└── task_embeds_clip_v3.pickle
```

### HDF5 File Structure

Each session directory contains:
- **`demo.hdf5`**: Standard demonstration file with original state information (no images)
- **`demo_im128.hdf5`**: Image version with 128x128 RGB images

#### demo_im128.hdf5 Structure

Below is the dataset structure of hdf5 file provided:

```
demo_im128.hdf5
└── data (group)
    └── [TaskName]_[Robot]_demo_1 (group)
    │   ├── actions (dataset) - shape: (T, 12)
    │   │   └── Action: [7 dimensions for arm, 4 dimensions for base, 1 dimension for arm / base mode]
    │   ├── action_dict (group) - structured action components
    │   ├── obs (group) - observations with images
    │   │   ├── robot0_joint_pos_cos - shape: (T, 7) - joint position cosine encoding
    │   │   ├── robot0_joint_pos_sin - shape: (T, 7) - joint position sine encoding
    │   │   ├── robot0_gripper_qpos - shape: (T, 2) - gripper position
    │   │   ├── robot0_agentview_center_image - shape: (T, 128, 128, 3) - RGB camera view
    │   │   ├── robot0_eye_in_hand_image - shape: (T, 128, 128, 3) - eye-in-hand camera view
    │   │   └── [other state information]
    │   ├── states (dataset) - MuJoCo states
    │   ├── dones (dataset) - episode termination flags
    │   ├── rewards (dataset) - reward signals
    │   └── policy_mode (dataset: 1 is )
    └── [TaskName]_[Robot]_demo_1 (group)
    └── ...
```

**Key Observation Fields:**
- **Joint States**: `robot0_joint_pos_cos` (T, 7) and `robot0_joint_pos_sin` (T, 7) - 7-DOF joint positions encoded as cosine/sine pairs
- **Gripper State**: `robot0_gripper_qpos` (T, 2) - gripper position for both fingers
- **Images**: 
  - `robot0_agentview_center_image` (T, 128, 128, 3) - third-person camera view
  - `robot0_eye_in_hand_image` (T, 128, 128, 3) - first-person camera view from gripper

-------
## Exploring the Data

Replay and visualize demonstrations using the `replay_dataset.py` script:

```bash
python robocasa/scripts/replay_dataset.py \
  --hdf5_path ReMemBench-Dataset/MemHeatPot/2025-07-24-22-26-20/demo_im128.hdf5 \
  --episode_idx 0 \
  --render
```

Common options: `--max_episodes`, `--replay_state`. See `--help` for full usage.

-------
## Converting to LeRobot Dataset Format

We did not use [lerobot](https://github.com/huggingface/lerobot) dataset format. However, we provide the conversion script for convenience.

Convert an existing dataset (with image keys) to LeRobot format:

```bash
python robocasa/scripts/port_to_lerobot.py \
  --dataset_path ReMemBench-Dataset/MemHeatPot/2025-07-24-22-26-20/demo_im128.hdf5 \
  --repo_name your_hf_username/MemHeatPot
```

Convert multiple datasets into one dataset:

```bash
python robocasa/scripts/port_to_lerobot.py \
  --dataset_path \
    ReMemBench-Dataset/MemHeatPot/2025-07-24-22-26-20/demo_im128.hdf5 \
    ReMemBench-Dataset/MemHeatPot/2025-07-24-22-30-15/demo_im128.hdf5 \
    ReMemBench-Dataset/MemHeatPot/2025-07-24-22-35-10/demo_im128.hdf5 \
  --repo_name your_hf_username/MemHeatPot
```

Or use a glob pattern to find all demo_im128.hdf5 files:

```bash
python robocasa/scripts/port_to_lerobot.py \
  --dataset_path $(find ReMemBench-Dataset/MemHeatPot -name "demo_im128.hdf5") \
  --repo_name your_hf_username/MemHeatPot
```

To push to Hugging Face Hub, add `--push_to_hub`. See `--help` for all options.

-------
## Citation
```bibtex
TODO: Add arXiv citation
```
