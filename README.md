# Robot-Vision
Computer Vision tools and pipelines for spacecraft robotics research.

This repository contains data generation, training, and inference code used to build and evaluate 6-DoF pose estimators and related tooling for robotic perception experiments.

## Contents
- **Overview:** 
- **Repository structure:** 
- **Quickstart:** 

## Overview
- Scripts and utilities to generate synthetic datasets (Blender-based).
	- Supports multiple objects(Might need few adjustments [FUTURE WORK]) OR multiple classes within the same object
	- Currently fully supports multiple classes within the same object. 
	- Object used in this case has the following classes: FaceA, FaceB, FaceC, FaceD
	- Outputs data in COCO format along with the rendered images
	- Supports YOLO format output too (needs a little adjustment [FUTURE WORK])
- Training and evaluation pipelines for pose estimation (multiple formats supported).
- Inference utilities to run models on images/videos and compute pose errors.
- Calculates Pose using PnP (Iterative and IPPE)

## Repository structure
- **data generator/**: Blender-based dataset generation and formatting utilities (e.g., `runinblender.py`, `sdgdata.py`, `bbox.py`). Use these to synthesize images, bounding boxes, keypoints, and transformation matrices.
- **estimator/**: Inference and model-related code. Key files:
	- [estimator/inference.py](estimator/inference.py) — single-class inference pipeline.
	- [estimator/inferencemulticlass.py](estimator/inferencemulticlass.py) — multiclass inference.
	- [estimator/model/] — model assets and coordinate formats.
- **solver/**: Pose estimation algorithms, pose metrics, and evaluation utilities (e.g., `pose.py`, `posevideo.py`, `error_metrics.py`).
- **TRAIN/**: Training helper scripts and notebooks. See `TRAIN/training/train.py` and the notebooks for examples.
- **local/**: Local development helpers, Blender-specific packages, and debugging scripts used during dataset generation and experimentation.
- **coordinatesUtils/**: Coordinate transform helpers and utilities for building / parsing transformation matrices.
- **test_dataset/**: Example dataset layouts and pretrained weights for quick testing.
- **weights/**: Trained model checkpoints organized by dataset/format.

## Quickstart
Prerequisites: Python 3.8+, common packages such as `numpy`, `opencv-python`, and a deep learning framework (PyTorch) for model code. Install required packages as appropriate for your setup.

Run inference (example):

```bash
python3 estimator/solver/inference.py
```

Solve Pose and Visualize (example):

```bash
python3 estimator/solver/pose.py
```

Generate synthetic data (Blender):

```bash
blender /path/to/blenderFile --background --python /data generator/runinblender.py
```

<!-- Train a model (example):

```bash
needs update
``` -->

Notes: each script contains its own CLI and usage notes. Inspect the top of the script files for specific flags and configuration options.

## Evaluation and logs
- Pose error CSVs (examples): `pose_errors.csv`

## License
See the `LICENSE` file in the repository root.
