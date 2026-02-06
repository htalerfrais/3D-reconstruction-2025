# 3D Reconstruction from Video Sequence

Academic project for the **Computer Vision** course (S9) at **ENSEIRB-MATMECA** (2024-2025).

This project implements an incremental Structure from Motion (SfM) pipeline to reconstruct a 3D scene from a video sequence using Bundle Adjustment with the Levenberg-Marquardt algorithm.

## Our Contributions

### Main Reconstruction Pipeline (`main_reconstruction.py`)

We developed the complete incremental SfM pipeline from scratch, implementing:

**1. Initialization Phase**
- Essential matrix estimation using RANSAC + five-point algorithm
- Relative pose recovery with chirality validation
- Initial 3D point triangulation between the first two views
- Track structure creation for multi-view correspondences

**2. Incremental Camera Localization**
- Robust camera pose estimation for each new image using existing 3D-2D correspondences
- Pose initialization from previous camera position
- Positive depth validation before optimization

**3. Intelligent Triangulation**
- New 3D point detection from orphan features
- Parallax angle computation between camera pairs
- Minimum parallax threshold (5°) to ensure geometric stability
- Optimal camera pair selection for each new 3D point

**4. Track Management**
- Dynamic track structure updates as new points are triangulated
- Global mapping between 3D point keys and indices
- Multi-view consistency maintenance

**5. Pipeline Orchestration**
- Sequential processing of all video frames
- Visualization of reprojection errors at each step
- Global Bundle Adjustment integration after each new view
- Scale normalization to maintain reconstruction consistency

### Camera Localization Module (`BA_LM_localization.py`)

We derived a specialized Bundle Adjustment variant from the provided `BA_LM_schur.py`:
- Optimizes **only camera pose** (rotation + translation) while keeping 3D points fixed
- Essential for the incremental reconstruction workflow
- Enables fast and stable camera localization using PnP refinement

## Results

The pipeline successfully reconstructs the 3D scene with:
- Mean reprojection error typically **< 1 pixel**
- Robust camera trajectory estimation across 100+ frames
- Dense point cloud with multi-view consistency

## Project Structure

```
├── main_reconstruction.py      # [OUR CODE] Main SfM pipeline
├── BA_LM_localization.py       # [OUR CODE] Camera-only Bundle Adjustment
├── BA_LM_schur.py              # [PROVIDED] Full Bundle Adjustment
├── BA_LM_two_views_schur.py    # [PROVIDED] Two-view Bundle Adjustment
├── utils.py                    # [PROVIDED] Utility functions
├── viewer.py                   # [PROVIDED] 3D visualization (Open3D)
├── main_show_final_reconstruction.py  # [PROVIDED] Result visualization
├── images/                     # Input image sequence
├── data_ready.pkl              # Preprocessed input data
├── final_reconstruction.pkl    # Output reconstruction
└── report.pdf                  # Detailed project report
```

## Installation

### Prerequisites
- Python 3.11+
- Conda (recommended)

### Setup

```bash
git clone https://github.com/yourusername/3D-reconstruction-2025.git
cd 3D-reconstruction-2025

conda env create -f environment.yml
conda activate spyder_env
```

### Key Dependencies

- NumPy
- SciPy
- OpenCV (`opencv`)
- Open3D (for visualization)
- Matplotlib
- Pillow

## Usage

### Run the Reconstruction

```bash
python main_reconstruction.py
```

This will:
1. Load the preprocessed data (`data_ready.pkl`)
2. Process all images incrementally
3. Save the final reconstruction to `final_reconstruction.pkl`

## Data

The project uses images from a video sequence. The input data (`data_ready.pkl`) contains:
- 2D keypoint detections for each image
- Camera calibration matrix (K)
- Feature tracks across images
- Image filenames

## Results

The pipeline successfully reconstructs the 3D scene with:
- Mean reprojection error typically < 1 pixel
- Robust camera trajectory estimation
- Dense point cloud with multi-view consistency

## View 3D points
In order to visualise the final_reconstruction.pkl you will have to install Open3D dependencies which are not part of the provided conda environment.

## Authors

- Hector Taler-Fraisse and Maxime Hurtubise, ENSEIRB-MATMECA 2025

## Acknowledgments

- Course instructor Guillaume Bourmaud for providing the base code and guidance
- ENSEIRB-MATMECA for the Computer Vision course

