# Music Source Separation using Deep Learning

## 1. Project Overview

This project investigates deep learning approaches for music source separation using the MUSDB18-HQ dataset.

The objective is to separate a musical mixture into four independent audio stems:

- **Vocals** — singing voice
- **Bass** — bass instruments
- **Drums** — percussion
- **Other** — remaining instruments

The project aims to implement, train and compare different deep learning architectures, evaluating their ability to reconstruct individual audio sources.

### Models under investigation

- U-Net (spectrogram-based separation)
- BLSTM (recurrent neural network)
- Conv-TasNet (optional, depending on computational resources)

### Evaluation metrics

- SDR (Signal-to-Distortion Ratio)
- SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
- SIR (Signal-to-Interference Ratio)
- SAR (Signal-to-Artifacts Ratio)
- Spectral MSE
- Inference time

---

## 2. Project Structure

```text
music-source-separation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── main.ipynb
│   ├── exploration.ipynb
│   └── experiments.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py
│   │   ├── blstm.py
│   │   └── conv_tasnet.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── losses.py
│   │   └── optimizer.py
│   │
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       ├── evaluate.py
│       └── visualization.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── checkpoints/
│
└── results/
    ├── metrics/
    ├── figures/
    └── audio/
```

## 3. Installation

### Requirements

- Python 3.10 or later
- Jupyter Notebook
- PyTorch
- MUSDB18-HQ dataset

### Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user \
    --name=music-separation \
    --display-name="Python (Music Separation)"
```

## 4. Dataset

The project uses the MUSDB18-HQ dataset.

The dataset contains musical tracks with their corresponding reference stems.

Expected directory structure:

```text
data/
├── raw/
│   └── musdb18hq/
│       ├── train/
│       └── test/
│
└── processed/
```

The original dataset is not included in this repository.

Preprocessed data will be generated locally and stored in `data/processed/`.

## 5. Workflow

The project follows the following pipeline:

1. **Dataset exploration:** inspect audio tracks, stems and dataset characteristics.
2. **Preprocessing:** resampling, segmentation, normalization and time-frequency transformations.
3. **Dataset preparation:** create PyTorch datasets and data loaders.
4. **Model implementation:** develop the different neural network architectures.
5. **Training:** train models using consistent training and validation procedures.
6. **Hyperparameter optimization:** investigate the influence of selected hyperparameters.
7. **Evaluation:** compare models using objective audio separation metrics.
8. **Reconstruction:** generate separated audio stems and perform listening tests.

## 6. Notebook Usage

`notebooks/main.ipynb` is the main entry point for the project.

It orchestrates the complete experimental workflow by importing functions and classes from `src/`.

Other notebooks are intended for exploratory analysis and isolated experiments.

The implementation of reusable functions should remain in Python scripts rather than being duplicated across notebooks.

## 7. Experiment Management

Each experiment should record:

- Model architecture
- Hyperparameter configuration
- Training and validation losses
- Evaluation metrics
- Training duration
- Model checkpoint

Trained weights are stored in `checkpoints/`, while evaluation results and visualizations are stored in `results/`.

## 8. Version Control

The following elements should not be committed to Git:

- Raw and processed audio datasets
- Virtual environments
- Model checkpoints
- Temporary files
- Large generated audio files

The source code, notebooks, documentation and experiment summaries should remain version-controlled.

---

**Project:** Music Source Separation using Deep Learning  
**Dataset:** MUSDB18-HQ  
**Framework:** PyTorch  
**Language:** Python
