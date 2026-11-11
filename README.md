# Music Source Separation using Deep Learning

Deep learning-based separation of musical recordings into four individual audio stems using the **MUSDB18-HQ** dataset and **PyTorch**.

This project investigates and compares different neural network architectures for music source separation, focusing on separation quality, computational efficiency, and the influence of model hyperparameters.

## 1. Project Overview

Music source separation consists of decomposing a musical mixture into its constituent sources.

Given a stereo audio mixture \(x(t)\), the objective is to estimate four individual signals:

- **Vocals** — singing voice
- **Bass** — bass instruments
- **Drums** — percussion
- **Other** — remaining instruments

The mixture can be expressed as:

\[
x(t) = s_{\mathrm{vocals}}(t) + s_{\mathrm{bass}}(t) + s_{\mathrm{drums}}(t) + s_{\mathrm{other}}(t)
\]

The objective is to train neural networks that estimate each source from the original mixture while minimizing reconstruction errors and interference between sources.

### Objectives

1. Develop a complete audio preprocessing and training pipeline.
2. Implement and compare multiple deep learning architectures.
3. Investigate the effect of selected hyperparameters on separation performance.
4. Evaluate the models using established audio separation metrics.
5. Analyze the trade-offs between separation quality, training cost, and inference efficiency.

### Models under investigation

| Architecture | Domain | Description |
|---|---|---|
| U-Net | Time-frequency | Encoder-decoder CNN for spectrogram-based source separation |
| BLSTM | Time-frequency | Bidirectional recurrent network for modeling temporal dependencies |
| Conv-TasNet | Time domain | Convolutional architecture operating directly on audio waveforms |

**U-Net** will serve as the initial baseline. BLSTM and Conv-TasNet will be investigated subsequently, depending on available computational resources.

### Evaluation metrics

The project considers six evaluation metrics:

| Metric | Description |
|---|---|
| SDR | Signal-to-Distortion Ratio |
| SI-SDR | Scale-Invariant Signal-to-Distortion Ratio |
| SIR | Signal-to-Interference Ratio |
| SAR | Signal-to-Artifacts Ratio |
| Spectral MSE | Mean Squared Error between reference and estimated spectral representations |
| Inference Time | Computational time required to separate an audio segment |

The exact metric implementations and evaluation protocols will be documented to ensure fair comparisons.

---

## 2. Development Environment

The primary development environment is a Windows PC equipped with an NVIDIA GPU.

### Hardware

- **Operating system:** Windows 10/11, 64-bit
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU
- **GPU acceleration:** NVIDIA CUDA
- **CPU/RAM:** Depending on the development machine

GPU acceleration is used for model training and inference whenever available.

### Software

| Component | Technology |
|---|---|
| Programming language | Python 3.12 |
| Deep learning framework | PyTorch |
| Audio processing | TorchAudio, Librosa, SoundFile |
| IDE | Visual Studio Code |
| Interactive development | Jupyter |
| Version control | Git and GitHub |
| Code formatting and linting | Ruff |
| Experiment tracking | TensorBoard |
| Testing | Pytest |
| AI coding assistance | OpenAI Codex (optional) |

### VS Code extensions

Recommended extensions:

- Python — Microsoft
- Pylance — Microsoft
- Python Debugger — Microsoft
- Python Environments — Microsoft
- Jupyter — Microsoft
- Ruff — Astral
- Codex — OpenAI (optional)
- GitLens — GitKraken (optional)
- GitHub Pull Requests — GitHub (optional)

For interface customization, the **Catppuccin Mocha** theme and Catppuccin Icons are recommended but not required.

---

## 3. Project Structure

```text
Stems-Separation-Deep-Learning-Project/
│
├── README.md
├── requirements.txt
├── requirements-cuda.txt
├── requirements-lock.txt         # Optional environment snapshot
├── pyproject.toml                # Tool configuration (planned)
├── .gitignore
│
├── .venv/                        # Local virtual environment (ignored)
│
├── .vscode/                      # Local VS Code settings (optional)
│
├── Ressources/                   # Project resources and documentation
│
├── configs/
│   └── unet_baseline.yaml        # Baseline experiment configuration
│
├── notebooks/
│   ├── main.ipynb                # Main experimental workflow
│   ├── exploration.ipynb         # Dataset exploration
│   └── experiments.ipynb         # Model experiments and comparisons
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── utils.py                  # Shared utilities
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # PyTorch Dataset and DataLoader
│   │   └── preprocessing.py      # Audio preprocessing
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
├── scripts/
│   └── check_environment.py      # Environment and CUDA verification
│
├── tests/                        # Automated tests
│
├── data/
│   ├── raw/
│   │   └── musdb18hq/
│   │       ├── train/
│   │       └── test/
│   │
│   └── processed/
│
├── checkpoints/                  # Trained model weights (ignored)
│
└── results/
    ├── metrics/
    ├── figures/
    └── audio/
```

Some files and directories represent the intended architecture and will be implemented progressively.

### Organization principles

The project separates reusable code, experimental configurations, interactive notebooks, and generated outputs.

- **`src/`** contains reusable Python implementations.
- **`notebooks/`** contains interactive experiments and data analysis.
- **`configs/`** stores reproducible experiment configurations.
- **`scripts/`** contains executable utility scripts.
- **`tests/`** contains automated verification tests.
- **`data/`** contains the original and preprocessed datasets.
- **`checkpoints/`** stores trained neural network weights.
- **`results/`** stores evaluation metrics, visualizations, and separated audio.

Large datasets, model weights, and generated audio files are excluded from Git.

---

## 4. Installation

### 4.1 Prerequisites

Before installing the project, ensure that the following software is available:

- Python 3.12
- Git
- Visual Studio Code (recommended)
- Compatible NVIDIA GPU driver (for GPU training)

The complete CUDA Toolkit is not required for standard PyTorch installations using precompiled CUDA-enabled packages.

### 4.2 Clone the repository

Open PowerShell:

```powershell
mkdir C:\Projects -Force
cd C:\Projects

git clone https://github.com/Edekwa/Stems-Separation-Deep-Learning-Project.git

cd Stems-Separation-Deep-Learning-Project
```

Open the project in VS Code:

```powershell
code .
```

### 4.3 Create a virtual environment

The project uses a dedicated Python virtual environment to isolate dependencies.

On Windows:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, local scripts can be permitted for the current user with:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Alternatively, use the virtual environment's Python executable directly without activation.

Verify the Python version:

```powershell
python --version
```

Expected: `Python 3.12.x`.

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

**Linux/macOS alternative:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 4.4 Install scientific dependencies

The project separates general Python dependencies from CUDA-specific PyTorch packages.

Install the general dependencies:

```powershell
python -m pip install -r requirements.txt
```

These include scientific computing, audio processing, evaluation, visualization, notebooks, and development tools.

### 4.5 Install PyTorch with NVIDIA CUDA

On a compatible NVIDIA GPU system, install the CUDA-enabled PyTorch and TorchAudio builds using:

```powershell
python -m pip install -r requirements-cuda.txt
```

The CUDA requirements file is configured to use the official PyTorch CUDA 12.8 package index.

**Important:** The NVIDIA driver must support the selected CUDA runtime. Check driver compatibility before installation.

For computers without an NVIDIA GPU, install the appropriate PyTorch distribution using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 4.6 Verify the installation

Check installed package consistency:

```powershell
python -m pip check
```

Check CUDA availability:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Display GPU information:

```powershell
nvidia-smi
```

Verify the detected GPU:

```python
import torch

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print(
        "VRAM (GB):",
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
    )
else:
    print("CUDA is not available.")
```

The environment is ready for GPU training when PyTorch correctly detects the NVIDIA GPU and successfully executes CUDA operations.

If available, the project environment verification script can also be executed:

```powershell
python scripts/check_environment.py
```

### 4.7 Configure Jupyter

Register the project's Python environment as a Jupyter kernel:

```powershell
python -m ipykernel install --user --name music-separation --display-name "Python (Music Separation)"
```

In VS Code:

1. Open a notebook.
2. Select **Kernel** in the upper-right corner.
3. Choose **Python (Music Separation)**.

This ensures that notebooks use the same libraries as the project's Python scripts.

### 4.8 Reproducing the environment

The dependency files serve different purposes:

| File | Purpose |
|---|---|
| `requirements.txt` | General scientific and development dependencies |
| `requirements-cuda.txt` | CUDA-enabled PyTorch and TorchAudio |
| `requirements-lock.txt` | Optional snapshot of exact installed package versions |

After validating the environment, an exact package snapshot can be generated:

```powershell
python -m pip freeze > requirements-lock.txt
```

Such a snapshot is platform-specific and should not be assumed to work unchanged across Windows, Linux, and macOS.

---

## 5. Dataset

### MUSDB18-HQ

The project uses the **MUSDB18-HQ** dataset, a benchmark for supervised music source separation.

The dataset contains:

- 150 stereo musical recordings
- 100 training tracks
- 50 test tracks
- Four reference stems per track
- Stereo WAV audio at 44.1 kHz

Each track contains the following files:

```text
track_name/
├── mixture.wav
├── vocals.wav
├── drums.wav
├── bass.wav
└── other.wav
```

### Dataset directory

The dataset should be extracted into:

```text
data/
└── raw/
    └── musdb18hq/
        ├── train/
        └── test/
```

The original dataset is not included in this repository.

The official dataset is available through the [SigSep MUSDB repository](https://github.com/sigsep/sigsep-mus-db).

### Training and validation strategy

The original MUSDB18-HQ training partition will be divided into training and validation subsets.

The official test partition will be reserved for final evaluation.

**Important:** Dataset splitting must be performed at the *track level*, not at the individual audio-segment level, to avoid data leakage between training and validation.

The precise split and random seed will be documented in the experiment configurations.

### Preprocessing

The preprocessing pipeline may include:

1. Loading stereo audio tracks and their reference stems.
2. Extracting fixed-duration audio segments.
3. Applying consistent normalization.
4. Computing STFT representations for spectrogram-based models.
5. Creating input-target pairs.
6. Preparing batches for training.

The original 44.1 kHz sample rate will be preserved initially for spectrogram-based models. Alternative sample rates may be investigated to reduce computational requirements.

Time-domain architectures may use a different preprocessing pipeline.

---

## 6. Modeling Approach

### 6.1 U-Net

U-Net will serve as the first baseline architecture.

The model processes time-frequency representations of the mixture through an encoder-decoder architecture with skip connections.

The initial implementation will investigate spectrogram-based separation, potentially using predicted masks applied to the mixture representation.

The separated waveforms can then be reconstructed using an inverse STFT.

### 6.2 BLSTM

The BLSTM model will investigate the benefits of bidirectional temporal modeling.

By processing sequential audio representations in both directions, the network can exploit temporal dependencies across frames.

### 6.3 Conv-TasNet

Conv-TasNet is considered as an optional time-domain architecture.

Unlike spectrogram-based approaches, Conv-TasNet operates directly on learned audio representations.

Its inclusion will depend on available GPU memory, computational resources, and project progress.

---

## 7. Experimental Methodology

### 7.1 Hyperparameter investigation

The project will investigate six hyperparameter families:

| Hyperparameter | Examples |
|---|---|
| Architecture | U-Net, BLSTM, Conv-TasNet |
| Model capacity | Network depth, channel counts, hidden dimensions |
| Learning rate | Different initial learning rates |
| Optimizer | Adam, AdamW |
| Loss function | L1, spectral losses, SI-SDR-based objectives |
| Audio representation | STFT window size, hop length, or segment duration |

Not all combinations will necessarily be evaluated. A controlled experimental plan will be established according to the available training budget.

### 7.2 Training protocol

Models will be trained under a consistent experimental protocol whenever possible.

The training procedure includes:

1. Loading the experiment configuration.
2. Initializing the model.
3. Preparing training and validation DataLoaders.
4. Computing predictions and loss values.
5. Updating model parameters.
6. Monitoring validation performance.
7. Saving relevant model checkpoints.
8. Recording experiment metadata and metrics.

GPU acceleration and efficient data loading will be used when available.

### 7.3 Experiment configurations

Experiment parameters are stored in YAML files under `configs/`.

For example:

```text
configs/
└── unet_baseline.yaml
```

Each configuration will document relevant parameters such as sample rate, segment duration, architecture settings, batch size, optimizer, learning rate, number of epochs, and random seed.

### 7.4 Experiment tracking

Training progress is monitored with TensorBoard.

To launch TensorBoard when training logs are available:

```powershell
tensorboard --logdir results/logs
```

Open the dashboard at:

`http://localhost:6006`

Each experiment should record:

- Architecture and model configuration
- Random seed
- Training and validation losses
- Evaluation metrics
- Training duration
- Inference time
- Best model checkpoint
- Relevant implementation or configuration changes

The output directory and log naming convention will be implemented alongside the training pipeline.

---

## 8. Evaluation

Each model will be evaluated on the same held-out test dataset using consistent preprocessing and reconstruction procedures.

The evaluation includes:

**Separation performance**

- SDR
- SI-SDR
- SIR
- SAR

**Spectral reconstruction**

- Spectral MSE

**Computational efficiency**

- Inference time

Where supported by the chosen metric implementation, scores will be aggregated across tracks and sources.

Results will be reported separately for vocals, bass, drums, and other.

### Visual analysis

The evaluation pipeline will generate:

- Training and validation loss curves
- Spectrogram comparisons
- Model performance comparisons
- Metric distributions
- Reconstructed audio examples

### Listening evaluation

Separated audio samples will also be used for qualitative listening tests.

These tests complement numerical metrics by helping identify perceptual artifacts, source leakage, and reconstruction issues.

---

## 9. Notebook Usage

The repository contains three main notebooks.

### `main.ipynb`

Main entry point for the project.

Intended to demonstrate the complete experimental workflow by importing reusable components from `src/`.

### `exploration.ipynb`

Used for exploratory data analysis, including:

- Waveform visualization
- Spectrogram analysis
- Track and stem inspection
- Audio statistics
- Preprocessing verification

### `experiments.ipynb`

Used for:

- Comparing model configurations
- Visualizing training histories
- Analyzing evaluation metrics
- Comparing separated audio and reference stems

**Development principle:** Reusable functionality should be implemented in Python modules under `src/`, rather than duplicated inside notebooks.

Long training jobs should be executable independently of interactive notebook sessions.

---

## 10. Testing and Code Quality

The project uses Ruff for Python formatting and linting, and Pytest for automated testing.

Format source files:

```powershell
ruff format src tests
```

Check Python code:

```powershell
ruff check src tests
```

Run tests:

```powershell
python -m pytest tests
```

These commands become useful as the corresponding modules and tests are implemented.

The `pyproject.toml` file will centralize development tool configurations.

---

## 11. Version Control

Git and GitHub are used to track source code, documentation, and experiment configurations.

### Version-controlled files

- Python source files
- Jupyter notebooks
- Experiment configurations
- Dependency definitions
- Documentation
- Small, reproducible experiment summaries when appropriate

### Excluded files

- Virtual environments
- Raw and processed audio datasets
- Model checkpoints and weights
- Large generated audio files
- Temporary files and caches
- Local environment variables and secrets
- Large TensorBoard logs

The `.gitignore` file defines the excluded paths.

### Recommended workflow

Before committing changes:

```powershell
git status
git add .
git diff --cached --stat
```

After verifying the staged files:

```powershell
git commit -m "Describe the changes"
git push
```

Dataset files and trained model weights must not be committed directly to Git.

---

## 12. Project Roadmap

- [x] Define the problem and dataset
- [x] Select candidate architectures and evaluation metrics
- [x] Create the GitHub repository
- [x] Configure the Python development environment
- [x] Install scientific and GPU dependencies
- [ ] Implement dataset exploration
- [ ] Implement the preprocessing pipeline
- [ ] Create PyTorch Dataset and DataLoader classes
- [ ] Implement the U-Net baseline
- [ ] Train and evaluate the baseline
- [ ] Investigate hyperparameter configurations
- [ ] Implement and compare the BLSTM model
- [ ] Investigate Conv-TasNet if resources permit
- [ ] Perform final quantitative and qualitative evaluation
- [ ] Produce the final report and presentation

---

## 13. References

- [MUSDB18 / MUSDB18-HQ — SigSep](https://github.com/sigsep/sigsep-mus-db)
- [PyTorch Documentation](https://docs.pytorch.org/)
- [TorchAudio Documentation](https://docs.pytorch.org/audio/)
- [Librosa Documentation](https://librosa.org/doc/latest/)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Visual Studio Code Python Documentation](https://code.visualstudio.com/docs/languages/python)

---

**Project:** Music Source Separation using Deep Learning  
**Dataset:** MUSDB18-HQ  
**Framework:** PyTorch  
**Language:** Python 3.12  
**Primary platform:** Windows / NVIDIA CUDA  
**Repository:** https://github.com/Edekwa/Stems-Separation-Deep-Learning-Project
