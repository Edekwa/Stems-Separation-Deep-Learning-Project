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
Projet/
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
│   ├── utils
