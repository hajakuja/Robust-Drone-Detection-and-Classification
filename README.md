# Robust Drone Detection and Classification Using Convolutional Neural Networks in Low SNR Environments
This repository was cloned from [the original authors!](!https://github.com/sgluege/Robust-Drone-Detection-and-Classification)

## Inference for drone detection
 The following scripts were added for inference:
  - run_inference.py
  - run_inference_tui.py
  - run_inference_qt.py

### Requirements
- [PyTorch](https://pytorch.org/) is required for loading the trained model and running inference.
- [UHD](https://github.com/EttusResearch/uhd) drivers are needed to interface with USRP SDR hardware.

### Usage

#### `run_inference.py`
Run inference on IQ samples from a file or SDR device. Supply the model weights and choose `--source` as `file` or `sdr`.

#### `run_inference_tui.py`
Provides a curses-based text interface that prompts for model and radio settings and continuously displays detection probabilities alongside an FFT graph.

#### `run_inference_qt.py`
Starts a PyQt5 GUI with a live spectrogram and prediction readout. Works with SDR streams or IQ files.

