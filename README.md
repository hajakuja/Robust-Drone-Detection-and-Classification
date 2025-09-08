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
- To actually run with the LibreSDR you need the firmware and bootloader located in [Gargoyle's Drive](!https://drive.google.com/drive/u/0/folders/1COamB_a2Bg5ggDWPynG69MCTEX21ZvTo)
- The pre-trained weights are also located in [Gargoyle's Drive](!https://drive.google.com/file/d/1yXM_R6BPF1eZlcainm6QYwOgNXnVasXs/view?usp=drive_link)
- Also pre-recorded I/Q samples are located [here](!https://drive.google.com/drive/u/0/folders/1QO-uxldw1FwpNfhTmhcEP1NpHBNKlaJc)

### Usage

#### `run_inference.py`
Run inference on IQ samples from a file or SDR device. Supply the model weights and choose `--source` as `file` or `sdr`.

**Arguments:**
- `--weights PATH` – path to model weights (.pth) (required)
- `--source {sdr,file}` – input source (required)
- `--file PATH` – IQ data file when `--source` is `file`
- `--num_samps N` – number of IQ samples to capture (default `1024*1024`)
- `--chunk_size N` – samples per inference chunk (default `1024*1024`)
- `--continuous` – continuously read from the SDR
- `--rate`, `--freq`, `--gain`, `--antenna` – SDR configuration
- `--device DEVICE` – computation device (e.g., `cpu`, `cuda`)
- `--num_classes N` – override number of output classes
- `--class-names LIST/FILE` – comma-separated list or text file of class names

#### `run_inference_tui.py`
Provides a curses-based text interface that prompts for model and radio settings and continuously displays detection probabilities alongside an FFT graph.

**Arguments:**
- `--models-dir DIR` – directory containing `.pth` files (default `.`)
- `--device DEVICE` – computation device (default `cpu`)

#### `run_inference_qt.py`
Starts a PyQt5 GUI with a live spectrogram and prediction readout. Works with SDR streams or IQ files.

**Arguments:**
- `--weights PATH` – path to model weights (.pth) (required)
- `--source {sdr,file}` – input source (required)
- `--file PATH` – IQ data file when `--source` is `file`
- `--num_samps N` – number of IQ samples to capture (default `1024*1024`)
- `--chunk_size N` – IQ samples per chunk (default `1024*1024`)
- `--continuous` – continuously read from the SDR
- `--rate`, `--freq`, `--gain`, `--antenna` – SDR configuration
- `--device DEVICE` – computation device
- `--spectrogram-size WIDTH HEIGHT` – spectrogram image size in pixels (default `800 800`)

