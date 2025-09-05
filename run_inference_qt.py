import argparse
import os
import csv
import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets

from lib import model_VGG2D
from load_dataset import transform_spectrogram
import run_inference

# Allow torch to load models saved with these classes
# (mirrors run_inference.py)
torch.serialization.add_safe_globals([model_VGG2D.VGG])
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
torch.serialization.add_safe_globals([torch.nn.modules.conv.Conv2d])
torch.serialization.add_safe_globals([torch.nn.modules.batchnorm.BatchNorm2d])
torch.serialization.add_safe_globals([torch.nn.modules.activation.ReLU])
torch.serialization.add_safe_globals([torch.nn.modules.pooling.MaxPool2d])
torch.serialization.add_safe_globals([torch.nn.modules.pooling.AdaptiveAvgPool2d])
torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])
torch.serialization.add_safe_globals([torch.nn.modules.dropout.Dropout])


class InferenceThread(QtCore.QThread):
    """Worker thread that yields spectrograms and predictions."""

    new_result = QtCore.pyqtSignal(np.ndarray, str)

    def __init__(self, iq_iter, model, transform, class_names, device):
        super().__init__()
        self.iq_iter = iq_iter
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.device = device
        self._stop = False

    def run(self) -> None:
        try:
            for iq in self.iq_iter:
                if self._stop:
                    break
                iq = iq.to(self.device)
                spec = self.transform(iq).unsqueeze(0)  # (1, 2, 1024, 1024)
                with torch.no_grad():
                    logits = self.model(spec)
                    pred_idx = logits.argmax(1).item()
                    pred_label = (
                        self.class_names[pred_idx]
                        if self.class_names is not None and pred_idx < len(self.class_names)
                        else str(pred_idx)
                    )
                img = spec[0].pow(2).sum(0).sqrt().cpu().numpy()
                self.new_result.emit(img, pred_label)
        except Exception:
            pass  # Suppress exceptions on shutdown

    def stop(self) -> None:
        self._stop = True
        # Close the IQ iterator if it supports it to release resources
        close = getattr(self.iq_iter, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class SpectrogramWindow(QtWidgets.QWidget):
    """Main application window displaying the spectrogram."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Spectrogram")
        self.resize(600, 600)
        layout = QtWidgets.QVBoxLayout(self)
        self.image_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_label)
        self.pred_label = QtWidgets.QLabel("Prediction: ")
        layout.addWidget(self.pred_label)

    @QtCore.pyqtSlot(np.ndarray, str)
    def update_display(self, img: np.ndarray, label: str) -> None:
        # Normalise and convert to 8-bit grayscale
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255).astype(np.uint8)
        h, w = img.shape
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.image_label.width(),
            self.image_label.height(),
            QtCore.Qt.KeepAspectRatio,
        )
        self.image_label.setPixmap(pix)
        self.pred_label.setText(f"Prediction: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model inference with a live Qt spectrogram")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--source", choices=["sdr", "file"], required=True, help="Input source")
    parser.add_argument("--file", help="Path to IQ file (.npy, .pt or .c16)")
    parser.add_argument("--num_samps", type=int, default=1024 * 1024, help="Number of IQ samples to capture")
    parser.add_argument("--chunk_size", type=int, default=1024 * 1024, help="IQ samples per chunk")
    parser.add_argument("--rate", type=float, default=14e6, help="SDR sample rate")
    parser.add_argument("--freq", type=float, default=2.4e9, help="SDR center frequency")
    parser.add_argument("--gain", type=float, default=0.0, help="SDR gain")
    parser.add_argument("--antenna", default="TX/RX", help="SDR antenna selection")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously read from the SDR in chunks of --chunk_size",
    )
    parser.add_argument("--device", default="cpu", help="Computation device")
    args = parser.parse_args()

    state = torch.load(args.weights, map_location=args.device)
    class_names = state.get("class_names") if isinstance(state, dict) else None

    if class_names is None:
        for path in ("class_stats.csv", os.path.join("data", "class_stats.csv")):
            if os.path.exists(path):
                with open(path, newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    try:
                        idx = header.index("class")
                    except ValueError:
                        idx = 1 if len(header) > 1 else 0
                    class_names = [row[idx] for row in reader if len(row) > idx]
                break

    state_dict = state.state_dict() if isinstance(state, torch.nn.Module) else state
    if "classifier.3.weight" in state_dict:
        num_classes = state_dict["classifier.3.weight"].shape[0]
    elif "module.classifier.3.weight" in state_dict:
        num_classes = state_dict["module.classifier.3.weight"].shape[0]
    else:
        raise KeyError("Could not infer number of classes from checkpoint")

    model = model_VGG2D.vgg11_bn(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    transform = transform_spectrogram(
        device=args.device,
        n_fft=1024,
        win_length=1024,
        hop_length=1024,
        window_fn=torch.hann_window,
        power=None,
        normalized=False,
        center=False,
        onesided=False,
    )

    if args.source == "file":
        if not args.file:
            raise ValueError("--file path required when source is 'file'")
        iq_iter = run_inference.iq_chunks_from_file(args.file, args.chunk_size)
    else:
        if args.continuous:
            iq_iter = run_inference.iq_chunks_from_sdr(
                args.chunk_size, args.rate, args.freq, args.gain, args.antenna
            )
        else:
            iq_full = run_inference.capture_from_sdr(
                args.num_samps, args.rate, args.freq, args.gain, args.antenna
            )
            iq_iter = [iq_full]

    app = QtWidgets.QApplication([])
    window = SpectrogramWindow()
    worker = InferenceThread(iq_iter, model, transform, class_names, args.device)
    worker.new_result.connect(window.update_display)
    worker.start()
    window.show()
    app.exec_()
    worker.stop()
    worker.wait()


if __name__ == "__main__":
    main()
