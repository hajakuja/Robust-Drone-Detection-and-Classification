import argparse
import os
import csv
import threading
import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from lib import model_VGG2D
from load_dataset import transform_spectrogram

torch.serialization.add_safe_globals([model_VGG2D.VGG])
torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
torch.serialization.add_safe_globals([torch.nn.modules.conv.Conv2d])
torch.serialization.add_safe_globals([torch.nn.modules.batchnorm.BatchNorm2d])
torch.serialization.add_safe_globals([torch.nn.modules.activation.ReLU])
torch.serialization.add_safe_globals([torch.nn.modules.pooling.MaxPool2d])
torch.serialization.add_safe_globals([torch.nn.modules.pooling.AdaptiveAvgPool2d])
torch.serialization.add_safe_globals([torch.nn.modules.linear.Linear])
torch.serialization.add_safe_globals([torch.nn.modules.dropout.Dropout])

def load_iq_from_file(path: str) -> torch.Tensor:
    """Load IQ samples from a ``.npy``, ``.pt`` or ``.c16`` file.

    The expected output format is a ``(2, N)`` float tensor where the first
    row contains the real part and the second row contains the imaginary part
    of the IQ signal.  Files that store complex values as a 1-D array are
    converted to this representation before being returned. ``.c16`` files are
    interpreted as interleaved int16 IQ samples produced by GNU Radio and are
    scaled to the range [-1, 1).
    """

    # Load numpy array or torch tensor from disk
    if path.endswith(".npy"):
        iq_np = np.load(path)
    elif path.endswith(".pt"):
        data = torch.load(path)
        if isinstance(data, dict) and "x_iq" in data:
            tensor = data["x_iq"]
            iq_np = tensor.cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
        else:
            iq_np = data if isinstance(data, np.ndarray) else np.asarray(data)
    elif path.endswith(".c16"):
        raw = np.fromfile(path, dtype=np.int16)
        if raw.size % 2:
            raise ValueError("IQ stream in .c16 file has odd length")
        iq_np = raw.astype(np.float32).reshape(-1, 2).T / 32768.0
    else:
        raise ValueError("Unsupported file format: expected .npy, .pt or .c16")

    # ``iq_np`` might be complex or have shape ``(N, 2)`` – normalise to ``(2, N)``
    if np.iscomplexobj(iq_np):
        iq_np = np.vstack((iq_np.real, iq_np.imag))
    elif iq_np.ndim == 2 and iq_np.shape[0] != 2 and iq_np.shape[1] == 2:
        iq_np = iq_np.T
    elif iq_np.ndim == 1:
        raise ValueError("IQ data must be complex or a (2, N) array")

    return torch.from_numpy(iq_np.astype(np.float32))


def iq_chunks_from_file(path: str, chunk_size: int):
    """Yield IQ samples from ``path`` in chunks of ``chunk_size`` samples.

    The returned tensors have shape ``(2, chunk_size)``. For ``.npy`` files the
    array is memory-mapped so only the processed chunk is loaded into RAM. When
    the final chunk is shorter than ``chunk_size`` it is zero-padded on the
    right so that every yielded tensor has the same length. ``.c16`` files are
    interpreted as interleaved int16 IQ samples and are scaled to [-1, 1).
    """

    if path.endswith(".npy"):
        iq_np = np.load(path, mmap_mode="r")
        if np.iscomplexobj(iq_np):
            total = iq_np.shape[0]
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                chunk = np.vstack((iq_np[start:end].real, iq_np[start:end].imag))
                if end - start < chunk_size:
                    pad = ((0, 0), (0, chunk_size - (end - start)))
                    chunk = np.pad(chunk, pad)
                yield torch.from_numpy(chunk.astype(np.float32))
        else:
            if iq_np.ndim == 2 and iq_np.shape[0] == 2:
                total = iq_np.shape[1]
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    chunk = iq_np[:, start:end]
                    if end - start < chunk_size:
                        pad = ((0, 0), (0, chunk_size - (end - start)))
                        chunk = np.pad(chunk, pad)
                    yield torch.from_numpy(chunk.astype(np.float32))
            elif iq_np.ndim == 2 and iq_np.shape[1] == 2:
                total = iq_np.shape[0]
                for start in range(0, total, chunk_size):
                    end = min(start + chunk_size, total)
                    chunk = iq_np[start:end].T
                    if end - start < chunk_size:
                        pad = ((0, 0), (0, chunk_size - (end - start)))
                        chunk = np.pad(chunk, pad)
                    yield torch.from_numpy(chunk.astype(np.float32))
            else:
                raise ValueError("Unsupported IQ array shape in npy file")
    elif path.endswith(".pt"):
        iq = load_iq_from_file(path)
        total = iq.shape[1]
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            chunk = iq[:, start:end]
            if end - start < chunk_size:
                chunk = F.pad(chunk, (0, chunk_size - (end - start)))
            yield chunk
    elif path.endswith(".c16"):
        raw = np.memmap(path, dtype=np.int16, mode="r")
        total = raw.size // 2
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            slice_ = raw[2 * start : 2 * end].astype(np.float32)
            chunk = slice_.reshape(-1, 2).T / 32768.0
            if end - start < chunk_size:
                pad = ((0, 0), (0, chunk_size - (end - start)))
                chunk = np.pad(chunk, pad)
            yield torch.from_numpy(chunk.astype(np.float32))
    else:
        raise ValueError("Unsupported file format: expected .npy, .pt or .c16")


def capture_from_sdr(
    num_samps: int, rate: float, freq: float, gain: float, antenna: str
) -> torch.Tensor:
    """Capture IQ samples from a UHD-compatible SDR.

    The antenna port is selected via ``antenna``. Previously the function
    issued a single ``recv`` call without first starting the stream. On some
    devices this resulted in zero samples being returned, which later caused
    spectrogram generation to fail. The revised implementation explicitly
    issues a stream command and keeps receiving until either the requested
    number of samples has been collected or the radio stops producing data.
    """

    import uhd

    # Configure the device
    usrp = uhd.usrp.MultiUSRP()
    usrp.set_rx_rate(rate)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq))
    usrp.set_rx_gain(gain)
    usrp.set_rx_antenna(antenna)

    # Create the RX stream
    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_stream = usrp.get_rx_stream(stream_args)
    md = uhd.types.RXMetadata()
    buff = np.empty((num_samps,), dtype=np.complex64)
    timeout = num_samps / rate + 0.1

    # Start streaming the requested number of samples
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = num_samps
    stream_cmd.stream_now = True
    rx_stream.issue_stream_cmd(stream_cmd)

    total_samps = 0
    while total_samps < num_samps:
        view = buff[total_samps:]
        num_rx_samps = rx_stream.recv(view, md, timeout=timeout)
        if num_rx_samps <= 0:
            break
        total_samps += num_rx_samps

    if total_samps == 0:
        raise RuntimeError("No samples received from SDR")

    iq_np = np.vstack((buff[:total_samps].real, buff[:total_samps].imag))
    return torch.from_numpy(iq_np).float()


class SdrCaptureBuffer:
    """Continuously read samples from ``rx_stream`` into a deque."""

    def __init__(self, rx_stream, rate: float, chunk_size: int):
        import uhd

        self.rx_stream = rx_stream
        self.md = uhd.types.RXMetadata()
        self.buff_size = chunk_size
        # allow a few chunks worth of backlog before dropping samples
        self.queue: deque = deque(maxlen=chunk_size * 10)
        self.timeout = chunk_size / rate + 0.1
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        buff = np.empty((self.buff_size,), dtype=np.complex64)
        while not self._stop.is_set():
            num_rx = self.rx_stream.recv(buff, self.md, timeout=self.timeout)
            if num_rx > 0:
                # append individual samples to the deque
                self.queue.extend(buff[:num_rx])
            else:
                time.sleep(0.001)

    def stop(self) -> None:
        self._stop.set()
        self.thread.join()


def iq_chunks_from_sdr(
    chunk_size: int, rate: float, freq: float, gain: float, antenna: str
):
    """Yield IQ samples from an SDR in chunks of ``chunk_size`` samples.

    Streaming is started in continuous mode and samples are collected until the
    caller stops iteration. The ``antenna`` parameter selects the RF port. Each
    yielded tensor has shape ``(2, chunk_size)``; if the radio provides fewer
    samples than requested the remainder is padded with zeros.
    """

    import uhd

    usrp = uhd.usrp.MultiUSRP()
    usrp.set_rx_rate(rate)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq))
    usrp.set_rx_gain(gain)
    usrp.set_rx_antenna(antenna)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_stream = usrp.get_rx_stream(stream_args)

    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    stream_cmd.stream_now = True
    rx_stream.issue_stream_cmd(stream_cmd)

    capture = SdrCaptureBuffer(rx_stream, rate, chunk_size)

    try:
        while True:
            while len(capture.queue) < chunk_size:
                time.sleep(0.001)
            samp = np.array([capture.queue.popleft() for _ in range(chunk_size)], dtype=np.complex64)
            iq_np = np.vstack((samp.real, samp.imag))
            yield torch.from_numpy(iq_np).float()
    finally:
        stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        rx_stream.issue_stream_cmd(stop_cmd)
        capture.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model inference on SDR or file input")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help=(
            "Number of output classes. "
            "If omitted, the value is inferred from the checkpoint"
        ),
    )
    parser.add_argument("--source", choices=["sdr", "file"], required=True, help="Input source")
    parser.add_argument("--file", help="Path to IQ file (.npy, .pt or .c16)")
    parser.add_argument("--num_samps", type=int, default=1024*1024, help="Number of IQ samples to capture")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1024 * 1024,
        help="Number of IQ samples per inference chunk when reading from a file or in continuous SDR mode",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continuously read from the SDR in chunks of --chunk_size",
    )
    parser.add_argument("--rate", type=float, default=14e6, help="SDR sample rate")
    parser.add_argument("--freq", type=float, default=2.4e9, help="SDR center frequency")
    parser.add_argument("--gain", type=float, default=0.0, help="SDR gain")
    parser.add_argument(
        "--antenna", default="TX/RX", help="SDR antenna selection"
    )
    parser.add_argument("--device", default="cpu", help="Computation device")
    parser.add_argument(
        "--class-names",
        default=None,
        help=(
            "Optional comma-separated list or path to a text file containing "
            "class names. One class per line when using a file. "
            "Defaults to names from class_stats.csv if present."
        ),
    )
    args = parser.parse_args()

    state = torch.load(args.weights, map_location=args.device)

    class_names = None
    if isinstance(state, dict) and "class_names" in state:
        class_names = state["class_names"]

    if args.class_names:
        if os.path.exists(args.class_names):
            with open(args.class_names, "r") as f:
                class_names = [line.strip() for line in f if line.strip()]
        else:
            class_names = [name.strip() for name in args.class_names.split(",")]

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

    # Convert checkpoint to a state_dict regardless of how it was saved
    if isinstance(state, torch.nn.Module):
        state_dict = state.state_dict()
    else:
        state_dict = state.get("state_dict", state)

    # Infer the number of classes from the checkpoint if not provided
    if args.num_classes is None:
        if "classifier.3.weight" in state_dict:
            num_classes = state_dict["classifier.3.weight"].shape[0]
        elif "module.classifier.3.weight" in state_dict:
            num_classes = state_dict["module.classifier.3.weight"].shape[0]
        else:
            raise KeyError(
                "Could not determine number of classes from checkpoint. "
                "Please provide --num_classes."
            )
    else:
        num_classes = args.num_classes
        ckpt_classes = None
        if "classifier.3.weight" in state_dict:
            ckpt_classes = state_dict["classifier.3.weight"].shape[0]
        elif "module.classifier.3.weight" in state_dict:
            ckpt_classes = state_dict["module.classifier.3.weight"].shape[0]
        if ckpt_classes is not None and ckpt_classes != num_classes:
            raise RuntimeError(
                f"Checkpoint expects {ckpt_classes} classes, "
                f"but --num_classes={num_classes}"
            )

    model = model_VGG2D.vgg11_bn(num_classes=num_classes)
    if isinstance(state, torch.nn.Module):
            state_dict = state.state_dict()
    else:
            state_dict = state.get("state_dict", state)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    # Use the exact same spectrogram configuration as during training to avoid
    # any mismatch between the training and inference pipelines.
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
        iq_iter = iq_chunks_from_file(args.file, args.chunk_size)
    else:
        if args.continuous:
            iq_iter = iq_chunks_from_sdr(
                args.chunk_size, args.rate, args.freq, args.gain, args.antenna
            )
        else:
            iq_full = capture_from_sdr(
                args.num_samps, args.rate, args.freq, args.gain, args.antenna
            )
            iq_iter = [iq_full]

    try:
        for i, iq in enumerate(iq_iter):
            iq = iq.to(args.device)
            spec = transform(iq).unsqueeze(0)  # (1, 2, 1024, 1024)

            with torch.no_grad():
                logits = model(spec)
                pred_idx = logits.argmax(1).item()
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            # pred_label = (
            #     class_names[pred_idx]
            #     if class_names is not None and pred_idx < len(class_names)
            #     else str(pred_idx)
            # )
            # print(
            #     f"Chunk {i}: Detected {pred_label} "
            #     f"({probs[pred_idx]:.2%})"
            # )
            # for j, p in enumerate(probs):
                # label = (
                    # class_names[j] if class_names is not None and j < len(class_names) else str(j)
                # )
            label = " "
            if class_names is not None:
                label = "Noise" if class_names[pred_idx] == "Noise" else "Drone DETECTED!" 
            marker = "  "
            print(f"{marker} {label:<15} {probs[pred_idx]:.4f}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
