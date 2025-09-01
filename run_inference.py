import argparse
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


def capture_from_sdr(num_samps: int, rate: float, freq: float, gain: float) -> torch.Tensor:
    """Capture IQ samples from a UHD-compatible SDR."""
    import uhd
    usrp = uhd.usrp.MultiUSRP()
    usrp.set_rx_rate(rate)
    usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq))
    usrp.set_rx_gain(gain)

    stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_stream = usrp.get_rx_stream(stream_args)
    md = uhd.types.RXMetadata()
    buff = np.zeros((num_samps,), dtype=np.complex64)
    rx_stream.recv([buff], num_samps, md)
    iq_np = np.vstack((buff.real, buff.imag))
    return torch.from_numpy(iq_np).float()


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
        help="Number of IQ samples per inference chunk when reading from a file",
    )
    parser.add_argument("--rate", type=float, default=14e6, help="SDR sample rate")
    parser.add_argument("--freq", type=float, default=2.4e9, help="SDR center frequency")
    parser.add_argument("--gain", type=float, default=0.0, help="SDR gain")
    parser.add_argument("--device", default="cpu", help="Computation device")
    args = parser.parse_args()

    state = torch.load(args.weights, map_location=args.device)

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

    transform = transform_spectrogram(device=args.device)

    if args.source == "file":
        if not args.file:
            raise ValueError("--file path required when source is 'file'")
        iq_iter = iq_chunks_from_file(args.file, args.chunk_size)
    else:
        iq_full = capture_from_sdr(args.num_samps, args.rate, args.freq, args.gain)
        iq_iter = [iq_full]

    for i, iq in enumerate(iq_iter):
        iq = iq.to(args.device)
        spec = transform(iq).unsqueeze(0)  # (1, 2, 1024, 1024)

        with torch.no_grad():
            logits = model(spec)
            pred_idx = logits.argmax(1).item()
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        print(f"Chunk {i}: Prediction index: {pred_idx}")
        print("Probabilities:", probs)


if __name__ == "__main__":
    main()
