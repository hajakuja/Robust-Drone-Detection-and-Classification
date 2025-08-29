import argparse
import numpy as np
import torch
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
    """Load IQ samples from a .npy or .pt file."""
    if path.endswith(".npy"):
        iq_np = np.load(path)
    elif path.endswith(".pt"):
        data = torch.load(path)
        if isinstance(data, dict) and "x_iq" in data:
            tensor = data["x_iq"]
            if torch.is_tensor(tensor):
                iq_np = tensor.cpu().numpy()
            else:
                iq_np = np.asarray(tensor)
        else:
            iq_np = data if isinstance(data, np.ndarray) else np.asarray(data)
    else:
        raise ValueError("Unsupported file format: expected .npy or .pt")
    return torch.from_numpy(iq_np).float()


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
    parser.add_argument("--file", help="Path to IQ file (.npy or .pt)")
    parser.add_argument("--num_samps", type=int, default=1024*1024, help="Number of IQ samples to capture")
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
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    transform = transform_spectrogram(device=args.device)

    if args.source == "file":
        if not args.file:
            raise ValueError("--file path required when source is 'file'")
        iq = load_iq_from_file(args.file)
    else:
        iq = capture_from_sdr(args.num_samps, args.rate, args.freq, args.gain)

    iq = iq.to(args.device)
    spec = transform(iq).unsqueeze(0)  # (1, 2, 1024, 1024)

    with torch.no_grad():
        logits = model(spec)
        pred_idx = logits.argmax(1).item()
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    print(f"Prediction index: {pred_idx}")
    print("Probabilities:", probs)


if __name__ == "__main__":
    main()
