import argparse
import os
import curses
import numpy as np
import torch
from lib import model_VGG2D
from load_dataset import transform_spectrogram
import run_inference
from run_inference import iq_chunks_from_sdr

# Ensure torch can deserialize the VGG model
# run_inference already registers safe globals, but include here for clarity
torch.serialization.add_safe_globals([model_VGG2D.VGG])


def _get_input(stdscr, y, prompt, default):
    """Prompt the user for a value on ``stdscr`` at line ``y``."""
    stdscr.addstr(y, 0, f"{prompt} [{default}]: ")
    stdscr.clrtoeol()
    curses.echo()
    val = stdscr.getstr(y, len(prompt) + len(str(default)) + 3).decode("utf-8").strip()
    curses.noecho()
    stdscr.refresh()
    return val if val else default


def _select_from_list(stdscr, title, items):
    """Allow the user to choose an item from ``items`` using arrow keys."""
    idx = 0
    while True:
        stdscr.erase()
        stdscr.addstr(0, 0, title)
        for i, item in enumerate(items):
            marker = ">" if i == idx else " "
            stdscr.addstr(i + 1, 0, f"{marker} {item}")
        stdscr.refresh()
        key = stdscr.getch()
        if key in (curses.KEY_ENTER, 10, 13):
            return items[idx]
        elif key == curses.KEY_UP:
            idx = (idx - 1) % len(items)
        elif key == curses.KEY_DOWN:
            idx = (idx + 1) % len(items)


def _fft_graph(win, iq, height, width):
    """Draw a simple FFT magnitude graph of ``iq`` into ``win``."""
    iq_np = iq.cpu().numpy()
    fft_vals = np.fft.fft(iq_np[0] + 1j * iq_np[1])
    mag = np.abs(fft_vals)[: len(fft_vals) // 2]
    if mag.max() == 0:
        mag = mag + 1e-6
    step = max(1, len(mag) // width)
    scaled = mag[::step]
    max_val = scaled.max()
    for x in range(min(width, len(scaled))):
        bar = int(scaled[x] / max_val * (height - 1))
        for y in range(height):
            ch = ord("█") if y >= height - bar else ord(" ")
            try:
                win.addch(y, x, ch)
            except curses.error:
                pass


def run(stdscr, args):
    curses.curs_set(0)
    stdscr.erase()

    model_dir = _get_input(stdscr, 0, "Model directory", args.models_dir)
    models = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not models:
        raise RuntimeError(f"No .pth files found in {model_dir}")
    model_file = _select_from_list(stdscr, "Select model:", models)

    rate = float(_get_input(stdscr, len(models) + 2, "Sample rate", "14e6"))
    freq = float(_get_input(stdscr, len(models) + 3, "Center frequency", "2.4e9"))
    gain = float(_get_input(stdscr, len(models) + 4, "Gain", "0"))
    chunk_size = int(_get_input(stdscr, len(models) + 5, "Chunk size", "4096"))

    state = torch.load(os.path.join(model_dir, model_file), map_location=args.device)
    class_names = state.get("class_names") if isinstance(state, dict) else None

    state_dict = state.state_dict() if isinstance(state, torch.nn.Module) else state
    if "classifier.3.weight" in state_dict:
        num_classes = state_dict["classifier.3.weight"].shape[0]
    elif "module.classifier.3.weight" in state_dict:
        num_classes = state_dict["module.classifier.3.weight"].shape[0]
    else:
        raise KeyError("Could not infer number of classes from checkpoint")

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    model = model_VGG2D.vgg11_bn(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    transform = transform_spectrogram(device=args.device)
    iq_iter = iq_chunks_from_sdr(chunk_size, rate, freq, gain)

    height, width = stdscr.getmaxyx()
    prob_lines = len(class_names) + 2
    fft_height = max(1, height - prob_lines)

    try:
        for iq in iq_iter:
            iq = iq.to(args.device)
            spec = transform(iq).unsqueeze(0)
            with torch.no_grad():
                logits = model(spec)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

            stdscr.erase()
            stdscr.addstr(0, 0, "Detection probabilities:")
            for i, p in enumerate(probs):
                label = class_names[i] if i < len(class_names) else str(i)
                stdscr.addstr(1 + i, 0, f"{label:<15} {p:6.4f}")

            fft_win = stdscr.derwin(fft_height, width, prob_lines, 0)
            fft_win.erase()
            _fft_graph(fft_win, iq.cpu(), fft_height, width)
            fft_win.noutrefresh()
            stdscr.noutrefresh()
            curses.doupdate()
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Interactive SDR inference")
    parser.add_argument("--models-dir", default=".", help="Directory containing .pth files")
    parser.add_argument("--device", default="cpu", help="Computation device")
    args = parser.parse_args()
    curses.wrapper(run, args)


if __name__ == "__main__":
    main()
