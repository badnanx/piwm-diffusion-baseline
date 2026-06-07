import os
from collections import defaultdict
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _flatten_metrics(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_metrics(next_prefix, child, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def plot_history_curves(history: list[dict[str, Any]], output_path: str) -> None:
    """
    Plot all scalar metrics in a training history.

    Expected rows look like:
        {"epoch": 1, "train": {"loss": ...}, "test": {"loss": ...}}
    """
    if not history:
        return

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row_idx, row in enumerate(history, start=1):
        epoch = int(row.get("epoch", row_idx))
        flat: dict[str, float] = {}
        for key, value in row.items():
            if key == "epoch":
                continue
            _flatten_metrics(str(key), value, flat)
        for key, value in flat.items():
            series[key].append((epoch, value))

    if not series:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in sorted(series):
        points = series[key]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3, label=key)

    ax.set_xlabel("epoch")
    ax.set_ylabel("value")
    ax.set_title("Training history")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
