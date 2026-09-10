"""Readable representative curves: fixed 7-way, 10-shot task."""

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import pandas as pd

from common.config import FIGURES_DIR, RESULTS_DIR

METHODS = ["protonet", "siamese", "relationnet", "feat"]
LABELS = {
    "mobilenetv4_conv_small.e2400_r224_in1k": "MobileNetV4-S",
    "tf_efficientnetv2_s.in21k": "EfficientNetV2-S",
    "wide_resnet50_2": "WideResNet-50-2",
}


def main():
    output = FIGURES_DIR / "training_curves.png"
    if output.exists() and "--force" not in sys.argv:
        print(f"completed; skipping {output}")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
    for method_i, method in enumerate(METHODS):
        loss_ax, acc_ax = axes[0, method_i], axes[1, method_i]
        for path in glob.glob(
            str(RESULTS_DIR / "episode_logs" / f"{method}_*_7way_10shot.csv")
        ):
            frame = pd.read_csv(path).sort_values("episode")
            smooth = frame[["loss", "acc"]].rolling(25, min_periods=1).mean()
            backbone = path.split(f"{method}_", 1)[1].rsplit("_7way_10shot.csv", 1)[0]
            label = LABELS.get(backbone, backbone)
            loss_ax.plot(frame.episode, smooth.loss, linewidth=1.5, label=label)
            acc_ax.plot(frame.episode, smooth.acc, linewidth=1.5, label=label)
        loss_ax.set_title(method.replace("net", "Net").title())
        loss_ax.set_ylabel("Loss (25-episode mean)")
        loss_ax.grid(alpha=0.25)
        acc_ax.set_xlabel("Episode")
        acc_ax.set_ylabel("Accuracy (%)")
        acc_ax.set_ylim(0, 102)
        acc_ax.grid(alpha=0.25)
        if method_i == 0:
            loss_ax.legend(fontsize=8)
            acc_ax.legend(fontsize=8)
    fig.suptitle("Training dynamics on the representative 7-way, 10-shot task", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
