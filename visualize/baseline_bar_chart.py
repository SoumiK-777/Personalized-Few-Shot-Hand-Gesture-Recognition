"""Fair EfficientNetV2-S comparison: FSL methods against adaptation baselines."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import FIGURES_DIR, RESULTS_DIR
from common.results import load_results

BACKBONE = "tf_efficientnetv2_s.in21k"
FSL = ["protonet", "siamese", "relationnet", "feat"]
DISPLAY = {
    "protonet": "ProtoNet",
    "siamese": "Siamese",
    "relationnet": "RelationNet",
    "feat": "FEAT",
    "frozen_centroid": "Frozen centroid",
    "best_linear": "Best linear FT",
}


def main():
    output = FIGURES_DIR / "baseline_bar_chart.png"
    if output.exists() and "--force" not in sys.argv:
        print(f"completed; skipping {output}")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results(RESULTS_DIR)
    data = data[(data.split == "test") & (data.backbone == BACKBONE)].copy()
    linear = data[data.method.str.startswith("linear_finetune_")]
    best_linear = linear.loc[
        linear.groupby(["n_way", "k_shot"])["acc_mean"].idxmax()
    ].copy()
    best_linear["method"] = "best_linear"
    selected = pd.concat(
        [data[data.method.isin(FSL + ["frozen_centroid"])], best_linear],
        ignore_index=True,
    )
    methods = FSL + ["frozen_centroid", "best_linear"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7F7F7F", "#56B4E9"]
    for ax, n_way in zip(axes, (5, 7)):
        subset = selected[selected.n_way == n_way]
        x = np.arange(3)
        width = 0.13
        for i, method in enumerate(methods):
            rows = (
                subset[subset.method == method].set_index("k_shot").reindex([1, 5, 10])
            )
            ax.bar(
                x + (i - 2.5) * width,
                rows.acc_mean,
                width,
                yerr=rows.acc_std,
                capsize=2,
                color=colors[i],
                label=DISPLAY[method],
            )
        ax.set_title(f"{n_way}-way tasks")
        ax.set_xticks(x, ["1-shot", "5-shot", "10-shot"])
        ax.set_xlabel("Support examples per class")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Held-out accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "EfficientNetV2-S: few-shot methods versus adaptation baselines", y=0.99
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
