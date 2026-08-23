"""Accuracy versus K-shot, separated by task complexity and backbone."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from common.config import RESULTS_DIR, FIGURES_DIR
from common.results import load_results

METHODS = ["protonet", "siamese", "relationnet", "feat"]
BACKBONES = ["mobilenetv4_conv_small.e2400_r224_in1k", "tf_efficientnetv2_s.in21k", "wide_resnet50_2"]
BACKBONE_LABELS = ["MobileNetV4-S", "EfficientNetV2-S", "WideResNet-50-2"]

def main():
    output = FIGURES_DIR / "acc_vs_kshot.png"
    if output.exists() and "--force" not in sys.argv: print(f"completed; skipping {output}"); return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results(RESULTS_DIR)
    data = data[(data.split == "test") & data.method.isin(METHODS)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    colors = dict(zip(METHODS, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
    for row, n_way in enumerate((5, 7)):
        for col, (backbone, title) in enumerate(zip(BACKBONES, BACKBONE_LABELS)):
            ax = axes[row, col]; subset = data[(data.n_way == n_way) & (data.backbone == backbone)]
            for method in METHODS:
                line = subset[subset.method == method].sort_values("k_shot")
                if not line.empty:
                    ax.errorbar(line.k_shot, line.acc_mean, yerr=line.acc_std, marker="o", capsize=3, linewidth=2, color=colors[method], label=method.replace("net", "Net").title())
            ax.set_title(title); ax.set_xticks([1, 5, 10]); ax.grid(axis="y", alpha=.25)
            if col == 0: ax.set_ylabel(f"{n_way}-way accuracy (%)")
            if row == 1: ax.set_xlabel("Shots per class (K)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Few-shot accuracy by task complexity and backbone", y=.98)
    fig.tight_layout(rect=(0, .07, 1, .95)); fig.savefig(output, dpi=300); plt.close(fig)
    print(f"wrote {output}")
if __name__ == "__main__": main()
