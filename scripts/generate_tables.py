"""Create canonical LaTeX result tables without duplicate rerun rows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common.config import RESULTS_DIR
from common.results import load_results

FSL = ["protonet", "siamese", "relationnet", "feat"]
BACKBONE_LABELS = {"mobilenetv4_conv_small.e2400_r224_in1k":"MobileNetV4-S", "tf_efficientnetv2_s.in21k":"EfficientNetV2-S", "wide_resnet50_2":"WideResNet-50-2"}

def table(rows, title):
    lines = [f"% {title}", "\\begin{tabular}{llrrrr}", "Backbone & N-way & K-shot & Loss & Accuracy (\\%) & Inference (ms) \\\\ \\hline"]
    for _, row in rows.iterrows():
        line = f"{BACKBONE_LABELS.get(row.backbone, row.backbone)} & {int(row.n_way)} & {int(row.k_shot)} & {row.loss_mean:.4f} & {row.acc_mean:.2f} $\\pm$ {row.acc_std:.2f} & {row.inf_ms_mean:.2f} \\\\"
        lines.append(line)
    return lines + ["\\end{tabular}", ""]

def main():
    output = RESULTS_DIR / "tables_I_VII.tex"
    if output.exists() and "--force" not in sys.argv:
        print(f"completed; skipping {output}"); return
    data = load_results(RESULTS_DIR); data = data[data.split == "test"]
    sections = []
    for method in FSL:
        rows = data[data.method == method].sort_values(["backbone", "n_way", "k_shot"])
        sections += table(rows, method.replace("net", "Net").title())
    frozen = data[data.method == "frozen_centroid"]
    linear = data[data.method.str.startswith("linear_finetune_")]
    best_linear = linear.loc[linear.groupby(["backbone", "n_way", "k_shot"])["acc_mean"].idxmax()].copy()
    best_linear["method"] = "Best linear fine-tune"
    sections += table(pd.concat([frozen, best_linear]).sort_values(["method", "n_way", "k_shot"]), "Baselines (best linear setting per task)")
    output.write_text("\n".join(sections)); print(f"wrote {output}")
if __name__ == "__main__": main()
