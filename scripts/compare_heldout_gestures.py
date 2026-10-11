"""Print the matched novel-gesture 5-way, 10-shot comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import RESULTS_DIR
from common.results import load_results

METHODS = {
    "supervised_linear_probe_novel": "Supervised EfficientNet + support linear probe",
    "protonet_novel": "ProtoNet + support prototypes",
}


def main():
    rows = load_results(RESULTS_DIR)
    rows = rows[(rows.split == "novel_test") & (rows.backbone == "tf_efficientnetv2_s.in21k")
                & (rows.n_way == 5) & (rows.k_shot == 10) & rows.method.isin(METHODS)]
    if rows.method.nunique() != len(METHODS):
        missing = set(METHODS) - set(rows.method)
        raise RuntimeError(f"Missing novel-class results: {sorted(missing)}. Run evaluate_heldout_gestures.py first.")
    # --force may append rows; use the latest result per method.
    rows = rows.sort_values("timestamp").groupby("method", as_index=False).tail(1)
    print("Novel held-out gestures — 5-way, 10-shot")
    print("Method                                        Accuracy           Episodes")
    for _, row in rows.sort_values("method").iterrows():
        print(f"{METHODS[row.method]:45} {row.acc_mean:5.2f} ± {row.acc_std:4.2f}%    {int(row.n_episodes)}")


if __name__ == "__main__":
    main()
