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
                & rows.method.isin(METHODS)]
    tasks = ((5, 1), (5, 5), (5, 10), (7, 1), (7, 5), (7, 10))
    print("Novel held-out gestures — matched support/query episodes")
    print("Task             Method                                        Accuracy           Episodes")
    for n_way, k_shot in tasks:
        subset = rows[(rows.n_way == n_way) & (rows.k_shot == k_shot)]
        missing = set(METHODS) - set(subset.method)
        if missing:
            print(f"{n_way}-way, {k_shot}-shot   MISSING: {', '.join(sorted(missing))}")
            continue
        for _, row in subset.sort_values("method").iterrows():
            print(f"{n_way}-way, {k_shot}-shot   {METHODS[row.method]:45} "
                  f"{row.acc_mean:5.2f} ± {row.acc_std:4.2f}%    {int(row.n_episodes)}")


if __name__ == "__main__":
    main()
