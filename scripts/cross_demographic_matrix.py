"""Evaluate the already trained best 5-way, 10-shot ProtoNet across demographics.

This script does not train or fine-tune anything. It evaluates the selected global
ProtoNet + EfficientNetV2-S checkpoint separately on Indian, White, Black, and
race-balanced Mix held-out episodes.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.config import CKPT_DIR, DATA_DIR, RESULTS_DIR, SEED
from common.data import sample_episode
from common.utils import load_portable_state_dict, set_seed
from proto_net import TRANSFORM, ProtoNet, device, logits_for_episode

BACKBONE = "tf_efficientnetv2_s.in21k"
N_WAY = 5
K_SHOT = 10
QUERY_PER_CLASS = 10
GROUPS = ("Indian", "White", "Black", "Mix")


@torch.no_grad()
def evaluate(model, group, n_episodes):
    model.eval()
    accuracies = []
    for _ in range(n_episodes):
        race = None if group == "Mix" else group
        sx, qx, sy, qy, *_ = sample_episode(
            DATA_DIR, "test", N_WAY, K_SHOT, QUERY_PER_CLASS, TRANSFORM, race
        )
        if sx is None:
            continue
        sx, sy, qx, qy = (x.to(device) for x in (sx, sy, qx, qy))
        logits, _ = logits_for_episode(model, sx, sy, qx, N_WAY)
        accuracies.append((logits.argmax(1) == qy).float().mean().item() * 100)
    if not accuracies:
        raise RuntimeError(f"No valid held-out test episodes for {group}")
    return float(np.mean(accuracies)), float(np.std(accuracies)), len(accuracies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CKPT_DIR / f"protonet_{BACKBONE}_5way_10shot_best.pt",
    )
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {args.checkpoint}")
    set_seed(SEED)
    model = ProtoNet(BACKBONE).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_portable_state_dict(
        model, state["model_state_dict"] if "model_state_dict" in state else state
    )
    rows = []
    for group in GROUPS:
        mean, std, count = evaluate(model, group, args.episodes)
        rows.append(
            {
                "model": "ProtoNet + EfficientNetV2-S (global training)",
                "test_group": group,
                "n_way": N_WAY,
                "k_shot": K_SHOT,
                "acc_mean": mean,
                "acc_std": std,
                "n_episodes": count,
            }
        )
        print(f"Global -> {group}: {mean:.2f} +/- {std:.2f}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "cross_demographic_best_protonet_5way_10shot.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
