"""Compare supervised and ProtoNet representations on novel HaGRID gestures.

Base classes are the project's 24 ALLOWED_GESTURES.  Novel labels are discovered
from the data directory and never used to train either representation.  In every
episode, support comes from the novel-class train split and query from its test
split, so calibration cannot include a query image.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms

from common.config import ALLOWED_GESTURES, CKPT_DIR, DATA_DIR, EVAL_EPISODES, RACES
from common.data import classes_in_dataset, sample_cross_split_episode
from common.utils import append_result, result_exists, set_seed

BACKBONE = "tf_efficientnetv2_s.in21k"
METHODS = ("supervised_linear_probe_novel", "protonet_novel")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _load_supervised(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("backbone") != BACKBONE or set(state.get("classes", [])) != set(ALLOWED_GESTURES):
        raise ValueError("The supervised checkpoint is not the required 24-base-class EfficientNetV2-S model.")
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(ALLOWED_GESTURES)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def _load_protonet(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    weights = state.get("model_state_dict", state)
    # Existing ProtoNet checkpoints store parameters beneath ``backbone.``.
    weights = {k.removeprefix("module.backbone.").removeprefix("backbone."): v for k, v in weights.items()}
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=0, global_pool="avg").to(device)
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


@torch.no_grad()
def _supervised_features(model, x):
    features = model.forward_features(x)
    return model.forward_head(features, pre_logits=True)


@torch.no_grad()
def _protonet_scores(model, sx, sy, qx, n_way):
    support, query = model(sx), model(qx)
    prototypes = torch.stack([support[sy == i].mean(0) for i in range(n_way)])
    return -torch.cdist(query, prototypes).pow(2)


def _linear_probe_scores(support, sy, query, n_way, steps, lr):
    head = torch.nn.Linear(support.shape[1], n_way).to(support.device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(head(support), sy)
        loss.backward()
        optimizer.step()
    return head(query)


def _novel_classes():
    labels = set()
    for race in RACES:
        labels.update(classes_in_dataset(DATA_DIR, race))
    return sorted(labels - set(ALLOWED_GESTURES))


def sample_episodes(novel, args):
    """Draw one fixed episode list reused by every compared representation."""
    episodes = []
    for _ in range(args.episodes):
        sx, qx, sy, qy, _, _ = sample_cross_split_episode(
            DATA_DIR, "train", "test", 5, 10, args.queries_per_class, TRANSFORM, novel
        )
        if sx is not None:
            episodes.append((sx, qx, sy, qy))
    if not episodes:
        raise RuntimeError("No valid novel-class episodes could be sampled.")
    return episodes


def evaluate(method, model, episodes, args, device):
    rows = []
    for sx, qx, sy, qy in episodes:
        sx, qx, sy, qy = (x.to(device) for x in (sx, qx, sy, qy))
        start = time.perf_counter()
        if method == "supervised_linear_probe_novel":
            # Only the temporary classifier is adapted on the 50 support images.
            with torch.no_grad():
                support, query = _supervised_features(model, sx), _supervised_features(model, qx)
            scores = _linear_probe_scores(support, sy, query, 5, args.probe_steps, args.probe_lr)
        else:
            scores = _protonet_scores(model, sx, sy, qx, 5)
        latency = (time.perf_counter() - start) * 1000 / len(qx)
        rows.append((F.cross_entropy(scores, qy).item(), (scores.argmax(1) == qy).float().mean().item() * 100, latency))
    values = np.asarray(rows)
    return dict(loss_mean=float(values[:, 0].mean()), acc_mean=float(values[:, 1].mean()),
                acc_std=float(values[:, 1].std()), inf_ms_mean=float(values[:, 2].mean()),
                n_episodes=len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised-checkpoint", type=Path,
                        default=CKPT_DIR / "supervised_effnetv2s_base24_best.pt")
    parser.add_argument("--protonet-checkpoint", type=Path,
                        default=CKPT_DIR / "protonet_tf_efficientnetv2_s.in21k_5way_10shot_best.pt")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--queries-per-class", type=int, default=10)
    parser.add_argument("--probe-steps", type=int, default=50)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed()
    novel = _novel_classes()
    if len(novel) < 5:
        raise RuntimeError(f"Need at least 5 excluded gesture labels with train/test images; found: {novel}")
    print("Novel held-out labels:", ", ".join(novel))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {
        "supervised_linear_probe_novel": _load_supervised(args.supervised_checkpoint, device),
        "protonet_novel": _load_protonet(args.protonet_checkpoint, device),
    }
    episodes = sample_episodes(novel, args)
    print(f"Sampled {len(episodes)} matched novel-class episodes")
    for method, model in models.items():
        if not args.force and result_exists(method, BACKBONE, 5, 10, split="novel_test"):
            print(f"completed; skipping {method}")
            continue
        metrics = evaluate(method, model, episodes, args, device)
        append_result(method=method, backbone=BACKBONE, n_way=5, k_shot=10,
                      split="novel_test", **metrics)
        print(method, metrics)


if __name__ == "__main__":
    main()
