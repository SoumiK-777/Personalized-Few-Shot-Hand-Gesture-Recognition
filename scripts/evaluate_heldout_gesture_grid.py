"""Evaluate supervised and ProtoNet transfer on all novel-gesture task sizes.

For every N-way/K-shot condition, both methods score the identical held-out
episodes.  The supervised representation is the 24-base-class cross-entropy
model; each ProtoNet uses the checkpoint trained for the matching N-way/K-shot
base-class task. Results are appended to ``all_results.csv`` with split
``novel_test``.
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

from common.config import ALLOWED_GESTURES, CKPT_DIR, DATA_DIR, EVAL_EPISODES, RACES, TASKS
from common.data import classes_in_dataset, sample_cross_split_episode
from common.utils import append_result, result_exists, set_seed

BACKBONE = "tf_efficientnetv2_s.in21k"
METHODS = ("supervised_linear_probe_novel", "protonet_novel")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_supervised(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("backbone") != BACKBONE or set(state.get("classes", [])) != set(ALLOWED_GESTURES):
        raise ValueError("The supervised checkpoint is not the required 24-base-class EfficientNetV2-S model.")
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(ALLOWED_GESTURES)).to(device)
    model.load_state_dict(state["model_state_dict"])
    return model.eval()


def load_protonet(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    weights = state.get("model_state_dict", state)
    weights = {key.removeprefix("module.backbone.").removeprefix("backbone."): value
               for key, value in weights.items()}
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=0, global_pool="avg").to(device)
    model.load_state_dict(weights, strict=True)
    return model.eval()


@torch.no_grad()
def supervised_features(model, images):
    return model.forward_head(model.forward_features(images), pre_logits=True)


@torch.no_grad()
def protonet_scores(model, support_images, support_labels, query_images, n_way):
    support = model(support_images)
    query = model(query_images)
    prototypes = torch.stack([support[support_labels == label].mean(0) for label in range(n_way)])
    return -torch.cdist(query, prototypes).pow(2)


def linear_probe_scores(support, support_labels, query, n_way, steps, learning_rate):
    classifier = torch.nn.Linear(support.shape[1], n_way).to(support.device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        F.cross_entropy(classifier(support), support_labels).backward()
        optimizer.step()
    return classifier(query)


def novel_classes():
    labels = set()
    for race in RACES:
        labels.update(classes_in_dataset(DATA_DIR, race))
    return sorted(labels - set(ALLOWED_GESTURES))


def sample_episodes(labels, n_way, k_shot, count, queries_per_class):
    episodes = []
    for _ in range(count):
        episode = sample_cross_split_episode(
            DATA_DIR, "train", "test", n_way, k_shot, queries_per_class, TRANSFORM, labels
        )
        if episode[0] is not None:
            episodes.append(episode[:4])
    if not episodes:
        raise RuntimeError(f"No valid novel-class episodes for {n_way}-way, {k_shot}-shot")
    return episodes


def evaluate(method, model, episodes, n_way, probe_steps, probe_lr, device):
    values = []
    for sx, qx, sy, qy in episodes:
        sx, qx, sy, qy = (value.to(device) for value in (sx, qx, sy, qy))
        start = time.perf_counter()
        if method == "supervised_linear_probe_novel":
            with torch.no_grad():
                support, query = supervised_features(model, sx), supervised_features(model, qx)
            scores = linear_probe_scores(support, sy, query, n_way, probe_steps, probe_lr)
        else:
            scores = protonet_scores(model, sx, sy, qx, n_way)
        latency = (time.perf_counter() - start) * 1000 / len(qx)
        values.append((F.cross_entropy(scores, qy).item(),
                       (scores.argmax(1) == qy).float().mean().item() * 100, latency))
    values = np.asarray(values)
    return dict(loss_mean=float(values[:, 0].mean()), acc_mean=float(values[:, 1].mean()),
                acc_std=float(values[:, 1].std()), inf_ms_mean=float(values[:, 2].mean()),
                n_episodes=len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervised-checkpoint", type=Path,
                        default=CKPT_DIR / "supervised_effnetv2s_base24_best.pt")
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--queries-per-class", type=int, default=10)
    parser.add_argument("--probe-steps", type=int, default=50)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true",
                        help="Recompute every task pair and retain only the newest result rows.")
    args = parser.parse_args()
    set_seed()
    labels = novel_classes()
    if len(labels) < max(n_way for n_way, _ in TASKS):
        raise RuntimeError(f"Need at least 7 held-out labels; found {labels}")
    print("Novel held-out labels:", ", ".join(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervised = load_supervised(args.supervised_checkpoint, device)

    for n_way, k_shot in TASKS:
        completed = all(result_exists(method, BACKBONE, n_way, k_shot, split="novel_test") for method in METHODS)
        if completed and not args.force:
            print(f"completed; skipping matched pair {n_way}-way {k_shot}-shot")
            continue
        protonet_path = CKPT_DIR / f"protonet_{BACKBONE}_{n_way}way_{k_shot}shot_best.pt"
        if not protonet_path.exists():
            raise FileNotFoundError(f"Missing task-matched ProtoNet checkpoint: {protonet_path}")
        episodes = sample_episodes(labels, n_way, k_shot, args.episodes, args.queries_per_class)
        print(f"{n_way}-way {k_shot}-shot: sampled {len(episodes)} matched episodes")
        models = {
            "supervised_linear_probe_novel": supervised,
            "protonet_novel": load_protonet(protonet_path, device),
        }
        for method, model in models.items():
            metrics = evaluate(method, model, episodes, n_way, args.probe_steps, args.probe_lr, device)
            append_result(method=method, backbone=BACKBONE, n_way=n_way, k_shot=k_shot,
                          split="novel_test", **metrics)
            print(f"{method} | {n_way}-way {k_shot}-shot | {metrics}")


if __name__ == "__main__":
    main()
