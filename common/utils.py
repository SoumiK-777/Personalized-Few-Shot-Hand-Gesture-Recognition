from __future__ import annotations

import csv
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import CKPT_DIR, RESULTS_DIR

RESULT_COLUMNS = [
    "method",
    "backbone",
    "n_way",
    "k_shot",
    "split",
    "loss_mean",
    "acc_mean",
    "acc_std",
    "inf_ms_mean",
    "n_episodes",
    "timestamp",
]
MANIFEST_COLUMNS = [
    "method",
    "backbone",
    "n_way",
    "k_shot",
    "path",
    "episode",
    "test_acc",
    "timestamp",
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_data_parallel(model):
    """Use all visible CUDA devices; harmlessly returns the original model on one/CPU GPU."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        return nn.DataParallel(model)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def portable_state_dict(model):
    return unwrap_model(model).state_dict()


def load_portable_state_dict(model, state_dict):
    return unwrap_model(model).load_state_dict(state_dict)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def append_csv(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader() if write_header else None
        writer.writerow(row)


def result_exists(method, backbone, n_way, k_shot, split="test"):
    path = RESULTS_DIR / "all_results.csv"
    if not path.exists():
        return False
    with path.open(newline="") as f:
        return any(
            r["method"] == method
            and r["backbone"] == backbone
            and int(r["n_way"]) == n_way
            and int(r["k_shot"]) == k_shot
            and r["split"] == split
            for r in csv.DictReader(f)
        )


def latest_path(method, backbone, n_way, k_shot):
    return CKPT_DIR / f"{method}_{backbone}_{n_way}way_{k_shot}shot_latest.pt"


def best_path(method, backbone, n_way, k_shot):
    return CKPT_DIR / f"{method}_{backbone}_{n_way}way_{k_shot}shot_best.pt"


def record_checkpoint(method, backbone, n_way, k_shot, path, episode, test_acc):
    append_csv(
        RESULTS_DIR / "checkpoints_manifest.csv",
        {
            "method": method,
            "backbone": backbone,
            "n_way": n_way,
            "k_shot": k_shot,
            "path": str(path),
            "episode": episode,
            "test_acc": test_acc,
            "timestamp": timestamp(),
        },
        MANIFEST_COLUMNS,
    )


def checkpoint_test_acc(path):
    if not Path(path).exists():
        return float("-inf")
    try:
        return float(
            torch.load(path, map_location="cpu", weights_only=False).get(
                "test_acc", float("-inf")
            )
        )
    except Exception:
        return float("-inf")


def save_state(state, method, backbone, n_way, k_shot, episode, test_acc, best=False):
    path = (
        best_path(method, backbone, n_way, k_shot)
        if best
        else latest_path(method, backbone, n_way, k_shot)
    )
    if best and test_acc <= checkpoint_test_acc(path):
        return False
    torch.save(state, path)
    record_checkpoint(method, backbone, n_way, k_shot, path, episode, test_acc)
    print(f"checkpoint saved: {path} | episode={episode} | test_acc={test_acc:.3f}")
    return True


def append_result(**row):
    append_csv(
        RESULTS_DIR / "all_results.csv",
        {**row, "timestamp": timestamp()},
        RESULT_COLUMNS,
    )


def append_episode(method, backbone, n_way, k_shot, episode, loss, acc, inf_ms):
    path = (
        RESULTS_DIR
        / "episode_logs"
        / f"{method}_{backbone}_{n_way}way_{k_shot}shot.csv"
    )
    append_csv(
        path,
        {
            "episode": episode,
            "loss": loss,
            "acc": acc,
            "inf_ms": inf_ms,
            "timestamp": timestamp(),
        },
        ["episode", "loss", "acc", "inf_ms", "timestamp"],
    )
