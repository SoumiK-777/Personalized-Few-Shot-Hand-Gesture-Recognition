"""FEAT: transformer-adapted prototypes with resumable episodic training."""

import os
import time

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from common.config import (
    AUTOSAVE_EVERY,
    BACKBONES,
    CKPT_DIR,
    DATA_DIR,
    EPISODES_PER_TASK,
    EVAL_EPISODES,
    QUERY_PER_CLASS,
    TASKS,
)
from common.data import sample_episode
from common.utils import (
    append_episode,
    append_result,
    checkpoint_test_acc,
    latest_path,
    maybe_data_parallel,
    result_exists,
    save_state,
    set_seed,
)

METHOD = "feat"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class FEATNet(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        backbone.eval()
        with torch.no_grad():
            self.feat_dim = backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        backbone.train()
        self.backbone = maybe_data_parallel(backbone)
        heads = next(h for h in range(8, 0, -1) if self.feat_dim % h == 0)
        layer = nn.TransformerEncoderLayer(
            self.feat_dim, heads, self.feat_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, 1)
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def adapt_prototypes(self, sx, sy, n_way):
        features = self.backbone(sx)
        prototypes = []
        for i in range(n_way):
            selected = features[sy == i]
            if selected.numel() == 0:
                raise ValueError(f"empty support mask for class {i}")
            prototypes.append(selected.mean(0))
        return self.transformer(torch.stack(prototypes).unsqueeze(0)).squeeze(0)


def feat_state_dict(model):
    return {
        key.replace("backbone.module.", "backbone."): value
        for key, value in model.state_dict().items()
    }


def load_feat_state_dict(model, state):
    if isinstance(model.backbone, nn.DataParallel):
        state = {
            (
                key.replace("backbone.", "backbone.module.", 1)
                if key.startswith("backbone.")
                else key
            ): value
            for key, value in state.items()
        }
    model.load_state_dict(state)


def logits_for_episode(model, sx, sy, qx, n_way):
    prototypes = model.adapt_prototypes(sx, sy, n_way)
    start = time.perf_counter()
    query = model.backbone(qx)
    logits = (
        model.temperature * F.normalize(query, dim=1) @ F.normalize(prototypes, dim=1).T
    )
    return logits, (time.perf_counter() - start) / len(qx)


def train_episode(model, opt, sx, sy, qx, qy, n):
    model.train()
    opt.zero_grad()
    sx, sy, qx, qy = (x.to(device) for x in (sx, sy, qx, qy))
    logits, t = logits_for_episode(model, sx, sy, qx, n)
    loss = F.cross_entropy(logits, qy)
    loss.backward()
    opt.step()
    return loss.item(), (logits.argmax(1) == qy).float().mean().item() * 100, t


@torch.no_grad()
def evaluate(model, split="test", n_episodes=200, n_way=5, k_shot=1):
    model.eval()
    rows = []
    for _ in range(n_episodes):
        sx, qx, sy, qy, *_ = sample_episode(
            DATA_DIR, split, n_way, k_shot, QUERY_PER_CLASS, TRANSFORM
        )
        if sx is None:
            continue
        sx, sy, qx, qy = (x.to(device) for x in (sx, sy, qx, qy))
        logits, t = logits_for_episode(model, sx, sy, qx, n_way)
        rows.append(
            (
                F.cross_entropy(logits, qy).item(),
                (logits.argmax(1) == qy).float().mean().item() * 100,
                t * 1000,
            )
        )
    a = np.asarray(rows) if rows else np.zeros((1, 3))
    return dict(
        loss_mean=float(a[:, 0].mean()),
        acc_mean=float(a[:, 1].mean()),
        acc_std=float(a[:, 1].std()),
        inf_ms_mean=float(a[:, 2].mean()),
        n_episodes=len(rows),
    )


def main():
    set_seed()
    episodes = int(os.getenv("EPISODES_PER_TASK", EPISODES_PER_TASK))
    for backbone in BACKBONES:
        for n, k in TASKS:
            if result_exists(METHOD, backbone, n, k):
                print("completed; skipping", backbone, n, k)
                continue
            model = FEATNet(backbone).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            latest = latest_path(METHOD, backbone, n, k)
            start = 0
            if latest.exists():
                state = torch.load(latest, map_location=device, weights_only=False)
                load_feat_state_dict(model, state["model_state_dict"])
                opt.load_state_dict(state["optimizer_state_dict"])
                start = state["episode"] + 1
            for ep in range(start, episodes):
                sx, qx, sy, qy, *_ = sample_episode(
                    DATA_DIR, "train", n, k, QUERY_PER_CLASS, TRANSFORM
                )
                if sx is None:
                    raise RuntimeError("No valid training episode")
                loss, acc, t = train_episode(model, opt, sx, sy, qx, qy, n)
                append_episode(METHOD, backbone, n, k, ep, loss, acc, t * 1000)
                if (ep + 1) % AUTOSAVE_EVERY == 0:
                    save_state(
                        {
                            "model_state_dict": feat_state_dict(model),
                            "optimizer_state_dict": opt.state_dict(),
                            "episode": ep,
                            "backbone": backbone,
                            "n_way": n,
                            "k_shot": k,
                            "test_acc": checkpoint_test_acc(latest),
                        },
                        METHOD,
                        backbone,
                        n,
                        k,
                        ep,
                        checkpoint_test_acc(latest),
                    )
            m = evaluate(model, "test", EVAL_EPISODES, n, k)
            state = {
                "model_state_dict": feat_state_dict(model),
                "optimizer_state_dict": opt.state_dict(),
                "episode": episodes - 1,
                "backbone": backbone,
                "n_way": n,
                "k_shot": k,
                "test_acc": m["acc_mean"],
            }
            save_state(state, METHOD, backbone, n, k, episodes - 1, m["acc_mean"], True)
            save_state(state, METHOD, backbone, n, k, episodes - 1, m["acc_mean"])
            append_result(
                method=METHOD, backbone=backbone, n_way=n, k_shot=k, split="test", **m
            )
            torch.save(feat_state_dict(model), CKPT_DIR / f"{METHOD}_{backbone}.pt")


if __name__ == "__main__":
    main()
