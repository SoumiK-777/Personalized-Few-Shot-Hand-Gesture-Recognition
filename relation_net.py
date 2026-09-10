"""Relation Network with full, resumable embed+relation checkpoints."""

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
    load_portable_state_dict,
    maybe_data_parallel,
    portable_state_dict,
    result_exists,
    save_state,
    set_seed,
)

METHOD = "relationnet"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class RelationModule(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, 512), nn.ReLU(), nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def logits(embed, relation, sx, sy, qx, n):
    support = embed(sx)
    proto = torch.stack([support[sy == i].mean(0) for i in range(n)])
    start = time.perf_counter()
    query = embed(qx)
    q, d = query.shape
    pairs = torch.cat(
        [proto.unsqueeze(0).expand(q, n, d), query.unsqueeze(1).expand(q, n, d)], 2
    ).reshape(q * n, 2 * d)
    return relation(pairs).reshape(q, n), (time.perf_counter() - start) / q


def train_episode(embed, rel, opt, sx, sy, qx, qy, n):
    embed.train()
    rel.train()
    opt.zero_grad()
    sx, sy, qx, qy = (x.to(device) for x in (sx, sy, qx, qy))
    scores, t = logits(embed, rel, sx, sy, qx, n)
    loss = F.mse_loss(scores, F.one_hot(qy, n).float())
    loss.backward()
    opt.step()
    return loss.item(), (scores.argmax(1) == qy).float().mean().item() * 100, t


@torch.no_grad()
def evaluate(model, split="test", n_episodes=200, n_way=5, k_shot=1):
    embed, rel = model
    embed.eval()
    rel.eval()
    rows = []
    for _ in range(n_episodes):
        sx, qx, sy, qy, *_ = sample_episode(
            DATA_DIR, split, n_way, k_shot, QUERY_PER_CLASS, TRANSFORM
        )
        if sx is None:
            continue
        sx, sy, qx, qy = (x.to(device) for x in (sx, sy, qx, qy))
        scores, t = logits(embed, rel, sx, sy, qx, n_way)
        rows.append(
            (
                F.mse_loss(scores, F.one_hot(qy, n_way).float()).item(),
                (scores.argmax(1) == qy).float().mean().item() * 100,
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


def state(embed, rel, opt, ep, b, n, k, acc):
    return {
        "embed_model_state_dict": portable_state_dict(embed),
        "relation_model_state_dict": portable_state_dict(rel),
        "optimizer_state_dict": opt.state_dict(),
        "episode": ep,
        "backbone": b,
        "n_way": n,
        "k_shot": k,
        "test_acc": acc,
    }


def main():
    set_seed()
    episodes = int(os.getenv("EPISODES_PER_TASK", EPISODES_PER_TASK))
    for b in BACKBONES:
        for n, k in TASKS:
            if result_exists(METHOD, b, n, k):
                print("completed; skipping", b, n, k)
                continue
            base_embed = timm.create_model(
                b, pretrained=True, num_classes=0, global_pool="avg"
            ).to(device)
            # A one-image probe must run in eval mode: MobileNet's BatchNorm cannot
            # compute training statistics from a 1x1 spatial feature map.
            base_embed.eval()
            with torch.no_grad():
                d = base_embed(torch.zeros(1, 3, 224, 224, device=device)).shape[1]
            base_embed.train()
            embed = maybe_data_parallel(base_embed)
            rel = maybe_data_parallel(RelationModule(d).to(device))
            opt = torch.optim.Adam(
                list(embed.parameters()) + list(rel.parameters()), lr=1e-3
            )
            latest = latest_path(METHOD, b, n, k)
            start = 0
            if latest.exists():
                s = torch.load(latest, map_location=device, weights_only=False)
                load_portable_state_dict(embed, s["embed_model_state_dict"])
                load_portable_state_dict(rel, s["relation_model_state_dict"])
                opt.load_state_dict(s["optimizer_state_dict"])
                start = s["episode"] + 1
            for ep in range(start, episodes):
                sx, qx, sy, qy, *_ = sample_episode(
                    DATA_DIR, "train", n, k, QUERY_PER_CLASS, TRANSFORM
                )
                if sx is None:
                    raise RuntimeError("No valid training episode")
                loss, acc, t = train_episode(embed, rel, opt, sx, sy, qx, qy, n)
                append_episode(METHOD, b, n, k, ep, loss, acc, t * 1000)
                if (ep + 1) % AUTOSAVE_EVERY == 0:
                    save_state(
                        state(
                            embed, rel, opt, ep, b, n, k, checkpoint_test_acc(latest)
                        ),
                        METHOD,
                        b,
                        n,
                        k,
                        ep,
                        checkpoint_test_acc(latest),
                    )
            m = evaluate((embed, rel), "test", EVAL_EPISODES, n, k)
            s = state(embed, rel, opt, episodes - 1, b, n, k, m["acc_mean"])
            save_state(s, METHOD, b, n, k, episodes - 1, m["acc_mean"], True)
            save_state(s, METHOD, b, n, k, episodes - 1, m["acc_mean"])
            append_result(
                method=METHOD, backbone=b, n_way=n, k_shot=k, split="test", **m
            )
            torch.save(
                {
                    "embed_model_state_dict": portable_state_dict(embed),
                    "relation_model_state_dict": portable_state_dict(rel),
                },
                CKPT_DIR / f"{METHOD}_{b}.pt",
            )


if __name__ == "__main__":
    main()
