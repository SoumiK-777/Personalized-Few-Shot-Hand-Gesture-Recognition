"""Participant-disjoint personalized ProtoNet evaluation for a consented dataset.

Expected layout: ROOT/PARTICIPANT/SUPPORT_SESSION/COMMAND/images and
ROOT/PARTICIPANT/QUERY_SESSION/COMMAND/images. Command names are arbitrary.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from common.config import CKPT_DIR, RESULTS_DIR, SEED
from common.data import IMAGE_SUFFIXES
from common.utils import load_portable_state_dict, set_seed
from proto_net import TRANSFORM, ProtoNet, device, logits_for_episode

BACKBONE = "tf_efficientnetv2_s.in21k"


def images(folder):
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def common_commands(participant, support_session, query_session):
    support, query = participant / support_session, participant / query_session
    return sorted(
        d.name
        for d in support.iterdir()
        if d.is_dir()
        and (query / d.name).is_dir()
        and images(d)
        and images(query / d.name)
    )


def sample(participant, support_session, query_session, n_way, k_shot, queries):
    commands = common_commands(participant, support_session, query_session)
    if len(commands) < n_way:
        return None
    commands = random.sample(commands, n_way)
    sx = []
    qx = []
    sy = []
    qy = []
    for label, command in enumerate(commands):
        support = images(participant / support_session / command)
        query = images(participant / query_session / command)
        support = (
            random.choices(support, k=k_shot)
            if len(support) < k_shot
            else random.sample(support, k_shot)
        )
        query = (
            random.choices(query, k=queries)
            if len(query) < queries
            else random.sample(query, queries)
        )
        sx += [TRANSFORM(Image.open(path).convert("RGB")) for path in support]
        sy += [label] * k_shot
        qx += [TRANSFORM(Image.open(path).convert("RGB")) for path in query]
        qy += [label] * queries
    return (
        torch.stack(sx),
        torch.tensor(sy),
        torch.stack(qx),
        torch.tensor(qy),
        commands,
    )


@torch.no_grad()
def assess(model, participant, args):
    scores = []
    latencies = []
    for _ in range(args.episodes):
        data = sample(
            participant,
            args.support_session,
            args.query_session,
            args.n_way,
            args.k_shot,
            args.queries,
        )
        if data is None:
            raise ValueError(
                f"{participant.name} has fewer than {args.n_way} commands common to both sessions"
            )
        sx, sy, qx, qy, _ = data
        sx, sy, qx, qy = (x.to(device) for x in (sx, sy, qx, qy))
        logits, latency = logits_for_episode(model, sx, sy, qx, args.n_way)
        scores.append((logits.argmax(1) == qy).float().mean().item() * 100)
        latencies.append(latency * 1000)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(latencies))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--support-session", default="session_1")
    parser.add_argument("--query-session", default="session_2")
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=10)
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CKPT_DIR / f"protonet_{BACKBONE}_5way_10shot_best.pt",
    )
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    set_seed(SEED)
    model = ProtoNet(BACKBONE).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_portable_state_dict(
        model, state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.eval()
    rows = []
    for participant in sorted(p for p in args.dataset_root.iterdir() if p.is_dir()):
        mean, std, latency = assess(model, participant, args)
        rows.append(
            {
                "participant": participant.name,
                "n_way": args.n_way,
                "k_shot": args.k_shot,
                "support_session": args.support_session,
                "query_session": args.query_session,
                "acc_mean": mean,
                "acc_std": std,
                "inf_ms_mean": latency,
                "n_episodes": args.episodes,
            }
        )
        print(f"{participant.name}: {mean:.2f} +/- {std:.2f}, {latency:.2f} ms/query")
    if not rows:
        raise ValueError(f"No participant folders in {args.dataset_root}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "atypical_anatomy_personalized_results.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
