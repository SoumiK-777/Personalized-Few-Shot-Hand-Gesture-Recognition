import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from common.config import CKPT_DIR, DATA_DIR, FIGURES_DIR
from common.data import fixed_gesture_episode
from proto_net import TRANSFORM, ProtoNet, device, logits_for_episode

GESTURES = ["four", "like", "one", "peace_inverted", "stop", "three", "three_gun"]


def main():
    output = FIGURES_DIR / "confusion_matrix.png"
    if output.exists():
        print(f"completed; skipping {output}")
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = ProtoNet("tf_efficientnetv2_s.in21k").to(device)
    s = torch.load(
        CKPT_DIR / "protonet_tf_efficientnetv2_s.in21k.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(s)
    model.eval()
    truth = []
    pred = []
    with torch.no_grad():
        for _ in range(200):
            sx, qx, sy, qy, *_ = fixed_gesture_episode(
                DATA_DIR, "test", GESTURES, 10, 10, TRANSFORM
            )
            if sx is None:
                continue
            l, _ = logits_for_episode(
                model, sx.to(device), sy.to(device), qx.to(device), 7
            )
            truth += qy.tolist()
            pred += l.argmax(1).cpu().tolist()
    cm = confusion_matrix(truth, pred, labels=range(7), normalize="true")
    sns.heatmap(
        cm,
        xticklabels=GESTURES,
        yticklabels=GESTURES,
        annot=True,
        fmt=".2f",
        cmap="Blues",
    )
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
