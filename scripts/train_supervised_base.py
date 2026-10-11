"""Train the matched non-episodic EfficientNetV2-S control on 24 base gestures.

The resulting representation is deliberately trained with ordinary 24-way
cross-entropy, not an episodic objective.  It is later adapted only with the
support images of novel held-out gestures.
"""

import argparse
import sys
from itertools import cycle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from common.config import ALLOWED_GESTURES, BACKBONES, CKPT_DIR, DATA_DIR, RACES
from common.data import available_classes, class_files
from common.utils import (
    load_portable_state_dict,
    maybe_data_parallel,
    portable_state_dict,
    set_seed,
)

BACKBONE = "tf_efficientnetv2_s.in21k"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class BaseGestureDataset(Dataset):
    def __init__(self, root, classes):
        self.samples = []
        index = {name: i for i, name in enumerate(classes)}
        root = Path(root)
        for race in RACES:
            for label in available_classes(root, "train", race, classes):
                paths = class_files(root, race, "train", label)
                self.samples.extend((p, index[label]) for p in paths)
        if not self.samples:
            raise RuntimeError("No base-class training images found; check DATA_DIR and split manifest.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return TRANSFORM(Image.open(path).convert("RGB")), label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000,
                        help="Cross-entropy updates (default matches ProtoNet episode count).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Total batch size across all GPUs (default: 16 images/GPU on two T4s).")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable CUDA mixed precision (uses substantially more GPU memory).")
    parser.add_argument("--checkpoint", type=Path,
                        default=CKPT_DIR / "supervised_effnetv2s_base24_best.pt")
    parser.add_argument("--save-every", type=int, default=50,
                        help="Write a resumable training state every N updates.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore a resumable state and start a fresh run.")
    args = parser.parse_args()
    if BACKBONE not in BACKBONES:
        raise RuntimeError(f"Expected {BACKBONE} in configured backbones")

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count:
        devices = ", ".join(torch.cuda.get_device_name(i) for i in range(gpu_count))
        print(f"Using {gpu_count} CUDA device(s): {devices}")
    else:
        print("CUDA unavailable; training on CPU")
    if gpu_count > 1 and args.batch_size < gpu_count:
        raise ValueError("--batch-size must be at least the number of visible GPUs.")
    use_amp = device.type == "cuda" and not args.no_amp
    print(f"total_batch_size={args.batch_size} per_gpu_batch≈{args.batch_size / max(gpu_count, 1):.1f} amp={use_amp}")
    dataset = BaseGestureDataset(DATA_DIR, ALLOWED_GESTURES)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=torch.cuda.is_available(), persistent_workers=args.workers > 0)
    model = maybe_data_parallel(
        timm.create_model(BACKBONE, pretrained=True, num_classes=len(ALLOWED_GESTURES)).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()
    latest_checkpoint = args.checkpoint.with_name(f"{args.checkpoint.stem}_latest{args.checkpoint.suffix}")
    start_step = 0
    if latest_checkpoint.exists() and not args.no_resume:
        state = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        if state.get("backbone") != BACKBONE or state.get("classes") != list(ALLOWED_GESTURES):
            raise ValueError(f"Refusing to resume incompatible checkpoint: {latest_checkpoint}")
        load_portable_state_dict(model, state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if "scaler_state_dict" in state:
            scaler.load_state_dict(state["scaler_state_dict"])
        start_step = int(state["step"])
        print(f"resuming {latest_checkpoint} at step {start_step}/{args.steps}")
    if start_step >= args.steps:
        print(f"completed; checkpoint already contains {start_step} updates: {args.checkpoint}")
        return

    stream = cycle(loader)
    model.train()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for step in range(start_step + 1, args.steps + 1):
        images, labels = next(stream)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(images.to(device, non_blocking=True)), labels.to(device, non_blocking=True))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(f"step={step}/{args.steps} loss={loss.item():.4f}")
        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model_state_dict": portable_state_dict(model),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "classes": list(ALLOWED_GESTURES), "backbone": BACKBONE,
                        "step": step, "objective": "24-way cross-entropy"}, latest_checkpoint)

    torch.save({"model_state_dict": portable_state_dict(model), "classes": list(ALLOWED_GESTURES),
                "backbone": BACKBONE, "steps": args.steps, "objective": "24-way cross-entropy"},
               args.checkpoint)
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()
