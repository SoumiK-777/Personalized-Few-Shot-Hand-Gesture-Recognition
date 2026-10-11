"""Evaluate the 24-way supervised EfficientNet on held-out base-class images."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from common.config import ALLOWED_GESTURES, CKPT_DIR, DATA_DIR, RACES, RESULTS_DIR
from common.data import available_classes, class_files

BACKBONE = "tf_efficientnetv2_s.in21k"
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class BaseTestDataset(Dataset):
    def __init__(self, root, classes, race=None):
        self.samples = []
        labels = {name: index for index, name in enumerate(classes)}
        for group in (race,) if race else RACES:
            for gesture in available_classes(root, "test", group, classes):
                self.samples.extend((path, labels[gesture]) for path in class_files(root, group, "test", gesture))
        if not self.samples:
            description = race if race else "all demographic groups"
            raise RuntimeError(f"No base-class test images found for {description}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return TRANSFORM(Image.open(path).convert("RGB")), label


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                        pin_memory=torch.cuda.is_available())
    correct = total = 0
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        correct += int((predictions == labels).sum())
        total += len(labels)
    return {"correct": correct, "images": total, "accuracy": 100 * correct / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=CKPT_DIR / "supervised_effnetv2s_base24_best.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / "supervised_effnetv2s_base24_test.json")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if state.get("backbone") != BACKBONE or state.get("classes") != list(ALLOWED_GESTURES):
        raise ValueError("Checkpoint does not match the expected 24-class EfficientNetV2-S training run.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(ALLOWED_GESTURES)).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    report = {
        "checkpoint": str(args.checkpoint),
        "backbone": BACKBONE,
        "classes": list(ALLOWED_GESTURES),
        "split": "test",
        "overall": evaluate(model, BaseTestDataset(DATA_DIR, ALLOWED_GESTURES), device,
                            args.batch_size, args.workers),
        "by_demographic_group": {},
    }
    for race in RACES:
        report["by_demographic_group"][race] = evaluate(
            model, BaseTestDataset(DATA_DIR, ALLOWED_GESTURES, race), device,
            args.batch_size, args.workers
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"24-way held-out base-class accuracy: {report['overall']['accuracy']:.2f}% "
          f"({report['overall']['correct']}/{report['overall']['images']})")
    for race, metrics in report["by_demographic_group"].items():
        print(f"{race}: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['images']})")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
