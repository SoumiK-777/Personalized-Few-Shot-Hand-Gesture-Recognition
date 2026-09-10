"""Measure the selected ProtoNet model on the machine where it will be deployed.

Run this script on the Jetson Nano (or other target) for meaningful latency numbers.
It reports backbone parameters, optional FLOPs, and synchronized inference latency.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from common.config import CKPT_DIR, RESULTS_DIR
from common.utils import load_portable_state_dict, set_seed
from proto_net import ProtoNet

BACKBONE = "tf_efficientnetv2_s.in21k"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CKPT_DIR / f"protonet_{BACKBONE}_5way_10shot_best.pt",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProtoNet(BACKBONE).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_portable_state_dict(
        model, state["model_state_dict"] if "model_state_dict" in state else state
    )
    model.eval()
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    if dtype == torch.float16:
        model.half()
    image = torch.randn(1, 3, 224, 224, device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(image)
        sync()
        samples = []
        for _ in range(args.runs):
            sync()
            start = time.perf_counter()
            model(image)
            sync()
            samples.append((time.perf_counter() - start) * 1000)
    flops = None
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = int(
            FlopCountAnalysis(model.float().cpu(), torch.zeros(1, 3, 224, 224)).total()
        )
        model.to(device)
    except Exception as error:
        print(f"FLOPs unavailable (optional fvcore dependency): {error}")
    report = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "precision": "fp16" if dtype == torch.float16 else "fp32",
        "checkpoint": str(args.checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()),
        "flops_per_224px_image": flops,
        "latency_ms_mean": float(np.mean(samples)),
        "latency_ms_p50": float(np.percentile(samples, 50)),
        "latency_ms_p95": float(np.percentile(samples, 95)),
        "throughput_images_per_second": float(1000 / np.mean(samples)),
        "runs": args.runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "edge_benchmark_protonet_effnetv2s_5way10shot.json"
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
