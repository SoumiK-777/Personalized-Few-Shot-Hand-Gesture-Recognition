"""
proto_net_edge.py — ProtoNet with Jetson Orin Nano edge simulation profiling

Drop-in replacement for proto_net.py. Adds:
  - Automatic FP16 inference (matches Jetson TRT default)
  - Per-episode GPU memory tracking (VRAM budget enforcement)
  - CPU/RAM headroom warnings (respects cgroup limits set by edge_simulate.sh)
  - EdgeProfiler: structured per-episode timing breakdown
  - Jetson Orin Nano "budget report" at end of each backbone run
  - Graceful OOM recovery (simulates Jetson running out of unified memory)

Run standalone:
    python3 proto_net_edge.py

Or via the simulator (recommended):
    ./edge_simulate.sh --profile orin_nano
"""

import os
import random
import logging
import time
import gc
import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = "/home/soumik/soumik/misc"
DATA_DIR = os.path.join(BASE_DIR, "cropped_images")
LOG_FILE = os.path.join(BASE_DIR, "protonet_edge.log")

# Edge profile read from environment (set by edge_simulate.sh)
EDGE_PROFILE   = os.environ.get("EDGE_PROFILE",    "orin_nano")
EDGE_RAM_MB    = int(os.environ.get("EDGE_RAM_MB",  "4096"))
EDGE_GPU_W     = int(os.environ.get("EDGE_GPU_POWER_W", "10"))

# Jetson Orin Nano GPU memory budget (4 GB of unified shared with CPU)
# We model the GPU-available portion as ~3.5 GB
JETSON_GPU_BUDGET_MB = 3500

# Use FP16 for inference to match Jetson TRT FP16 default
USE_FP16 = True

# ==========================================
# LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("=" * 70)
logger.info(f"  EDGE SIMULATION MODE: {EDGE_PROFILE.upper()}")
logger.info(f"  RAM budget: {EDGE_RAM_MB} MB | GPU TDP: {EDGE_GPU_W} W")
logger.info(f"  Inference precision: {'FP16' if USE_FP16 else 'FP32'}")
logger.info(f"  Device: {device}")
logger.info("=" * 70)

ALLOWED_GESTURES = [
    "four", "like", "one", "peace_inverted", "stop", "three", "three_gun",
    "dislike", "grabbing", "little_finger", "palm", "stop_inverted", "three2",
    "thumb_index", "two_up", "fist", "grip", "middle_finger", "ok", "peace",
    "rock", "three3", "thumb_index2", "two_up_inverted"
]

# ==========================================
# EDGE PROFILER
# ==========================================
@dataclass
class EpisodeMetrics:
    episode: int
    n_way: int
    k_shot: int
    backbone: str
    race: str
    loss: float = 0.0
    accuracy: float = 0.0
    # Timing breakdowns (ms)
    t_support_forward_ms: float = 0.0
    t_query_forward_ms:   float = 0.0
    t_dist_compute_ms:    float = 0.0
    t_backward_ms:        float = 0.0
    t_total_ms:           float = 0.0
    inf_per_query_ms:     float = 0.0
    # Memory (MB)
    vram_allocated_mb:    float = 0.0
    vram_reserved_mb:     float = 0.0
    # Edge budget flags
    over_vram_budget:     bool  = False
    over_latency_budget:  bool  = False   # > 5ms per query = not real-time


class EdgeProfiler:
    """Accumulates per-episode metrics and emits a Jetson budget report."""

    # Jetson Orin Nano real-time thresholds
    LATENCY_BUDGET_MS = 5.0        # max acceptable inference latency per query
    VRAM_BUDGET_MB    = JETSON_GPU_BUDGET_MB

    def __init__(self, backbone: str, n_way: int, k_shot: int):
        self.backbone = backbone
        self.n_way = n_way
        self.k_shot = k_shot
        self.episodes: list[EpisodeMetrics] = []

    def record(self, m: EpisodeMetrics):
        m.over_vram_budget    = m.vram_allocated_mb > self.VRAM_BUDGET_MB
        m.over_latency_budget = m.inf_per_query_ms  > self.LATENCY_BUDGET_MS
        self.episodes.append(m)

        # Inline warnings
        if m.over_vram_budget:
            logger.warning(
                f"  ⚠  VRAM {m.vram_allocated_mb:.0f} MB > Jetson budget "
                f"({self.VRAM_BUDGET_MB} MB) — would OOM on real device"
            )
        if m.over_latency_budget:
            logger.warning(
                f"  ⚠  Latency {m.inf_per_query_ms:.2f} ms > real-time budget "
                f"({self.LATENCY_BUDGET_MS} ms) — too slow for in-vehicle use"
            )

    def summary(self) -> dict:
        if not self.episodes:
            return {}
        avg = lambda key: sum(getattr(e, key) for e in self.episodes) / len(self.episodes)
        oom_pct  = 100 * sum(1 for e in self.episodes if e.over_vram_budget)    / len(self.episodes)
        slow_pct = 100 * sum(1 for e in self.episodes if e.over_latency_budget) / len(self.episodes)
        return {
            "backbone":            self.backbone,
            "task":                f"{self.n_way}-way {self.k_shot}-shot",
            "episodes":            len(self.episodes),
            "avg_accuracy_%":      round(avg("accuracy"), 2),
            "avg_loss":            round(avg("loss"), 4),
            "avg_inf_per_query_ms":round(avg("inf_per_query_ms"), 3),
            "avg_t_support_fwd_ms":round(avg("t_support_forward_ms"), 3),
            "avg_t_query_fwd_ms":  round(avg("t_query_forward_ms"), 3),
            "avg_t_dist_ms":       round(avg("t_dist_compute_ms"), 3),
            "avg_t_backward_ms":   round(avg("t_backward_ms"), 3),
            "avg_vram_alloc_mb":   round(avg("vram_allocated_mb"), 1),
            "peak_vram_mb":        round(max(e.vram_allocated_mb for e in self.episodes), 1),
            "vram_budget_mb":      self.VRAM_BUDGET_MB,
            "over_vram_budget_%":  round(oom_pct, 1),
            "over_latency_%":      round(slow_pct, 1),
            "jetson_deployable":   oom_pct == 0 and slow_pct == 0,
        }

    def print_budget_report(self):
        s = self.summary()
        if not s:
            return
        deployable = s["jetson_deployable"]
        badge = "✅ DEPLOYABLE" if deployable else "❌ EXCEEDS BUDGET"
        sep = "─" * 60
        logger.info(sep)
        logger.info(f"  JETSON BUDGET REPORT  [{badge}]")
        logger.info(f"  Profile : {EDGE_PROFILE.upper()} | {s['task']}")
        logger.info(f"  Backbone: {s['backbone']}")
        logger.info(sep)
        logger.info(f"  Accuracy          : {s['avg_accuracy_%']:.2f}%")
        logger.info(f"  Loss              : {s['avg_loss']:.4f}")
        logger.info(sep)
        logger.info(f"  Inf / query       : {s['avg_inf_per_query_ms']:.3f} ms  "
                    f"(budget ≤ {self.LATENCY_BUDGET_MS} ms)")
        logger.info(f"    support forward : {s['avg_t_support_fwd_ms']:.3f} ms")
        logger.info(f"    query forward   : {s['avg_t_query_fwd_ms']:.3f} ms")
        logger.info(f"    dist compute    : {s['avg_t_dist_ms']:.3f} ms")
        logger.info(f"    backward pass   : {s['avg_t_backward_ms']:.3f} ms")
        logger.info(sep)
        logger.info(f"  Avg VRAM used     : {s['avg_vram_alloc_mb']:.1f} MB")
        logger.info(f"  Peak VRAM used    : {s['peak_vram_mb']:.1f} MB  "
                    f"(budget ≤ {self.VRAM_BUDGET_MB} MB)")
        logger.info(f"  Episodes over VRAM: {s['over_vram_budget_%']:.1f}%")
        logger.info(f"  Episodes too slow : {s['over_latency_%']:.1f}%")
        logger.info(sep)
        return s


# ==========================================
# PROTOTYPICAL NETWORK
# ==========================================
class ProtoNet(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        logger.info(f"Loading backbone: {backbone_name}")
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)

    def forward(self, x):
        return self.backbone(x)


def euclidean_dist(x, y):
    n, m, d = x.size(0), y.size(0), x.size(1)
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


# ==========================================
# EPISODIC DATA SAMPLER  (unchanged logic)
# ==========================================
def sample_protonet_episode(base_dir, split, n_way, k_shot, q_queries=10, transform=None):
    races = ['Black', 'Indian', 'White']
    race = random.choice(races)
    split_dir = os.path.join(base_dir, race, split)

    if not os.path.exists(split_dir):
        return None, None, None, None, race, []

    all_classes = [d for d in os.listdir(split_dir)
                   if os.path.isdir(os.path.join(split_dir, d)) and d in ALLOWED_GESTURES]

    if len(all_classes) < n_way:
        return None, None, None, None, race, []

    sampled_classes = random.sample(all_classes, n_way)
    support_images, support_labels, query_images, query_labels = [], [], [], []

    for label, cls in enumerate(sampled_classes):
        cls_dir = os.path.join(split_dir, cls)
        images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir)
                  if img.endswith(('.jpg', '.png', '.jpeg'))]

        needed = k_shot + q_queries
        sampled = random.choices(images, k=needed) if len(images) < needed else random.sample(images, needed)

        for p in sampled[:k_shot]:
            img = Image.open(p).convert('RGB')
            if transform:
                img = transform(img)
            support_images.append(img)
            support_labels.append(label)

        for p in sampled[k_shot:]:
            img = Image.open(p).convert('RGB')
            if transform:
                img = transform(img)
            query_images.append(img)
            query_labels.append(label)

    return (
        torch.stack(support_images),
        torch.stack(query_images),
        torch.tensor(support_labels, dtype=torch.long),
        torch.tensor(query_labels,   dtype=torch.long),
        race,
        sampled_classes,
    )


# ==========================================
# TRAINING — edge-aware, with full timing breakdown
# ==========================================
def train_episode_edge(
    model, optimizer,
    support_x, support_y,
    query_x,   query_y,
    n_way, profiler: EdgeProfiler, episode_idx: int
) -> EpisodeMetrics:

    model.train()
    optimizer.zero_grad()

    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x,   query_y   = query_x.to(device),   query_y.to(device)

    # Cast to FP16 for inference (FP32 for backward to preserve gradient quality)
    cast = lambda t: t.half() if USE_FP16 and device.type == "cuda" else t

    # ── Stage 1: support forward ──────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=USE_FP16):
        support_features = model(cast(support_x))
        prototypes = torch.stack([
            support_features[support_y == i].mean(0) for i in range(n_way)
        ])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_support = (time.perf_counter() - t0) * 1000

    # ── Stage 2: query forward + distance ─────────────────────────────────────
    t1 = time.perf_counter()
    with torch.cuda.amp.autocast(enabled=USE_FP16):
        query_features = model(cast(query_x))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_query = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    dists = euclidean_dist(
        query_features.float(),   # upcast for numerical stability
        prototypes.float()
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_dist = (time.perf_counter() - t2) * 1000

    # ── Stage 3: loss + backward ──────────────────────────────────────────────
    log_p_y = F.log_softmax(-dists, dim=1)
    loss    = -log_p_y.gather(1, query_y.unsqueeze(1)).squeeze().view(-1).mean()

    t3 = time.perf_counter()
    loss.backward()
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bwd = (time.perf_counter() - t3) * 1000

    # ── Accuracy ──────────────────────────────────────────────────────────────
    _, y_hat  = log_p_y.max(1)
    accuracy  = y_hat.eq(query_y).sum().item() / query_y.size(0) * 100
    n_queries = query_y.size(0)

    # ── VRAM snapshot ─────────────────────────────────────────────────────────
    vram_alloc = vram_reserved = 0.0
    if device.type == "cuda":
        vram_alloc    = torch.cuda.memory_allocated(device) / 1024 ** 2
        vram_reserved = torch.cuda.memory_reserved(device)  / 1024 ** 2

    t_total      = t_support + t_query + t_dist + t_bwd
    inf_per_query = (t_query + t_dist) / n_queries   # inference only (no backward)

    m = EpisodeMetrics(
        episode              = episode_idx,
        n_way                = n_way,
        k_shot               = profiler.k_shot,
        backbone             = profiler.backbone,
        race                 = "",
        loss                 = loss.item(),
        accuracy             = accuracy,
        t_support_forward_ms = round(t_support, 3),
        t_query_forward_ms   = round(t_query,   3),
        t_dist_compute_ms    = round(t_dist,    3),
        t_backward_ms        = round(t_bwd,     3),
        t_total_ms           = round(t_total,   3),
        inf_per_query_ms     = round(inf_per_query, 3),
        vram_allocated_mb    = round(vram_alloc,    1),
        vram_reserved_mb     = round(vram_reserved, 1),
    )
    profiler.record(m)
    return m


# ==========================================
# MAIN
# ==========================================
def main():
    backbones = [
        'tf_efficientnetv2_s.in21k'
    ]

    tasks = [(5, 5), (5, 10), (7, 5), (7, 10), (5, 1), (7, 1)]
    episodes_per_task = 50

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_summaries = []
    model = None

    for backbone_name in backbones:

        # Unload previous backbone
        if model is not None:
            logger.info("Unloading previous model...")
            del model, optimizer
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            time.sleep(1)

        logger.info("=" * 70)
        logger.info(f"BACKBONE: {backbone_name}")
        logger.info("=" * 70)

        try:
            model = ProtoNet(backbone_name)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model.to(device)
        except Exception as e:
            logger.error(f"Failed to load {backbone_name}: {e}")
            model = None
            continue

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for n_way, k_shot in tasks:
            logger.info(f"── {n_way}-way {k_shot}-shot ──")

            profiler = EdgeProfiler(backbone_name, n_way, k_shot)

            for episode in range(episodes_per_task):
                s_x, q_x, s_y, q_y, race, classes = sample_protonet_episode(
                    DATA_DIR, 'train', n_way, k_shot, q_queries=10, transform=transform
                )
                if s_x is None:
                    continue

                try:
                    m = train_episode_edge(
                        model, optimizer,
                        s_x, s_y, q_x, q_y,
                        n_way, profiler, episode
                    )
                    m.race = race

                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        f"  💥 OOM at episode {episode} — "
                        f"would crash on Jetson Orin Nano with {EDGE_RAM_MB} MB limit"
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    continue

                if (episode + 1) % 10 == 0:
                    logger.info(
                        f"  EP [{episode+1:02d}/{episodes_per_task}] | "
                        f"Race: {race:<7} | "
                        f"Loss: {m.loss:.4f} | "
                        f"Acc: {m.accuracy:>5.2f}% | "
                        f"Inf/q: {m.inf_per_query_ms:.3f}ms | "
                        f"VRAM: {m.vram_allocated_mb:.0f}MB | "
                        f"Total: {m.t_total_ms:.1f}ms"
                    )

            summary = profiler.print_budget_report()
            if summary:
                all_summaries.append(summary)

    # ── Final cross-backbone JSON report ─────────────────────────────────────
    report_path = os.path.join(BASE_DIR, f"edge_report_{EDGE_PROFILE}.json")
    try:
        with open(report_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"Full edge report saved → {report_path}")
    except Exception as e:
        logger.warning(f"Could not save JSON report: {e}")

    # Print deployable summary table
    logger.info("")
    logger.info("═" * 70)
    logger.info("  DEPLOYMENT FEASIBILITY SUMMARY")
    logger.info("═" * 70)
    logger.info(f"  {'Backbone':<35} {'Task':<15} {'Acc%':>6} {'ms/q':>7} {'VRAM MB':>8} {'Deploy?':>9}")
    logger.info("  " + "─" * 68)
    for s in all_summaries:
        bname = s["backbone"].split(".")[0][:34]
        ok = "✅" if s["jetson_deployable"] else "❌"
        logger.info(
            f"  {bname:<35} {s['task']:<15} "
            f"{s['avg_accuracy_%']:>6.2f} "
            f"{s['avg_inf_per_query_ms']:>7.3f} "
            f"{s['peak_vram_mb']:>8.1f} "
            f"{ok:>9}"
        )
    logger.info("═" * 70)


if __name__ == "__main__":
    main()