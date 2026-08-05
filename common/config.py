from __future__ import annotations

import os
from pathlib import Path

ROOT = Path("/home/soumik/soumik/misc")
DATA_DIR = ROOT / "cropped_images"
WORK_DIR = ROOT / "output"

RESULTS_DIR = WORK_DIR / "results"
CKPT_DIR = WORK_DIR / "checkpoints"
FIGURES_DIR = WORK_DIR / "figures"

for directory in (RESULTS_DIR, CKPT_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SEED = 42
RACES = ("Black", "Indian", "White")
ALLOWED_GESTURES = (
    "four", "like", "one", "peace_inverted", "stop", "three", "three_gun",
    "dislike", "grabbing", "little_finger", "palm", "stop_inverted", "three2",
    "thumb_index", "two_up", "fist", "grip", "middle_finger", "ok", "peace",
    "rock", "three3", "thumb_index2", "two_up_inverted",
)
BACKBONES = ("mobilenetv4_conv_small.e2400_r224_in1k", "tf_efficientnetv2_s.in21k", "wide_resnet50_2")
TASKS = ((5, 1), (5, 5), (5, 10), (7, 1), (7, 5), (7, 10))
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "1000"))
EVAL_EPISODES = int(os.environ.get("EVAL_EPISODES", "200"))
AUTOSAVE_EVERY = int(os.environ.get("AUTOSAVE_EVERY", "50"))
QUERY_PER_CLASS = int(os.environ.get("QUERY_PER_CLASS", "10"))