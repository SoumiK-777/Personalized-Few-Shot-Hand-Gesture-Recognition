from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image

from .config import ALLOWED_GESTURES, RACES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _files(folder: Path):
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


@lru_cache(maxsize=8)
def _manifest(base_text: str):
    path = Path(
        os.environ.get("SPLIT_MANIFEST_PATH", Path(base_text) / "split_manifest.json")
    )
    return json.loads(path.read_text()).get("splits", {}) if path.exists() else {}


def _class_files(base: Path, race: str, split: str, cls: str):
    folder = base / race / split / cls
    if folder.is_dir():
        return _files(folder)
    # Manifest support keeps Kaggle's read-only, unsplit inputs usable without copying images.
    entries = _manifest(str(base)).get(race, {}).get(cls, {}).get(split, [])
    return [base / entry for entry in entries]


def class_files(base_dir, race: str, split: str, cls: str):
    """Return image paths for one class, including manifest-backed datasets."""
    return _class_files(Path(base_dir), race, split, cls)


def available_classes(base_dir, split: str, race: str, allowed_gestures=None):
    """List non-empty classes for a split and demographic group.

    ``allowed_gestures`` defaults to the project's original 24 base gestures.
    Passing an explicit collection is used by the novel-class experiment, where
    the labels must not overlap with those base gestures.
    """
    allowed = set(ALLOWED_GESTURES if allowed_gestures is None else allowed_gestures)
    folder = Path(base_dir) / race / split
    if folder.is_dir():
        return sorted(
            d.name
            for d in folder.iterdir()
            if d.is_dir() and d.name in allowed and _files(d)
        )
    return sorted(
        cls
        for cls in _manifest(str(Path(base_dir))).get(race, {})
        if cls in allowed and _class_files(Path(base_dir), race, split, cls)
    )


def sample_episode(
    base_dir,
    split: str,
    n_way: int,
    k_shot: int,
    q_queries: int = 10,
    transform=None,
    race: str | None = None,
):
    """Return tensor episode plus its demographic and sampled gesture names.

    Sampling is within one race per episode, matching the paper's protocol. Images are
    sampled with replacement only when a class has fewer than K+Q images.
    """
    base = Path(base_dir)
    races = [race] if race else list(RACES)
    viable = [r for r in races if len(available_classes(base, split, r)) >= n_way]
    if not viable:
        return None, None, None, None, None, []
    chosen_race = random.choice(viable)
    classes = random.sample(available_classes(base, split, chosen_race), n_way)
    support_images, query_images, support_labels, query_labels = [], [], [], []
    for label, cls in enumerate(classes):
        images = _class_files(base, chosen_race, split, cls)
        selected = (
            random.choices(images, k=k_shot + q_queries)
            if len(images) < k_shot + q_queries
            else random.sample(images, k_shot + q_queries)
        )
        for path in selected[:k_shot]:
            image = Image.open(path).convert("RGB")
            support_images.append(transform(image) if transform else image)
            support_labels.append(label)
        for path in selected[k_shot:]:
            image = Image.open(path).convert("RGB")
            query_images.append(transform(image) if transform else image)
            query_labels.append(label)
    return (
        torch.stack(support_images),
        torch.stack(query_images),
        torch.tensor(support_labels),
        torch.tensor(query_labels),
        chosen_race,
        classes,
    )


def classes_in_dataset(base_dir, race: str):
    """Return every gesture label available for ``race`` in either split."""
    base = Path(base_dir)
    labels = set()
    for split in ("train", "test"):
        folder = base / race / split
        if folder.is_dir():
            labels.update(d.name for d in folder.iterdir() if d.is_dir() and _files(d))
        else:
            labels.update(_manifest(str(base)).get(race, {}).keys())
    return sorted(labels)


def sample_cross_split_episode(
    base_dir,
    support_split: str,
    query_split: str,
    n_way: int,
    k_shot: int,
    q_queries: int,
    transform,
    allowed_gestures,
):
    """Sample an episode with support and query images from different splits.

    This prevents a held-out class evaluation from using query images during
    calibration.  A single demographic group is used per episode, consistent
    with the existing episodic protocol.
    """
    base = Path(base_dir)
    allowed = tuple(allowed_gestures)
    viable = []
    for race in RACES:
        support = set(available_classes(base, support_split, race, allowed))
        query = set(available_classes(base, query_split, race, allowed))
        if len(support & query) >= n_way:
            viable.append((race, sorted(support & query)))
    if not viable:
        return None, None, None, None, None, []

    race, candidates = random.choice(viable)
    classes = random.sample(candidates, n_way)
    sx, qx, sy, qy = [], [], [], []
    for label, cls in enumerate(classes):
        support_images = _class_files(base, race, support_split, cls)
        query_images = _class_files(base, race, query_split, cls)
        selected_support = (
            random.choices(support_images, k=k_shot)
            if len(support_images) < k_shot
            else random.sample(support_images, k_shot)
        )
        selected_query = (
            random.choices(query_images, k=q_queries)
            if len(query_images) < q_queries
            else random.sample(query_images, q_queries)
        )
        sx.extend(transform(Image.open(p).convert("RGB")) for p in selected_support)
        qx.extend(transform(Image.open(p).convert("RGB")) for p in selected_query)
        sy.extend([label] * k_shot)
        qy.extend([label] * q_queries)
    return torch.stack(sx), torch.stack(qx), torch.tensor(sy), torch.tensor(qy), race, classes


def fixed_gesture_episode(base_dir, split, gestures, k_shot, q_queries, transform):
    """Episode sampler with a fixed class subset, used for a reproducible confusion matrix."""
    base = Path(base_dir)
    viable = [
        race
        for race in RACES
        if all(_class_files(base, race, split, g) for g in gestures)
    ]
    if not viable:
        return None, None, None, None, None, []
    race = random.choice(viable)
    sx, qx, sy, qy = [], [], [], []
    for label, gesture in enumerate(gestures):
        images = _class_files(base, race, split, gesture)
        chosen = (
            random.choices(images, k=k_shot + q_queries)
            if len(images) < k_shot + q_queries
            else random.sample(images, k_shot + q_queries)
        )
        for path in chosen[:k_shot]:
            sx.append(transform(Image.open(path).convert("RGB")))
            sy.append(label)
        for path in chosen[k_shot:]:
            qx.append(transform(Image.open(path).convert("RGB")))
            qy.append(label)
    return (
        torch.stack(sx),
        torch.stack(qx),
        torch.tensor(sy),
        torch.tensor(qy),
        race,
        list(gestures),
    )
