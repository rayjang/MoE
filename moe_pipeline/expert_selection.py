import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .config import ExpertSelectionConfig


def load_router_statistics(stats_path: Path) -> Dict[str, List[float]]:
    """Load router usage/entropy stats.

    The JSON is expected to contain keys like "usage" and "entropy" per expert.
    Example structure:
    {
        "usage": [0.10, 0.07, ...],
        "entropy": [1.2, 1.1, ...],
        "loss_delta": [0.0, -0.01, ...]
    }
    """

    if not stats_path.exists():
        raise FileNotFoundError(f"Router statistics not found at {stats_path}")

    with stats_path.open() as f:
        stats = json.load(f)

    if "usage" not in stats:
        raise ValueError("Expected 'usage' key in stats JSON")
    return stats


def softmax(values: List[float], temperature: float = 1.0) -> List[float]:
    scaled = [v / max(temperature, 1e-5) for v in values]
    max_v = max(scaled)
    exps = [math.exp(v - max_v) for v in scaled]
    denom = sum(exps)
    return [v / denom for v in exps]


def score_experts(stats: Dict[str, List[float]], metric: str = "usage") -> List[float]:
    if metric not in stats:
        raise ValueError(f"Metric {metric} not present in stats: {list(stats.keys())}")
    return stats[metric]


def estimate_num_new_experts(stats: Dict[str, List[float]], cfg: ExpertSelectionConfig) -> int:
    usage = stats["usage"]
    entropy = stats.get("entropy", [1.0 for _ in usage])
    imbalance = max(usage) - min(usage)
    entropy_score = sum(entropy) / len(entropy)

    base_count = cfg.num_new_experts or int(len(usage) * imbalance + 1)
    if entropy_score < cfg.min_entropy:
        base_count = min(base_count + 1, cfg.max_new_experts)

    return min(max(base_count, 1), cfg.max_new_experts)


def select_experts_to_clone(stats: Dict[str, List[float]], cfg: ExpertSelectionConfig) -> List[int]:
    usage_scores = score_experts(stats, cfg.metric)
    entropy = stats.get("entropy", [1.0 for _ in usage_scores])

    candidates = [
        (idx, u)
        for idx, (u, e) in enumerate(zip(usage_scores, entropy))
        if u >= cfg.min_usage and e >= cfg.min_entropy
    ]

    if not candidates:
        raise RuntimeError("No experts met the selection criteria. Lower thresholds or inspect stats.")

    num_new = estimate_num_new_experts(stats, cfg)
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    if cfg.tie_break_random:
        top_usage = candidates[: num_new * 2]
        rng = random.Random(0)
        rng.shuffle(top_usage)
        chosen = sorted(top_usage[:num_new], key=lambda x: x[0])
        return [idx for idx, _ in chosen]

    return [idx for idx, _ in candidates[:num_new]]


def duplicate_expert_weights(expert: torch.nn.Module) -> torch.nn.Module:
    """Deep copy expert parameters for cloning."""

    cloned = type(expert)()
    cloned.load_state_dict(expert.state_dict())
    return cloned


def plan_expert_cloning(
    model: torch.nn.Module, selected_experts: List[int], copies_per_expert: int = 1
) -> List[Tuple[int, torch.nn.Module]]:
    """Return a list of (insert_position, new_expert_module)."""

    planned_clones: List[Tuple[int, torch.nn.Module]] = []
    for idx in selected_experts:
        expert_module = model.model.moe.experts[idx]
        for _ in range(copies_per_expert):
            planned_clones.append((idx, duplicate_expert_weights(expert_module)))
    return planned_clones
