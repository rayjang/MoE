import argparse
from pathlib import Path

import torch

from moe_pipeline.config import TrainingConfig
from moe_pipeline.expert_selection import load_router_statistics, select_experts_to_clone
from moe_pipeline.modeling import load_base_model, prepare_model_with_new_experts
from moe_pipeline.training import dataloader_from_path, run_phase, save_checkpoint, upload_to_hub


def parse_args():
    parser = argparse.ArgumentParser(description="Train additional experts for GPT-OSS-20B")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON file with TrainingConfig overrides")
    parser.add_argument("--hub_repo", type=str, default=None)
    parser.add_argument("--hub_token", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path | None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()
    import json
    from dataclasses import asdict

    base_cfg = TrainingConfig()
    with path.open() as f:
        overrides = json.load(f)
    cfg_dict = asdict(base_cfg)
    cfg_dict.update(overrides)
    return TrainingConfig(**cfg_dict)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    tokenizer, model = load_base_model(cfg.base_model_name)
    stats = load_router_statistics(cfg.selection.stats_path)
    selected = select_experts_to_clone(stats, cfg.selection)
    model = prepare_model_with_new_experts(model, selected, copies_per_expert=1)

    dataloader = dataloader_from_path(tokenizer, cfg.dataset_path, cfg.max_length, cfg.batch_size)
    for phase in cfg.phases:
        run_phase(model, phase, dataloader, cfg, cfg.fsdp)

    save_checkpoint(model, cfg.output_dir, tokenizer)
    if args.hub_repo:
        upload_to_hub(cfg.output_dir, repo_id=args.hub_repo, token=args.hub_token, private=True)


if __name__ == "__main__":
    main()
