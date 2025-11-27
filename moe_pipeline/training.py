import json
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
)
from transformers import (
    AutoTokenizer,
    default_data_collator,
    get_cosine_schedule_with_warmup,
)
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from .config import FsdpConfig, TrainingConfig, TrainingPhaseConfig
from .modeling import freeze_existing_experts, unfreeze_layernorm_and_router


def resolve_sharding_strategy(name: str) -> ShardingStrategy:
    mapping = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    }
    return mapping.get(name.upper(), ShardingStrategy.FULL_SHARD)


def build_fsdp_model(model: torch.nn.Module, cfg: FsdpConfig):
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={MixtralDecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cfg.cpu_offload,
        mixed_precision=cfg.mixed_precision,
        sharding_strategy=resolve_sharding_strategy(cfg.sharding_strategy),
        sync_module_states=cfg.sync_module_states,
    )
    return model


def configure_optimizer(model: torch.nn.Module, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)


def dataloader_from_path(tokenizer: AutoTokenizer, dataset_path: str, max_length: int, batch_size: int):
    with open(dataset_path) as f:
        data = [json.loads(line) for line in f]

    inputs = tokenizer(
        [sample["text"] for sample in data],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = [{"input_ids": ids, "attention_mask": mask, "labels": ids} for ids, mask in zip(inputs.input_ids, inputs.attention_mask)]

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
        shuffle=True,
    )


def run_phase(
    model: torch.nn.Module,
    phase: TrainingPhaseConfig,
    dataloader: Iterable,
    cfg: TrainingConfig,
    fsdp_cfg: FsdpConfig,
):
    if phase.freeze_existing_experts:
        freeze_existing_experts(model, freeze=True)
    else:
        freeze_existing_experts(model, freeze=False)
    if phase.unfreeze_layernorm:
        unfreeze_layernorm_and_router(model)
    if phase.train_router_only:
        for name, p in model.named_parameters():
            if "router" not in name and "gate" not in name:
                p.requires_grad = False

    if fsdp_cfg.use_fsdp2:
        model = build_fsdp_model(model, fsdp_cfg)

    optimizer = configure_optimizer(model, phase.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps or len(dataloader) * phase.epochs,
    )

    model.train()
    step = 0
    for epoch in range(phase.epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % cfg.log_every == 0:
                print(f"[{phase.name}] step {step} loss={loss.item():.4f}")
            if cfg.max_steps and step >= cfg.max_steps:
                return
            step += 1


@torch.no_grad()
def save_checkpoint(model: torch.nn.Module, output_dir: Path, tokenizer: AutoTokenizer):
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    if isinstance(model, FSDP):
        model_state = model.state_dict()
    else:
        model_state = model.state_dict()
    torch.save(model_state, output_dir / "pytorch_model.bin")


@torch.no_grad()
def upload_to_hub(output_dir: Path, repo_id: str, token: Optional[str] = None, private: bool = True):
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id)
