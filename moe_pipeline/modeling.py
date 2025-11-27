from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .expert_selection import plan_expert_cloning


def load_base_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    return tokenizer, model


def freeze_existing_experts(model: torch.nn.Module, freeze: bool = True):
    if not hasattr(model, "model") or not hasattr(model.model, "moe"):
        raise AttributeError("Model does not expose a 'model.moe' module for experts")

    for expert in model.model.moe.experts:
        for p in expert.parameters():
            p.requires_grad = not freeze


def unfreeze_layernorm_and_router(model: torch.nn.Module):
    for name, module in model.named_modules():
        if "layernorm" in name.lower() or "ln" in name.lower():
            for p in module.parameters():
                p.requires_grad = True
        if "router" in name.lower() or "gate" in name.lower():
            for p in module.parameters():
                p.requires_grad = True


def insert_new_experts(model: torch.nn.Module, clones) -> List[int]:
    """Insert cloned experts and return their indices."""

    moe_block = model.model.moe
    inserted_indices: List[int] = []
    for insert_idx, clone in clones:
        moe_block.experts.insert(insert_idx, clone)
        inserted_indices.append(insert_idx)
    moe_block.num_experts = len(moe_block.experts)
    return inserted_indices


def add_new_experts(model: torch.nn.Module, selected_experts: List[int], copies_per_expert: int = 1):
    clones = plan_expert_cloning(model, selected_experts, copies_per_expert=copies_per_expert)
    inserted_indices = insert_new_experts(model, clones)
    return inserted_indices


def mark_router_bias_to_old_experts(model: torch.nn.Module, bias: float = 1.0):
    router = model.model.moe.router
    with torch.no_grad():
        router_logits = router.gate.weight if hasattr(router, "gate") else router.weight
        num_old = router_logits.shape[-1]
        router_bias = torch.zeros_like(router_logits)
        router_bias[..., :num_old] = bias
        router_logits.add_(router_bias)


def prepare_model_with_new_experts(model: torch.nn.Module, selected: List[int], copies_per_expert: int):
    add_new_experts(model, selected, copies_per_expert)
    mark_router_bias_to_old_experts(model, bias=0.3)
    return model
