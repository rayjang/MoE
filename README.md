# MoE

This repository provides a reference pipeline for extending **openai/gpt-oss-20b** with additional experts, training the new experts under an FSDP2 (or FSDP) setup, and optionally uploading the resulting checkpoint to the Hugging Face Hub.

## Key components
- **Expert selection** (`moe_pipeline/expert_selection.py`): loads router statistics, estimates how many new experts to add, and selects which existing experts to clone based on usage/entropy thresholds.
- **Model surgery** (`moe_pipeline/modeling.py`): loads the base GPT-OSS 20B model, clones the selected experts, biases the router toward existing experts to avoid collapse, and freezes/unfreezes modules as needed.
- **Training loop with FSDP2** (`moe_pipeline/training.py`): runs progressive training phases (freeze old experts → router alignment → progressive unfreeze) with AdamW, cosine schedule, and FSDP wrapping around decoder layers.
- **End-to-end script** (`moe_pipeline/scripts/train_new_experts.py`): CLI that ties together selection, cloning, training, checkpoint saving, and optional Hub upload.

## Usage
1. Generate router statistics (usage, entropy, optional loss deltas) into `sample_data/router_stats.json` (a tiny example is provided).
2. Prepare a JSONL dataset with `{"text": "..."}` entries for continual pretraining the new experts (a tiny sample lives at `sample_data/science_tiny.jsonl`).
3. Run the training script:

```bash
python -m moe_pipeline.scripts.train_new_experts \
  --config configs/train.json \
  --hub_repo your-user/gpt-oss-20b-new-experts \
  --hub_token $HF_TOKEN
```

The default configuration (see `moe_pipeline/config.py`) performs:
- Expert selection based on usage ≥ 2% and entropy ≥ 0.5, with the number of clones inferred from imbalance.
- Phase 1: freeze old experts, train new experts + router with LayerNorms unfrozen.
- Phase 2: router-only alignment on mixed data.
- Phase 3: progressive unfreeze for joint tuning.
- FSDP2 wrapping of Mixtral decoder layers with optional CPU offload and mixed precision.

### Quick smoke test

To verify the end-to-end code path without large downloads, point the config to the bundled sample files (already set as defaults) and run:

```
python -m moe_pipeline.scripts.train_new_experts --hub_repo ''
```

This will load `openai/gpt-oss-20b`, clone the experts indicated by `sample_data/router_stats.json`, perform continual pretraining on the JSONL lines in `sample_data/science_tiny.jsonl` (using the `text` key), and write a checkpoint to `outputs/new_experts`.

Adjust thresholds, number of new experts, sharding strategy, and learning rates via the config file.
