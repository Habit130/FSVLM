# FSVLM Linux 4090 Runbook

## Scope
This delivery targets a Linux server with:
- 1x RTX 4090
- CUDA 11.8
- Python 3.10
- Miniconda

The repository assumes the dataset lives next to the repository root:
- repository: `FSVLM/`
- dataset: `../plantseg/`

## Required external assets
- Llama 2 7B base model on Hugging Face
- `liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview`
- `openai/clip-vit-large-patch14`
- SAM ViT-H checkpoint file

The dataset must keep this structure untouched:
- `plantseg/main.json`
- `plantseg/images/...`
- `plantseg/ann/...`

## Dataset contract
- Training uses `split=train`
- Validation uses `split=val`
- Final testing uses `split=test`
- Text prompt source is fixed to `caption[3]`
- Masks are treated as binary foreground/background
- The conversation template remains the repository template with `[SEG]`

## Environment artifact
Use `environment.linux-4090.yml` as the server environment source of truth for this delivery.

## Training entry
The supported server-side training command surface is:

```bash
python train_ds.py \
  --dataset_dir ../plantseg \
  --train_split train \
  --val_split val \
  --caption_index 3 \
  --target_name "diseased region" \
  --version liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview \
  --model_base meta-llama/Llama-2-7b-hf \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth
```

## Post-training merge
The supported weight merge command surface is:

```bash
python merge_lora_weights_and_save_hf_model.py \
  --version liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview \
  --model_base meta-llama/Llama-2-7b-hf \
  --weight /path/to/pytorch_model.bin \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth \
  --save_path /path/to/merged_model
```

## Final test entry
The supported server-side final test command surface is:

```bash
python test.py \
  --base_dir ../plantseg \
  --split test \
  --caption_index 3 \
  --target_name "diseased region" \
  --version /path/to/merged_model \
  --vision_pretrained /path/to/sam_vit_h_4b8939.pth
```

## Output contract
- Validation metrics are written to `runs/<exp_name>/val_metrics.csv`
- Final test metrics are written to `vis_output/test_metrics.csv` by default
- Final reported metrics are:
  - `IoU`
  - `Dice`
  - `Recall`
  - `mIoU`
  - `mACC`
