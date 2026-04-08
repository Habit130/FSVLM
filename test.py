import argparse
import csv
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig

from model.llava import conversation as conversation_lib
from utils.dataset import ValDataset, collate_fn, save_mask
from utils.model_loading import load_fsvlm_model
from utils.server_metrics import Evaluator
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, dict_to_cuda


def parse_args(args):
    parser = argparse.ArgumentParser(description="FSVLM test")
    parser.add_argument("--version", default="your weight")
    parser.add_argument(
        "--model_base",
        default="",
        type=str,
        help="Optional base model path when --version points to a LoRA preview repo.",
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--base_dir", default="../plantseg", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--caption_index", default=3, type=int)
    parser.add_argument("--target_name", default="diseased region", type=str)
    parser.add_argument("--metrics_path", default="./vis_output/test_metrics.csv", type=str)
    parser.add_argument(
        "--vision_pretrained",
        default="PATH_TO_SAM_ViT-H/sam_vit_h_4b8939.pth",
        type=str,
    )
    parser.add_argument(
        "--bridge_type",
        default="mlp",
        type=str,
        choices=["mlp", "query"],
    )
    parser.add_argument("--bridge_dim", default=256, type=int)
    parser.add_argument("--bridge_num_queries", default=4, type=int)
    parser.add_argument("--bridge_num_heads", default=8, type=int)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_llama_2",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--workers", default=2, type=int)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    pretrained_kwargs = {}
    if args.load_in_4bit:
        pretrained_kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        pretrained_kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    tokenizer, model = load_fsvlm_model(
        args.version,
        model_kwargs={
            "vision_tower": args.vision_tower,
            "seg_token_idx": 0,
            "vision_pretrained": args.vision_pretrained,
            "train_mask_decoder": True,
            "out_dim": 256,
            "use_mm_start_end": args.use_mm_start_end,
            "bridge_type": args.bridge_type,
            "bridge_dim": args.bridge_dim,
            "bridge_num_queries": args.bridge_num_queries,
            "bridge_num_heads": args.bridge_num_heads,
        },
        tokenizer_kwargs={
            "cache_dir": None,
            "model_max_length": args.model_max_length,
            "padding_side": "right",
            "use_fast": False,
        },
        model_base=args.model_base or None,
        torch_dtype=pretrained_kwargs.get("torch_dtype", torch_dtype),
        pretrained_kwargs=pretrained_kwargs,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.seg_token_idx = args.seg_token_idx
    model.resize_token_embeddings(len(tokenizer))

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif args.precision == "fp16" and not args.load_in_4bit and not args.load_in_8bit:
        model = model.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()
    else:
        model = model.cuda()

    model.eval()

    test_dataset = ValDataset(
        args.base_dir,
        tokenizer,
        args.vision_tower,
        split=args.split,
        image_size=args.image_size,
        caption_index=args.caption_index,
        target_name=args.target_name,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
    )

    evaluator = Evaluator(2)
    evaluator.reset()

    with torch.no_grad():
        for input_dict in test_loader:
            torch.cuda.empty_cache()
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            evaluator.add_batch(masks_list.cpu().numpy(), output_list.cpu().numpy())
            assert len(pred_masks) == 1
            save_mask(pred_masks, args.vis_save_path, input_dict["image_paths"][0])

    metrics = evaluator.compute_metrics()
    print(
        "IoU:{IoU:.4f}, Dice:{Dice:.4f}, Recall:{Recall:.4f}, mIoU:{mIoU:.4f}, mACC:{mACC:.4f}".format(
            **metrics
        )
    )

    os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
    with open(args.metrics_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["IoU", "Dice", "Recall", "mIoU", "mACC"])
        writer.writerow(
            [
                metrics["IoU"],
                metrics["Dice"],
                metrics["Recall"],
                metrics["mIoU"],
                metrics["mACC"],
            ]
        )


if __name__ == "__main__":
    main(sys.argv[1:])
