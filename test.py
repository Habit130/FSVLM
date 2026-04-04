import argparse
import os
import sys
import deepspeed
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPImageProcessor
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from model.FSVLM import FSVLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
import csv
from utils.model_loading import load_fsvlm_model
from utils.orgin_dataset import load_plantseg_records
from utils.server_metrics import Evaluator
from PIL import Image

def parse_args(args):
    parser = argparse.ArgumentParser(description="chat")
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
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--encode_type", default="clip-vit")
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
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
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {}
    if args.load_in_4bit:
        kwargs.update(
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
        kwargs.update(
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
        },
        tokenizer_kwargs={
            "cache_dir": None,
            "model_max_length": args.model_max_length,
            "padding_side": "right",
            "use_fast": False,
        },
        model_base=args.model_base or None,
        torch_dtype=kwargs.get("torch_dtype", torch_dtype),
        pretrained_kwargs=kwargs,
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
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    records = load_plantseg_records(
        args.base_dir,
        split=args.split,
        caption_index=args.caption_index,
    )

    with torch.no_grad():
        evaluator = Evaluator(2)
        evaluator.reset()
        for record in records:
            image_path = record["image_path"]
            mask_path = record["mask_path"]
            text = record["caption"]
            prompt = f" {text} Please segment the {args.target_name}"
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []
            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))
                continue

            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            image_clip = (
                clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )

            if args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (
                preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
            )
            if not os.path.exists(mask_path):
                continue
            mask = Image.open(mask_path)
            masks = np.array(mask)
            masks = masks[np.newaxis, :, :]
            masks = torch.from_numpy(masks)
            masks[masks != 0] = 1
            if args.precision == "bf16":
                image = image.bfloat16()
            elif args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()

            output_ids, pred_masks = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=tokenizer,
            )
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            print("text_output: ", text_output)

            masks_list = masks.int()
            output_list = (pred_masks[0] > 0).int()

            evaluator.add_batch(masks_list.cpu().numpy(), output_list.cpu().numpy())
            assert len(pred_masks) == 1
            for i, pred_mask in enumerate(pred_masks):
                if pred_mask.shape[0] == 0:
                    continue

                pred_mask = pred_mask.detach().cpu().numpy()[0] > 0
                stem = os.path.splitext(os.path.basename(image_path))[0]

                save_path = os.path.join(args.vis_save_path, f"{stem}_mask_{i}.png")
                cv2.imwrite(save_path, pred_mask * 255)
                print("{} has been saved.".format(save_path))

                save_path = os.path.join(args.vis_save_path, f"{stem}_masked_img_{i}.png")
                save_img = image_np.copy()
                save_img[pred_mask] = (
                    image_np * 0.5
                    + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                )[pred_mask]
                save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_path, save_img)
                print("{} has been saved.".format(save_path))

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
