import argparse
import os
import shutil
import sys
import time
from functools import partial
import csv
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.FSVLM import FSVLMForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn, save_mask
from utils.model_loading import load_fsvlm_model
from utils.server_metrics import Evaluator
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
def parse_args(args):
    parser = argparse.ArgumentParser(description="FSVLM Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version",
        default="liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview",
    )
    parser.add_argument(
        "--model_base",
        default="meta-llama/Llama-2-7b-hf",
        type=str,
        help="Base Llama checkpoint when --version points to a LoRA preview repo.",
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int) 
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )

    parser.add_argument("--encode_type", default="clip-vit", type=str)

    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--dataset", default="plantseg", type=str
    )
    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument("--farmland_seg_data", default="plantseg", type=str)
    parser.add_argument("--train_split", default="train", type=str)
    parser.add_argument("--val_split", default="val", type=str)
    parser.add_argument("--test_split", default="test", type=str)
    parser.add_argument("--caption_index", default=3, type=int)
    parser.add_argument("--target_name", default="diseased region", type=str)
    parser.add_argument("--dataset_dir", default="../plantseg", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="fsvlm", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=1000, type=int)
    parser.add_argument(
        "--batch_size", default=1, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=8,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)#0.05
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument(
        "--vision_pretrained",
        default="PATH_TO_SAM_ViT-H/sam_vit_h_4b8939.pth",
        type=str,
    )
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_llama_2",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def is_main_process(args):
    return args.local_rank == 0


def maybe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def append_metrics_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["epoch", "IoU", "Dice", "Recall", "mIoU", "mACC"])
        writer.writerow(row)


def is_better_checkpoint(metrics, best_metrics):
    if best_metrics is None:
        return True
    if metrics["mIoU"] > best_metrics["mIoU"]:
        return True
    if metrics["mIoU"] == best_metrics["mIoU"] and metrics["IoU"] > best_metrics["IoU"]:
        return True
    return False


def main(args):
    args = parse_args(args)
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        args.local_rank = int(env_local_rank)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    args.val_metrics_path = os.path.join(args.log_dir, "val_metrics.csv")
    if is_main_process(args):
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    tokenizer_kwargs = {
        "cache_dir": None,
        "model_max_length": args.model_max_length,
        "padding_side": "right",
        "use_fast": False,
    }
    tokenizer_source = args.model_base if args.model_base else args.version
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source, **tokenizer_kwargs
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }

    tokenizer, model = load_fsvlm_model(
        args.version,
        model_kwargs=model_args,
        tokenizer_kwargs=tokenizer_kwargs,
        model_base=args.model_base,
        torch_dtype=torch_dtype,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.seg_token_idx = args.seg_token_idx

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_fsvlm_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # model.get_model().initialize_vision_modules(model.get_model().config,encode_type=args.encode_type,RSCLIP_weight_path=args.RSCLIP_weight_path,GeoRS_path=args.GeoRS_path)#llava初始化
    # for p in vision_tower.parameters():
    #     p.requires_grad = False
    # for p in model.get_model().mm_projector.parameters():
    #     p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]
# "vision_tower"
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
                    # print(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
  
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
 
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    model.resize_token_embeddings(len(tokenizer))

    
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True
   

    world_size = max(1, torch.cuda.device_count())
    args.distributed = world_size > 1
 
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        seg_data=args.farmland_seg_data,
        explanatory=args.explanatory,
        split=args.train_split,
        caption_index=args.caption_index,
        target_name=args.target_name,
    )

    if args.no_eval is False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            split=args.val_split,
            image_size=args.image_size,
            caption_index=args.caption_index,
            target_name=args.target_name,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")
 
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
        "log_dir": "./deepspeed_logs/"
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
        dist_init_required=args.distributed,
    )
    print( ' train_loader:',train_loader)
  
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "iter_train")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_dataset, shuffle=False, drop_last=False
            )
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False if val_sampler is not None else False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_metrics = None

    if args.eval_only:
        metrics = validate(val_loader, model_engine, 0, writer, args)
        if is_main_process(args):
            print(metrics)
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        is_best = args.no_eval
        if args.no_eval is False:
            metrics = validate(val_loader, model_engine, epoch, writer, args)
            is_best = is_better_checkpoint(metrics, best_metrics)
            if is_best:
                best_metrics = metrics

            if is_main_process(args):
                append_metrics_row(
                    args.val_metrics_path,
                    [
                        epoch,
                        metrics["IoU"],
                        metrics["Dice"],
                        metrics["Recall"],
                        metrics["mIoU"],
                        metrics["mACC"],
                    ],
                )

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "iter_train")
            if is_main_process(args):
                best_iou = 0.0 if best_metrics is None else best_metrics["IoU"]
                best_miou = 0.0 if best_metrics is None else best_metrics["mIoU"]
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_iou{:.3f}_miou{:.3f}.pth".format(
                            best_iou, best_miou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)

            maybe_barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
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
            print(input_dict["image_paths"])
            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    evaluator = Evaluator(2)
    evaluator.reset()
    model_engine.eval()
    with torch.no_grad():
        for input_dict in tqdm.tqdm(val_loader):
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
            output_dict = model_engine(**input_dict)

            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            evaluator.add_batch(masks_list.cpu().numpy(), output_list.cpu().numpy())
            assert len(pred_masks) == 1
            if is_main_process(args):
                save_mask(pred_masks, args.vis_save_path, input_dict["image_paths"][0])

    metrics = evaluator.compute_metrics()
    if is_main_process(args) and writer is not None:
        writer.add_scalar("val/iou_fg", metrics["IoU"], epoch)
        writer.add_scalar("val/dice_fg", metrics["Dice"], epoch)
        writer.add_scalar("val/recall_fg", metrics["Recall"], epoch)
        writer.add_scalar("val/miou", metrics["mIoU"], epoch)
        writer.add_scalar("val/macc", metrics["mACC"], epoch)
        print(
            "IoU:{IoU:.4f}, Dice:{Dice:.4f}, Recall:{Recall:.4f}, mIoU:{mIoU:.4f}, mACC:{mACC:.4f}".format(
                **metrics
            )
        )

    return metrics


if __name__ == "__main__":
    main(sys.argv[1:])
