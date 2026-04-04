import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


def _resolve_data_root(base_image_dir):
    return os.path.abspath(base_image_dir)


def load_plantseg_records(base_image_dir, split, caption_index=3):
    data_root = _resolve_data_root(base_image_dir)
    metadata_path = os.path.join(data_root, "main.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"PlantSeg metadata not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as handle:
        samples = json.load(handle)

    records = []
    for sample in samples:
        if sample.get("split") != split:
            continue

        captions = sample.get("caption") or []
        if len(captions) <= caption_index:
            sample_id = sample.get("id", "<unknown>")
            raise ValueError(
                f"Sample {sample_id} is missing caption[{caption_index}] in {metadata_path}"
            )

        image_path = os.path.join(data_root, sample["image"])
        mask_path = os.path.join(data_root, sample["mask"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"PlantSeg image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"PlantSeg mask not found: {mask_path}")

        records.append(
            {
                "id": sample.get("id"),
                "image_path": image_path,
                "mask_path": mask_path,
                "caption": captions[caption_index].strip(),
                "disease_label": sample.get("disease_label", "plant disease"),
                "split": split,
            }
        )

    if not records:
        raise ValueError(f"No PlantSeg samples found for split={split} in {metadata_path}")

    return records


def build_segmentation_question(text, target_name):
    question_template = random.choice(SHORT_QUESTION_LIST)
    return question_template.format(text_name=text, class_name=target_name)


class orginDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        sem_seg_data="plantseg",
        split="train",
        caption_index=3,
        target_name="diseased region",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = _resolve_data_root(base_image_dir)
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.answer_list = ANSWER_LIST
        self.target_name = target_name

        if sem_seg_data not in {"plantseg", "farmsegvl"}:
            raise ValueError(f"Unsupported dataset name: {sem_seg_data}")

        self.records = load_plantseg_records(
            self.base_image_dir,
            split=split,
            caption_index=caption_index,
        )
        self.length = len(self.records)
        print(f"{split} samples: {self.length}")

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        record = self.records[random.randint(0, self.length - 1)]
        image_path = record["image_path"]
        label_path = record["mask_path"]
        text = record["caption"]
        sampled_classes = text

        question = build_segmentation_question(text, self.target_name)
        answer = random.choice(self.answer_list)

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversations = [conv.get_prompt()]
        questions = [question]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        mask = Image.open(label_path)
        masks = np.array(mask)
        masks[masks != 0] = 1
        masks = masks[np.newaxis, :, :]
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )
