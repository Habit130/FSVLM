import os

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.llava.constants import IGNORE_INDEX
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .orgin_dataset import (
    build_segmentation_question,
    load_plantseg_records,
    orginDataset,
)
from .utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN


def save_mask(pred_masks, save_path_dir, image_path):
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    image_np = cv2.imread(image_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    for i, pred_mask in enumerate(pred_masks):
        if pred_mask.shape[0] == 0:
            continue
        pred_mask = pred_mask.detach().cpu().numpy()[0]
        pred_mask = pred_mask > 0

        save_path = "{}/{}_mask_{}.png".format(
            save_path_dir, os.path.splitext(os.path.basename(image_path))[0], i
        )
        cv2.imwrite(save_path, pred_mask * 100)
        print("{} has been saved.".format(save_path))

        save_path = "{}/{}_masked_img_{}.png".format(
            save_path_dir, os.path.splitext(os.path.basename(image_path))[0], i
        )
        save_img = image_np.copy()
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
        )[pred_mask]
        save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
        print("{} has been saved.".format(save_path))


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    # local_text=[]
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)
        # local_text.append("farmland")

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )
            #<im_start>+<image>+<im_end>
   
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
   
    #填充序列
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
   
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
       
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_INDEX, tokenizer.unk_token_id, z)
            if local_rank == 0:
                print(
                    "conversation: ",
                    conversation,
                    "tokenizer.decode(z): ",
                    tokenizer.decode(z),
                )

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    if inferences[0] == False:
        truncate_len = tokenizer.model_max_length - 255
        # print('input_ids.shape[1]',input_ids.shape[1])
        # print('input_ids.shape[0]',input_ids.shape[0])
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            targets = targets[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
     
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


class HybridDataset(torch.utils.data.Dataset):
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
        dataset="plantseg",
        sample_rate=[1],
        seg_data="plantseg",
        explanatory=0.1,
        split="train",
        caption_index=3,
        target_name="diseased region",
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.datasets = dataset.split("||")
        self.all_datasets = []
        for dataset_name in self.datasets:
            if dataset_name in {"plantseg", "farmsegvl"}:
                self.all_datasets.append(
                    orginDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        seg_data,
                        split=split,
                        caption_index=caption_index,
                        target_name=target_name,
                    )
                )
            else:
                raise ValueError(f"Unsupported dataset name: {dataset_name}")
           
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        split="val",
        image_size=1024,
        caption_index=3,
        target_name="diseased region",
    ):
        if "ViT-L-14" in vision_tower:
            vision_tower = "openai/clip-vit-large-patch14"
        self.base_image_dir = base_image_dir
        self.records = load_plantseg_records(
            self.base_image_dir,
            split=split,
            caption_index=caption_index,
        )
        print(f"{split}: {len(self.records)}")
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.target_name = target_name

    def __len__(self):
        return len(self.records)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = record["image_path"]
        text = record["caption"]
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN
            + "\n {} Can you segment the {} in this image?".format(
                text, self.target_name
            ),
        )
        conv.append_message(conv.roles[1], "[SEG].")
        conversations = [conv.get_prompt()]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        mask = Image.open(record["mask_path"])
        masks = np.array(mask)
        masks[masks != 0] = 1
        masks = masks[np.newaxis, :, :]
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
