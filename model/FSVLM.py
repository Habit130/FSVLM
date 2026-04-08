from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN,IMAGE_TOKEN_INDEX)

from .llava.model.language_model.llava_llama import (LlavaLlamaForCausalLM,
                                                     LlavaLlamaModel)
from .segment_anything import build_sam_vit_h
from .segment_anything.modeling.transformer import Attention as SamAttention
import cv2
def masks_noise(masks):
	def get_incoherent_mask(input_masks, sfact):
		mask = input_masks.float()
		w = input_masks.shape[-1]
		h = input_masks.shape[-2]
		mask_small = F.interpolate(mask, (h//sfact, w//sfact), mode='bilinear')
		mask_recover = F.interpolate(mask_small, (h, w), mode='bilinear')
		mask_residue = (mask - mask_recover).abs()
		mask_residue = (mask_residue >= 0.01).float()
		return mask_residue
	gt_masks_vector = masks / 255
	mask_noise = torch.randn(gt_masks_vector.shape, device= gt_masks_vector.device) * 1.0
	inc_masks = get_incoherent_mask(gt_masks_vector,  8)
	gt_masks_vector = ((gt_masks_vector + mask_noise * inc_masks) > 0.5).float()
	gt_masks_vector = gt_masks_vector * 255

	return gt_masks_vector

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


class SegQueryBridge(nn.Module):
    def __init__(
        self,
        llm_dim: int,
        sam_dim: int,
        out_dim: int,
        bridge_dim: int,
        num_queries: int,
        num_heads: int,
    ):
        super().__init__()
        if bridge_dim % num_heads != 0:
            raise ValueError(
                "bridge_dim must be divisible by num_heads, got {} and {}.".format(
                    bridge_dim, num_heads
                )
            )

        self.out_dim = out_dim
        self.bridge_dim = bridge_dim
        self.num_queries = num_queries
        self.query_proj = nn.Linear(llm_dim, num_queries * bridge_dim)
        self.query_norm = nn.LayerNorm(bridge_dim)
        self.sam_proj = nn.Linear(sam_dim, bridge_dim)
        self.sam_norm = nn.LayerNorm(bridge_dim)
        self.cross_attn = SamAttention(bridge_dim, num_heads)
        self.out_norm = nn.LayerNorm(bridge_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(bridge_dim, bridge_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bridge_dim, out_dim),
            nn.Dropout(0.0),
        )

    def forward(self, h_seg: torch.Tensor, sam_feat: torch.Tensor) -> torch.Tensor:
        if h_seg.shape[0] == 0:
            return h_seg.new_zeros((0, self.out_dim))

        # h_seg comes from the existing <SEG> token selection logic in the LLM.
        queries = self.query_proj(h_seg).view(
            h_seg.shape[0], self.num_queries, self.bridge_dim
        )
        queries = self.query_norm(queries)

        # Use SAM image encoder features instead of LLaVA/CLIP visual features so
        # the bridge queries the same visual space that the SAM prompt branch uses.
        if sam_feat.dim() == 4:
            # [1, C, H, W] -> [1, H * W, C], keeping the SAM spatial order.
            sam_feat = sam_feat.flatten(2).permute(0, 2, 1)
        elif sam_feat.dim() != 3:
            raise ValueError(
                "sam_feat must be [B, C, H, W] or [B, N, C], got {} dims.".format(
                    sam_feat.dim()
                )
            )

        memory = self.sam_proj(sam_feat)
        memory = self.sam_norm(memory)
        if memory.shape[0] == 1:
            memory = memory.expand(h_seg.shape[0], -1, -1)
        elif memory.shape[0] != h_seg.shape[0]:
            raise ValueError(
                "SAM memory batch {} does not match query batch {}.".format(
                    memory.shape[0], h_seg.shape[0]
                )
            )

        attn_out = self.cross_attn(q=queries, k=memory, v=memory)
        queries = self.out_norm(queries + attn_out)
        pooled_queries = queries.mean(dim=1)
        return self.out_proj(pooled_queries)


class FSVLMMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(FSVLMMetaModel, self).__init__(config)

        self.config = config
        has_fsvlm_config = hasattr(self.config, "train_mask_decoder")
        train_mask_decoder = kwargs.get(
            "train_mask_decoder", getattr(self.config, "train_mask_decoder", None)
        )
        out_dim = kwargs.get("out_dim", getattr(self.config, "out_dim", None))
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", train_mask_decoder
        )
        self.config.out_dim = getattr(self.config, "out_dim", out_dim)
        self.config.bridge_type = getattr(
            self.config, "bridge_type", kwargs.get("bridge_type", "mlp")
        )
        self.config.bridge_dim = getattr(
            self.config, "bridge_dim", kwargs.get("bridge_dim", self.config.out_dim)
        )
        self.config.bridge_num_queries = getattr(
            self.config, "bridge_num_queries", kwargs.get("bridge_num_queries", 4)
        )
        self.config.bridge_num_heads = getattr(
            self.config, "bridge_num_heads", kwargs.get("bridge_num_heads", 8)
        )
        self.vision_pretrained = kwargs.get("vision_pretrained", None)
        if has_fsvlm_config:
            self.initialize_fsvlm_modules(self.config)

    def initialize_fsvlm_modules(self, config):
        # SAM即视觉编码加mask生成，前者冻结后者可训练
        self.visual_model = build_sam_vit_h(self.vision_pretrained)#初始化SAM预训练参数
        for param in self.visual_model.parameters():
            param.requires_grad = False#冻结视觉编码部分参数
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()#mask解码部分训练参数可更新
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer，MLP初始化可训练
        in_dim = config.hidden_size
        out_dim = config.out_dim
        self.text_hidden_fcs = None
        self.query_bridge = None
        if config.bridge_type == "mlp":
            text_fc = [
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, out_dim),
                nn.Dropout(0.0),
            ]
            self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
            self.text_hidden_fcs.train()
            for param in self.text_hidden_fcs.parameters():
                param.requires_grad = True
        elif config.bridge_type == "query":
            self.query_bridge = SegQueryBridge(
                llm_dim=in_dim,
                sam_dim=self.visual_model.prompt_encoder.embed_dim,
                out_dim=out_dim,
                bridge_dim=config.bridge_dim,
                num_queries=config.bridge_num_queries,
                num_heads=config.bridge_num_heads,
            )
            self.query_bridge.train()
            for param in self.query_bridge.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Unsupported bridge_type: {}".format(config.bridge_type))


class FSVLMModel(FSVLMMetaModel, LlavaLlamaModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(FSVLMModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


class FSVLMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            
        else:
            config.mm_vision_tower = config.vision_tower
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        super().__init__(config)

        self.model = FSVLMModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def _gather_generation_hidden_states(self, generation_outputs):
        hidden_states = generation_outputs.hidden_states
        if hidden_states is None:
            raise ValueError("Generation did not return hidden states.")

        if isinstance(hidden_states, tuple) and len(hidden_states) > 0:
            first_item = hidden_states[0]
            if isinstance(first_item, tuple):
                pieces = []
                for step_idx, step_hidden_states in enumerate(hidden_states):
                    if not step_hidden_states:
                        continue
                    step_last_hidden = step_hidden_states[-1]
                    if step_last_hidden is None:
                        continue
                    if step_idx == 0:
                        pieces.append(step_last_hidden)
                    else:
                        pieces.append(step_last_hidden[:, -1:, :])
                if not pieces:
                    raise ValueError("Generation hidden states were empty.")
                return torch.cat(pieces, dim=1)

            if torch.is_tensor(hidden_states[-1]):
                return hidden_states[-1]

        if torch.is_tensor(hidden_states):
            return hidden_states

        raise TypeError("Unsupported hidden_states format returned by generate().")

    def _build_seg_prompt_embeddings(
        self,
        last_hidden_state: torch.Tensor,
        seg_token_mask: torch.Tensor,
        image_embeddings: torch.Tensor,
        offset: Optional[torch.LongTensor] = None,
    ) -> List[torch.Tensor]:
        if last_hidden_state.shape[1] != seg_token_mask.shape[1]:
            matched_len = min(last_hidden_state.shape[1], seg_token_mask.shape[1])
            last_hidden_state = last_hidden_state[:, :matched_len, :]
            seg_token_mask = seg_token_mask[:, :matched_len]

        if offset is None:
            if image_embeddings.shape[0] != last_hidden_state.shape[0]:
                raise ValueError(
                    "offset is required when image batch {} does not match sequence batch {}.".format(
                        image_embeddings.shape[0], last_hidden_state.shape[0]
                    )
                )
            offset = torch.arange(
                0,
                last_hidden_state.shape[0] + 1,
                device=last_hidden_state.device,
                dtype=torch.long,
            )

        pred_embeddings = []
        bridge_type = self.model.config.bridge_type
        offset_list = offset.tolist()
        for image_idx, (start_i, end_i) in enumerate(
            zip(offset_list[:-1], offset_list[1:])
        ):
            seq_hidden = last_hidden_state[start_i:end_i]
            seq_mask = seg_token_mask[start_i:end_i]

            if bridge_type == "mlp":
                if self.model.text_hidden_fcs is None:
                    raise RuntimeError("MLP bridge is not initialized.")
                prompt_hidden = self.model.text_hidden_fcs[0](seq_hidden)
                pred_embeddings.append(prompt_hidden[seq_mask])
                continue

            if bridge_type != "query":
                raise ValueError("Unsupported bridge_type: {}".format(bridge_type))
            if self.model.query_bridge is None:
                raise RuntimeError("Query bridge is not initialized.")

            # Keep the <SEG> selection unchanged, then let all <SEG> tokens of the
            # same image attend to the shared SAM image encoder memory in parallel.
            h_seg = seq_hidden[seq_mask]
            pred_embeddings.append(
                self.model.query_bridge(
                    h_seg=h_seg,
                    sam_feat=image_embeddings[image_idx].unsqueeze(0),
                )
            )

        return pred_embeddings

    def predict(self,points=None,boxes=None,masks=None,text_embeding=None,image_embeddings=None,multimask_output=1):
        (
            sparse_embeddings,
            dense_embeddings,
        ) = self.model.visual_model.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
            text_embeds=text_embeding,
        )
        sparse_embeddings = sparse_embeddings.to(text_embeding.dtype)
        
        low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        # avg_attention_map =low_res_masks.cpu().to(torch.float32).numpy()
        # avg_attention_map=avg_attention_map.reshape(256,256)
        # attention_map_reshaped = cv2.resize(avg_attention_map,(512,512))
        return low_res_masks,iou_predictions
    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,
        images_clip: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False,
        **kwargs,
    ):
        image_embeddings = self.get_visual_embs(images)
        
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1
        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )
      
        seg_token_mask = torch.cat(
            [torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(), seg_token_mask],
            dim=1,
        )
    
        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None
   
        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)
           
            output = super().forward(
                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_attentions=True,
                output_hidden_states=True,
            )
          
            output_hidden_states = output.hidden_states

        pred_embeddings = self._build_seg_prompt_embeddings(
            last_hidden_state=output_hidden_states[-1],
            seg_token_mask=seg_token_mask,
            image_embeddings=image_embeddings,
            offset=offset,
        )

        multimask_output = False
        pred_masks = []

        for i in range(len(pred_embeddings)):
            
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                text_embeds=pred_embeddings[i].unsqueeze(1),
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),        
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(), 
                sparse_prompt_embeddings=sparse_embeddings,                     
                dense_prompt_embeddings=dense_embeddings,                       
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,
            )
            pred_masks.append(pred_mask[:, 0])

        model_output = output
        gt_masks = masks_list

        if inference:
            return {
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        ce_loss = model_output.loss
        ce_loss = ce_loss * self.ce_loss_weight
        mask_bce_loss = 0
        mask_dice_loss = 0
        num_masks = 0
        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            num_masks += gt_mask.shape[0]

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

        loss = ce_loss + mask_loss

        return {
            "loss": loss,
            "ce_loss": ce_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
            "mask_loss": mask_loss,
        }

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
        indices=image_token_indices.item()
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )
            # make_attention_map(outputs,input_ids,tokenizer)
            output_hidden_states = self._gather_generation_hidden_states(outputs)
            output_ids = outputs.sequences
            # attention_out=Attention_aggregation(outputs.attentions[-1])
            # attention_out=attention_out[:,-1,indices:indices+256].view(16,16)
            # output_attention=output_attention[-1][:,:,indices:indices+256,indices:indices+256]
            
            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
            # hack for IMAGE_TOKEN_INDEX (we suppose that there is only one image, and it is in the front)
            seg_token_mask = torch.cat(
                [
                    torch.zeros((seg_token_mask.shape[0], 255)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )
            image_embeddings = self.get_visual_embs(images)
            pred_embeddings = self._build_seg_prompt_embeddings(
                last_hidden_state=output_hidden_states,
                seg_token_mask=seg_token_mask,
                image_embeddings=image_embeddings,
            )

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                point=None
                low_res_masks,iou_predictions=self.predict(points=point,text_embeding=pred_embeddings[i].unsqueeze(1),
                                                                 image_embeddings=image_embeddings[i].unsqueeze(0),multimask_output=multimask_output)
                
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])
        return output_ids, pred_masks
