import os

import torch
from transformers import AutoConfig, AutoTokenizer

from model.FSVLM import FSVLMForCausalLM


def _load_non_lora_trainables(model_path):
    non_lora_path = os.path.join(model_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        return torch.load(non_lora_path, map_location="cpu")

    from huggingface_hub import hf_hub_download

    cache_file = hf_hub_download(
        repo_id=model_path,
        filename="non_lora_trainables.bin",
    )
    return torch.load(cache_file, map_location="cpu")


def _normalize_state_dict_keys(state_dict):
    normalized = {
        (key[11:] if key.startswith("base_model.") else key): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("model.model.") for key in normalized):
        normalized = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in normalized.items()
        }
    return normalized


def load_fsvlm_model(
    version,
    model_kwargs,
    tokenizer_kwargs=None,
    model_base=None,
    torch_dtype=torch.float32,
    pretrained_kwargs=None,
):
    tokenizer_kwargs = tokenizer_kwargs or {}
    pretrained_kwargs = pretrained_kwargs or {}
    tokenizer_source = model_base if model_base else version
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)

    if model_base:
        config = AutoConfig.from_pretrained(version)
        model = FSVLMForCausalLM.from_pretrained(
            model_base,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **model_kwargs,
            **pretrained_kwargs,
        )

        non_lora_trainables = _normalize_state_dict_keys(
            _load_non_lora_trainables(version)
        )
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, version)
        model = model.merge_and_unload()
    else:
        model = FSVLMForCausalLM.from_pretrained(
            version,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            **model_kwargs,
            **pretrained_kwargs,
        )

    return tokenizer, model
