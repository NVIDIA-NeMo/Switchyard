# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face Transformers prefill forward used by the Rust crate."""

from __future__ import annotations

from typing import Any


def _detect_device(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device(torch: Any, override: str | None) -> str:
    device = override.lower().strip() if override is not None else _detect_device(torch)
    if device == "cpu":
        return device
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return device
        raise ValueError("MPS was requested but is not available")
    if device == "cuda":
        if torch.cuda.is_available():
            return device
        raise ValueError("CUDA was requested but is not available")
    if device.startswith("cuda:") and device.removeprefix("cuda:").isdigit():
        index = int(device.removeprefix("cuda:"))
        if torch.cuda.is_available() and index < torch.cuda.device_count():
            return device
        raise ValueError(f"CUDA device {device} is not available")
    raise ValueError(f"Unsupported device: {device}")


class TransformersForward:
    """Lazily load a causal LM and return pooled prefill hidden states."""

    def __init__(
        self,
        model: str,
        *,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self._model_path = model
        self._cache_dir = cache_dir
        self._device_override = device
        self._model = None
        self._tokenizer = None
        self._torch = None
        self.n_layers = 0
        self.hidden_dim = 0

    def _ensure_loaded(self) -> str:
        if self._model is not None:
            return str(self._model.device)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        device = _resolve_device(torch, self._device_override)
        if device == "cpu":
            dtype = torch.float32
        elif device == "mps":
            dtype = torch.float16
        else:
            dtype = (
                torch.bfloat16
                if torch.cuda.get_device_capability(device)[0] >= 8
                else torch.float16
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            cache_dir=self._cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "cache_dir": self._cache_dir,
        }

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            **load_kwargs,
        )
        if device != "cpu":
            self._model.to(device)
        self._model.eval()
        self.n_layers = self._model.config.num_hidden_layers
        self.hidden_dim = self._model.config.hidden_size
        return str(self._model.device)

    def extract_batch(
        self,
        prompts: list[str],
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        extract_layers: list[int] | str = "upper_half",
        pooling_modes: list[str] | None = None,
        batch_size: int = 4,
        max_length: int = 2048,
    ) -> dict[str, Any]:
        """Extract pooled hidden states using the blueprint's direct indexing."""
        self._ensure_loaded()

        if extract_layers == "all":
            layers = list(range(self.n_layers))
        elif extract_layers == "upper_half":
            layers = list(range(self.n_layers // 2, self.n_layers))
        elif isinstance(extract_layers, list):
            layers = [int(layer) for layer in extract_layers]
        else:
            raise ValueError(f"Unsupported layer selection: {extract_layers}")
        if not layers:
            raise ValueError("extract_layers resolved to an empty list")
        invalid = [layer for layer in layers if layer < 0 or layer >= self.n_layers]
        if invalid:
            raise ValueError(
                f"Requested indexes {invalid} are outside hidden-state range 0..{self.n_layers - 1}"
            )

        pools = {"last", "mean"} if pooling_modes is None else set(pooling_modes)
        unknown_pools = pools - {"last", "mean"}
        if unknown_pools:
            raise ValueError(f"Unknown pooling modes: {sorted(unknown_pools)}")
        if not pools:
            raise ValueError("At least one pooling mode is required")

        template_kwargs = chat_template_kwargs or {}
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        formatted = self._tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        all_last = {layer: [] for layer in layers} if "last" in pools else {}
        all_mean = {layer: [] for layer in layers} if "mean" in pools else {}

        for batch_start in range(0, len(formatted), batch_size):
            inputs = self._tokenizer(
                formatted[batch_start : batch_start + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = inputs["input_ids"].to(self._model.device)
            attention_mask = inputs["attention_mask"].to(self._model.device)

            with self._torch.inference_mode():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states
            token_mask = attention_mask.bool()
            token_count = token_mask.sum(dim=1, keepdim=True)
            positions = self._torch.arange(token_mask.shape[1], device=token_mask.device).expand_as(
                token_mask
            )
            last_index = positions.masked_fill(~token_mask, -1).max(dim=1).values
            batch_index = self._torch.arange(token_mask.shape[0], device=token_mask.device)

            for layer in layers:
                hidden = hidden_states[layer].float()
                if "last" in pools:
                    all_last[layer].append(hidden[batch_index, last_index].cpu())
                if "mean" in pools:
                    masked = hidden.masked_fill(~token_mask.unsqueeze(-1), 0)
                    all_mean[layer].append((masked.sum(dim=1) / token_count).cpu())

            del outputs, hidden_states, input_ids, attention_mask

        return {
            "hidden_last": {
                layer: self._torch.cat(rows).contiguous().numpy().tobytes()
                for layer, rows in all_last.items()
            },
            "hidden_mean": {
                layer: self._torch.cat(rows).contiguous().numpy().tobytes()
                for layer, rows in all_mean.items()
            },
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
        }

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self.n_layers = 0
        self.hidden_dim = 0
        if self._torch is not None:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            if hasattr(self._torch, "mps") and self._torch.backends.mps.is_available():
                self._torch.mps.empty_cache()
