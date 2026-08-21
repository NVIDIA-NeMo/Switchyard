# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face Transformers prefill forward used by the Rust crate."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _detect_device(torch: Any) -> str:
    override = os.environ.get("ROUTER_DEVICE", "").lower()
    if override in ("cpu", "cuda", "mps"):
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        device = self._device_override or _detect_device(torch)
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        cache_dir = self._cache_dir or os.environ.get("HF_HUB_CACHE")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            cache_dir=cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "cache_dir": cache_dir,
        }
        if device != "cpu":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            **load_kwargs,
        )
        self._model.eval()
        self.n_layers = self._model.config.num_hidden_layers
        self.hidden_dim = self._model.config.hidden_size

    def extract_batch(
        self,
        prompts: list[str],
        *,
        chat_template_kwargs: dict[str, Any] | None = None,
        extract_layers: list[int] | str | None = None,
        pooling_modes: list[str] | None = None,
        batch_size: int = 4,
        max_length: int = 2048,
    ) -> dict[str, Any]:
        """Extract pooled hidden states using the blueprint's direct indexing."""
        self._ensure_loaded()

        if extract_layers == "all":
            layers = list(range(self.n_layers))
        elif extract_layers is None:
            layers = list(range(self.n_layers // 2, self.n_layers))
        else:
            layers = [int(layer) for layer in extract_layers]
        if not layers:
            raise ValueError("extract_layers resolved to an empty list")
        invalid = [layer for layer in layers if layer < 0 or layer >= self.n_layers]
        if invalid:
            raise ValueError(
                f"Requested layers {invalid} are outside encoder range "
                f"0..{self.n_layers - 1}"
            )

        pools = set(pooling_modes or ["last", "mean"])
        unknown_pools = pools - {"last", "mean"}
        if unknown_pools:
            raise ValueError(f"Unknown pooling modes: {sorted(unknown_pools)}")
        if not pools:
            raise ValueError("At least one pooling mode is required")

        template_kwargs = chat_template_kwargs or {}
        formatted = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            for prompt in prompts
        ]
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

            with self._torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states
            for batch_index in range(input_ids.shape[0]):
                token_mask = attention_mask[batch_index].bool()
                for layer in layers:
                    hidden = hidden_states[layer][batch_index, token_mask, :].float()
                    if "last" in pools:
                        all_last[layer].append(hidden[-1].cpu())
                    if "mean" in pools:
                        all_mean[layer].append(hidden.mean(dim=0).cpu())

            del outputs, hidden_states, input_ids, attention_mask
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()

        return {
            "hidden_last": {
                layer: self._torch.stack(rows).tolist() for layer, rows in all_last.items()
            },
            "hidden_mean": {
                layer: self._torch.stack(rows).tolist() for layer, rows in all_mean.items()
            },
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
        }

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
