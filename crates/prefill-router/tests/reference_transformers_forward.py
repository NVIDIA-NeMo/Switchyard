# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen tensor oracle from llm-router v3 commit 8a9d3509fbde879d9795258081bab9553b458e04."""

from __future__ import annotations

import os
from typing import Any

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class PrefillExtractor:
    """Upstream Transformers extractor used as the parity oracle."""

    def __init__(
        self,
        hf_path: str,
        *,
        device: str | None = None,
        dtype: Any = None,
        cache_dir: str | None = None,
    ):
        self._hf_path = hf_path
        self._cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self.n_layers = 0
        self.hidden_dim = 0

        self._device = device or "cpu"
        if dtype is not None:
            self._dtype = dtype
        elif self._device == "cpu":
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = self._cache_dir or os.environ.get("HF_HUB_CACHE")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hf_path,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "dtype": self._dtype,
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }
        if self._device != "cpu":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self._hf_path, **load_kwargs)
        self._model.eval()
        self.n_layers = self._model.config.num_hidden_layers
        self.hidden_dim = self._model.config.hidden_size

    def extract_batch(
        self,
        questions: list[str],
        *,
        chat_template_kwargs: dict | None = None,
        extract_layers: list[int] | str | None = None,
        pooling_modes: list[str] | None = None,
        batch_size: int = 4,
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Run the upstream extraction path and return serializable tensors."""
        self._ensure_loaded()

        template_kwargs = chat_template_kwargs or {}
        if extract_layers == "all":
            layers = list(range(self.n_layers))
        elif extract_layers is None:
            half = self.n_layers // 2
            layers = list(range(half, self.n_layers))
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

        formatted = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            for question in questions
        ]
        all_last = {layer: [] for layer in layers} if "last" in pools else {}
        all_mean = {layer: [] for layer in layers} if "mean" in pools else {}

        for batch_start in range(0, len(formatted), batch_size):
            batch_texts = formatted[batch_start : batch_start + batch_size]
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = inputs["input_ids"].to(self._model.device)
            attention_mask = inputs["attention_mask"].to(self._model.device)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states
            sequence_lengths = attention_mask.sum(dim=1)
            for batch_index in range(input_ids.shape[0]):
                sequence_length = int(sequence_lengths[batch_index].item())
                for layer in layers:
                    hidden = hidden_states[layer][batch_index, :sequence_length, :].float()
                    if "last" in pools:
                        all_last[layer].append(hidden[-1].cpu())
                    if "mean" in pools:
                        all_mean[layer].append(hidden.mean(dim=0).cpu())

            del outputs, hidden_states, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "hidden_last": {
                layer: torch.stack(rows).tolist() for layer, rows in all_last.items()
            },
            "hidden_mean": {
                layer: torch.stack(rows).tolist() for layer, rows in all_mean.items()
            },
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
        }


def extract_reference(
    model: str,
    prompts: list[str],
    pooling_modes: list[str],
    enable_thinking: bool,
) -> dict[str, Any]:
    """Extract the complete tensor oracle for the Rust parity assertion."""
    extractor = PrefillExtractor(model, device="cpu")
    return extractor.extract_batch(
        prompts,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        extract_layers="all",
        pooling_modes=pooling_modes,
        batch_size=2,
        max_length=32,
        show_progress=False,
    )
