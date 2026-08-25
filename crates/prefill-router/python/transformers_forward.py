# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete Transformers prefill and checkpoint inference for the Rust crate."""

from __future__ import annotations

from pathlib import Path
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
    """Run encoder extraction and learned confidence inference in one pass."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        import numpy as np
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["format_version"] != 1:
            raise ValueError("unsupported checkpoint format_version")

        encoder = checkpoint["encoder"]
        architecture = checkpoint["architecture"]
        pipeline = checkpoint["feature_pipeline"]
        if encoder["view"] != "task_prompt_only":
            raise ValueError("checkpoint encoder view must be task_prompt_only")
        if pipeline["pooling"] != "mean of independently standardized selected layers":
            raise ValueError("unsupported checkpoint feature pooling")
        if architecture["activation"] != "ReLU":
            raise ValueError("checkpoint activation must be ReLU")
        if architecture["ensemble_reduction"] != "mean(sigmoid(logits))":
            raise ValueError("unsupported checkpoint ensemble reduction")

        self._numpy = np
        self._torch = torch
        self._model_path = str(encoder["name"])
        self._expected_layers = int(encoder["n_layers"])
        self._hidden_dim = int(encoder["hidden_dim"])
        self._models = [str(model) for model in checkpoint["models"]]
        self._selected_layers = [int(layer) for layer in pipeline["selected_layers"]]
        self._layer_mean = torch.stack(
            [pipeline["layer_mean"][str(layer)].float() for layer in self._selected_layers]
        ).numpy()
        self._layer_std = torch.stack(
            [pipeline["layer_std"][str(layer)].float() for layer in self._selected_layers]
        ).numpy()
        self._scaler_mean = pipeline["scaler_mean"].numpy()
        self._scaler_scale = pipeline["scaler_scale"].numpy()
        self._pca_mean = pipeline["pca_mean"].numpy()
        self._pca_components = pipeline["pca_components"].numpy()
        self._states = checkpoint["model_state_dicts"]
        self._cache_dir = cache_dir
        self._device_override = device
        self._model = None
        self._tokenizer = None

        if not self._selected_layers or len(set(self._selected_layers)) != len(
            self._selected_layers
        ):
            raise ValueError("checkpoint selected layers must be non-empty and unique")
        if not self._models or not self._states:
            raise ValueError("checkpoint must contain models and ensemble members")
        if self._layer_mean.shape != self._layer_std.shape or self._layer_mean.shape != (
            len(self._selected_layers),
            self._hidden_dim,
        ):
            raise ValueError("checkpoint layer normalization shape is inconsistent")
        if not bool(np.all(self._layer_std > 0)) or not bool(np.all(self._scaler_scale > 0)):
            raise ValueError("checkpoint normalization scales must be positive")

    def metadata(self) -> tuple[str, int]:
        """Return the encoder and ordered output count consumed by Rust."""
        return self._model_path, len(self._models)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = self._torch
        device = _resolve_device(torch, self._device_override)
        dtype = (
            torch.float32
            if device == "cpu"
            else torch.float16
            if device == "mps"
            else torch.bfloat16
            if torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )
        model_kwargs: dict[str, Any] = {}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        elif device.startswith("cuda:"):
            model_kwargs["device_map"] = {"": device}
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            cache_dir=self._cache_dir,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        causal_model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            dtype=dtype,
            cache_dir=self._cache_dir,
            **model_kwargs,
        )
        self._model = causal_model.base_model
        del causal_model
        if device == "mps":
            self._model.to(device)
        self._model.eval()
        model_config = getattr(self._model.config, "text_config", self._model.config)
        if (
            model_config.num_hidden_layers != self._expected_layers
            or model_config.hidden_size != self._hidden_dim
        ):
            raise ValueError("loaded encoder dimensions do not match checkpoint metadata")

    def forward(self, prompts: list[str], batch_size: int, max_length: int) -> bytes:
        """Return a row-major F32 probability matrix for ordered prompts."""
        self._ensure_loaded()
        if not prompts or any(not prompt for prompt in prompts):
            raise ValueError("prompts must be non-empty")
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("batch_size and max_length must be positive")

        formatted = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        predictions = []
        for start in range(0, len(formatted), batch_size):
            inputs = self._tokenizer(
                formatted[start : start + batch_size],
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

            layers = []
            for layer in self._selected_layers:
                hidden = outputs.hidden_states[layer].float()
                # The checkpoint consumes the last real token from each selected layer.
                token_mask = attention_mask.to(hidden.device).bool()
                positions = self._torch.arange(hidden.shape[1], device=hidden.device)
                last_token = (
                    positions.expand_as(token_mask).masked_fill(~token_mask, -1).max(dim=1).values
                )
                pooled = hidden[
                    self._torch.arange(hidden.shape[0], device=hidden.device), last_token
                ]
                layers.append(pooled.cpu().numpy())
            predictions.append(self._predict(layers))

        return self._numpy.ascontiguousarray(
            self._numpy.concatenate(predictions, axis=0), dtype=self._numpy.float32
        ).tobytes()

    def _predict(self, layers: list[Any]) -> Any:
        np = self._numpy
        torch = self._torch
        stacked = np.stack(layers)
        if stacked.ndim != 3 or stacked.shape[0] != len(self._selected_layers):
            raise ValueError("encoder returned invalid selected-layer features")
        standardized = (stacked - self._layer_mean[:, None, :]) / self._layer_std[:, None, :]
        pooled = standardized.mean(axis=0)
        scaled = (pooled - self._scaler_mean) / self._scaler_scale
        features = torch.from_numpy(
            np.ascontiguousarray(
                (scaled - self._pca_mean) @ self._pca_components.T,
                dtype=np.float32,
            )
        )

        members = []
        with torch.inference_mode():
            for state in self._states:
                logits = []
                for index in range(len(self._models)):
                    adapter = torch.nn.functional.relu(
                        torch.nn.functional.linear(
                            features,
                            state[f"adapters.{index}.weight"],
                            state[f"adapters.{index}.bias"],
                        )
                    )
                    trunk = torch.nn.functional.relu(
                        torch.nn.functional.linear(
                            adapter,
                            state["trunk.0.weight"],
                            state["trunk.0.bias"],
                        )
                    )
                    logits.append(
                        torch.nn.functional.linear(
                            trunk,
                            state[f"heads.{index}.weight"],
                            state[f"heads.{index}.bias"],
                        )
                    )
                members.append(torch.sigmoid(torch.cat(logits, dim=1)))
        return torch.stack(members).mean(dim=0).contiguous().numpy()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        if hasattr(self._torch, "mps") and self._torch.backends.mps.is_available():
            self._torch.mps.empty_cache()
