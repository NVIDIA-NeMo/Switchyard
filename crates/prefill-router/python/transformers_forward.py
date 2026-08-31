# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete Transformers prefill and checkpoint inference for the Rust crate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_FORMAT_VERSION = 1
EXPECTED_ENCODER_VIEW = "task_prompt_only"
EXPECTED_POOLING = "mean of independently standardized selected layers"
EXPECTED_ACTIVATION = "ReLU"
EXPECTED_ENSEMBLE_REDUCTION = "mean(sigmoid(logits))"


def _expect_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"unsupported checkpoint {name}: expected {expected!r}, got {actual!r}")


def _validate_checkpoint_contract(checkpoint: dict[str, Any]) -> None:
    _expect_equal("format_version", checkpoint["format_version"], EXPECTED_FORMAT_VERSION)

    encoder = checkpoint["encoder"]
    architecture = checkpoint["architecture"]
    pipeline = checkpoint["feature_pipeline"]
    _expect_equal("encoder view", encoder["view"], EXPECTED_ENCODER_VIEW)
    _expect_equal("feature pooling", pipeline["pooling"], EXPECTED_POOLING)
    _expect_equal("activation", architecture["activation"], EXPECTED_ACTIVATION)
    _expect_equal(
        "ensemble reduction",
        architecture["ensemble_reduction"],
        EXPECTED_ENSEMBLE_REDUCTION,
    )


def _select_dtype(torch: Any, device: str) -> Any:
    if device == "cpu":
        return torch.float32
    if device == "mps":
        return torch.float16
    if torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _state_value(state: dict[str, Any], key: str) -> Any:
    try:
        return state[key]
    except KeyError as exc:
        raise ValueError(f"checkpoint model state is missing {key!r}") from exc


def _linear_from_state(torch: Any, state: dict[str, Any], prefix: str) -> Any:
    weight = _state_value(state, f"{prefix}.weight")
    bias = _state_value(state, f"{prefix}.bias")
    if len(weight.shape) != 2:
        raise ValueError(
            f"checkpoint {prefix}.weight has invalid shape: expected 2 dimensions, "
            f"got {tuple(weight.shape)!r}"
        )
    output_features, input_features = weight.shape
    if tuple(bias.shape) != (output_features,):
        raise ValueError(
            f"checkpoint {prefix}.bias has invalid shape: expected {(output_features,)!r}, "
            f"got {tuple(bias.shape)!r}"
        )
    return torch.nn.Linear(input_features, output_features)


def _build_router_member(torch: Any, state: dict[str, Any], output_count: int) -> Any:
    class RouterMember(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapters = torch.nn.ModuleList(
                [_linear_from_state(torch, state, f"adapters.{index}") for index in range(output_count)]
            )
            self.trunk = torch.nn.Sequential(
                _linear_from_state(torch, state, "trunk.0"),
                torch.nn.ReLU(),
            )
            self.heads = torch.nn.ModuleList(
                [_linear_from_state(torch, state, f"heads.{index}") for index in range(output_count)]
            )

        def forward(self, features: Any) -> Any:
            logits = []
            for adapter, head in zip(self.adapters, self.heads, strict=True):
                adapter_output = torch.nn.functional.relu(adapter(features))
                logits.append(head(self.trunk(adapter_output)))
            return torch.cat(logits, dim=1)

    member = RouterMember()
    try:
        member.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"checkpoint model state does not match router head: {exc}") from exc
    member.eval()
    return member


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
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_checkpoint_contract(checkpoint)

        encoder = checkpoint["encoder"]
        pipeline = checkpoint["feature_pipeline"]
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
        self._ensemble = [
            _build_router_member(torch, state, len(self._models))
            for state in checkpoint["model_state_dicts"]
        ]
        self._cache_dir = cache_dir
        self._device_override = device
        self._model = None
        self._tokenizer = None

        self._validate_checkpoint_arrays()

    def _validate_checkpoint_arrays(self) -> None:
        if not self._selected_layers:
            raise ValueError("checkpoint selected layers are invalid: expected non-empty, got []")
        if len(set(self._selected_layers)) != len(self._selected_layers):
            raise ValueError(
                "checkpoint selected layers are invalid: expected unique values, "
                f"got {self._selected_layers!r}"
            )
        if not self._models:
            raise ValueError("checkpoint models are invalid: expected non-empty, got []")
        if not self._ensemble:
            raise ValueError("checkpoint ensemble is invalid: expected non-empty, got []")
        expected_shape = (len(self._selected_layers), self._hidden_dim)
        if self._layer_mean.shape != self._layer_std.shape:
            raise ValueError(
                "checkpoint layer normalization shape is inconsistent: "
                f"expected layer_mean shape to equal layer_std shape, got "
                f"{self._layer_mean.shape!r} and {self._layer_std.shape!r}"
            )
        if self._layer_mean.shape != expected_shape:
            raise ValueError(
                "checkpoint layer normalization shape is inconsistent: "
                f"expected {expected_shape!r}, got {self._layer_mean.shape!r}"
            )
        if not bool(np.all(self._layer_std > 0)):
            raise ValueError("checkpoint layer_std is invalid: expected all positive values")
        if not bool(np.all(self._scaler_scale > 0)):
            raise ValueError("checkpoint scaler_scale is invalid: expected all positive values")

    def metadata(self) -> tuple[str, int]:
        """Return the encoder and ordered output count consumed by Rust."""
        return self._model_path, len(self._models)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = self._torch
        device = _resolve_device(torch, self._device_override)
        dtype = _select_dtype(torch, device)
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
            expected = (self._expected_layers, self._hidden_dim)
            actual = (model_config.num_hidden_layers, model_config.hidden_size)
            raise ValueError(
                "loaded encoder dimensions do not match checkpoint metadata: "
                f"expected {expected!r}, got {actual!r}"
            )

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
        predictions: list[np.ndarray] = []
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

        return np.ascontiguousarray(np.concatenate(predictions, axis=0), dtype=np.float32).tobytes()

    def _predict(self, layers: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        stacked = np.stack(layers)
        if stacked.ndim != 3 or stacked.shape[0] != len(self._selected_layers):
            expected = (len(self._selected_layers), "batch", self._hidden_dim)
            raise ValueError(
                "encoder returned invalid selected-layer features: "
                f"expected {expected!r}, got {stacked.shape!r}"
            )
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
            for member in self._ensemble:
                members.append(torch.sigmoid(member(features)))
        return torch.stack(members).mean(dim=0).contiguous().numpy()

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        if hasattr(self._torch, "mps") and self._torch.backends.mps.is_available():
            self._torch.mps.empty_cache()
