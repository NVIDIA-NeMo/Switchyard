# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapter for the external research feature extractor and saved LightGBM bundle."""

import sys
from pathlib import Path


class ResearchBackend:
    """Reuse the training feature builder and keep verified models resident."""

    def __init__(self, research_root, checkpoint):
        root = Path(research_root).resolve()
        scripts = root / "v1/scripts"
        demo = root / "v1/demo"
        if not (scripts / "predict_lightgbm_context.py").is_file():
            raise ValueError("Expected an agent-progress-research checkout")
        sys.path[:0] = [str(scripts), str(demo)]
        from mini_swe_forecast import feature_from_prefix
        from predict_lightgbm_context import PredictorBundle

        self.checkpoint = Path(checkpoint)
        self.bundle = PredictorBundle(self.checkpoint)
        self.manifest = self.bundle.manifest
        self.features = feature_from_prefix

    def predict(self, features, view="call_level"):
        return self.bundle.predict(features, view)
