# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configured routing decisions with trusted target requirements.

Load deployment TOML once, then supply requirements from authenticated host policy::

    runner = Runner.from_toml(source)
    outcome = await runner.decide(
        normalized_request,
        required_labels=["private", "customer_data"],
        task_id=trusted_task_id,
    )

The host serves final completions within ``outcome.selected_model_ids``. Prefer
``outcome.candidate_requests[model]`` when present; it preserves that candidate's
request changes. Omitted labels are unspecified; an empty list supplies an explicit
empty set. Neither clears retained task requirements. This API does not classify data.
"""

from switchyard_rust.runner import Runner

__all__ = ["Runner"]
