# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Search Harbor adapter.

Converts search TaskCandidate JSON files into Harbor task directories. Builds
fresh Dockerfiles that clone the repo and install agent runtimes.
"""
