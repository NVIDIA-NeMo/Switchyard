// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"

	switchyard "github.com/NVIDIA-NeMo/Switchyard/bindings/go"
)

func main() { fmt.Println(switchyard.Backend()) }
