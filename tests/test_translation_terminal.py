# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from switchyard.translation import TranslationEngine


async def test_decoder_stops_at_message_stop():
    async def upstream():
        yield b'data: {"type":"message_stop"}\n\n'
        raise AssertionError("decoder read after message_stop")

    async with TranslationEngine().decode_stream("anthropic_messages", upstream()) as decoded:
        assert [event["preservation"]["raw"] async for event in decoded] == [
            {"type": "message_stop"}
        ]
