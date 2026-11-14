# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from lightgbm_worker import build_rows, load_model, forecast

class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.entry = load_model()

    def request(self):
        messages = [
            {'role': 'user', 'content': 'Fix parser and test it'},
            {'role': 'assistant', 'model': 'openai/gpt-6.1-sol',
             'usage': {'input': 100, 'cacheRead': 20, 'cacheWrite': 5, 'output': 10, 'reasoning': 3},
             'content': [{'type': 'toolCall', 'name': 'bash', 'id': 'a', 'arguments': {'command': 'pytest tests/test_parser.py'}}]},
            {'role': 'toolResult', 'toolCallId': 'a', 'toolName': 'bash', 'isError': True,
             'details': {'exitCode': 1}, 'content': [{'type': 'text', 'text': 'FAILED tests/test_parser.py - AssertionError: wrong value'}]},
        ]
        return {'session_id': 'fixture', 'cwd': '/workspace', 'model': 'openrouter/openai/gpt-6.1-sol',
                'messages': messages, 'request_messages': messages}

    def test_usage_and_horizons_are_source_only(self):
        request = self.request()
        rows, completed, adapted = build_rows(request, self.model.config['ir_features'])
        self.assertEqual(completed, 1)
        self.assertEqual([r['position'] for r in rows], ['call_start', 'task_update'])
        self.assertEqual(rows[0]['features']['last_call_input_tokens'], 125)
        self.assertEqual(rows[0]['features']['last_call_output_tokens'], 10)
        self.assertIsNone(rows[0]['features']['current_request_tokens_local'])
        self.assertIsNone(rows[0]['features']['configured_call_limit'])
        self.assertIn('AssertionError', rows[0]['ir_features']['execution_text'])
        self.assertNotIn('labels', rows[0])
        result = forecast(request, self.model, self.entry)
        self.assertGreaterEqual(result['next_call']['tokens'], 0)
        self.assertIsNone(result['next_call']['requirements'])
        self.assertIsNone(result['remaining_task']['requirements'])

    def test_native_read_and_write_have_explicit_ir_adapters(self):
        request = self.request()
        request['messages'][1]['content'][0].update(name='write', arguments={'path': 'result.py', 'content': 'print(1)\n'})
        request['messages'][2].update(isError=False, details={}, content=[{'type': 'text', 'text': 'Successfully wrote result.py'}])
        rows, _, adapted = build_rows(request, self.model.config['ir_features'])
        self.assertEqual(adapted, ['write'])
        self.assertIn('result.py', rows[0]['ir_features']['artifact_text'])
        request['messages'][1]['content'][0].update(name='read', arguments={'path': 'result.py'})
        rows, _, adapted = build_rows(request, self.model.config['ir_features'])
        self.assertEqual(adapted, ['read'])

    def test_unknown_tool_does_not_produce_fake_ir(self):
        request = self.request()
        request['messages'][1]['content'][0]['name'] = 'unknown_custom_tool'
        with self.assertRaisesRegex(ValueError, 'No IR adapter'):
            build_rows(request, self.model.config['ir_features'])

if __name__ == '__main__':
    unittest.main()
