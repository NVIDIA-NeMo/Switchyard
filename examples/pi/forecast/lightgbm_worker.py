# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local JSON-lines inference worker; recorded Pi tools are parsed, never executed."""
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shlex
import sys
import time

def required_root(variable):
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f'Set {variable} or configure the equivalent forecast checkout path')
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f'{variable} does not name a directory: {root}')
    return root


RESEARCH = required_root('SY_FORECAST_RESEARCH_ROOT')
TOKENCAST = required_root('SY_FORECAST_TOKENCAST_ROOT')
sys.path[:0] = [str(RESEARCH / 'scripts'), str(TOKENCAST / 'src')]
from token_prediction.contracts import CanonicalEvent, EventType
from token_prediction.features.reducer import FeatureState
from ir_grouped import GroupedObserver, grouped_project


def load_model():
    root = RESEARCH / 'checkpoints/grouped-ir'
    manifest = json.loads((root / 'manifest.json').read_text())
    # First published seed, not a winner selected on validation.
    entry = manifest['checkpoints'][0]
    path = root / entry['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    with path.open('rb') as f:
        model = pickle.load(f)
    return model, entry


def text(message):
    content = message.get('content', '')
    if isinstance(content, str):
        return content
    return '\n'.join(b.get('text', '') for b in content if b.get('type') == 'text')


def shell_surface(name, args):
    """Explicit native-tool adaptation for the checkpoint's shell-only IR parser."""
    q = shlex.quote
    path = args.get('path', '.')
    if name == 'bash':
        return args['command']
    if name == 'read':
        return 'cat -- ' + q(path)
    if name == 'write':
        marker = 'PI_CONTENT_' + hashlib.sha256(args.get('content', '').encode()).hexdigest()[:12]
        return f'cat > {q(path)} <<\'{marker}\'\n{args.get("content", "")}\n{marker}'
    if name == 'edit':
        # Preserve requested old/new text without claiming an observed filesystem image.
        old = args.get('oldText', '').splitlines()
        new = args.get('newText', '').splitlines()
        patch = '\n'.join(['*** Begin Patch', f'*** Update File: {path}', '@@',
                           *('-' + l for l in old), *('+' + l for l in new), '*** End Patch'])
        return "apply_patch <<'PI_PATCH'\n" + patch + '\nPI_PATCH'
    if name == 'grep':
        return 'rg -- ' + q(args.get('pattern', '')) + ' ' + q(path)
    if name == 'find':
        return 'find ' + q(path) + ' -name ' + q(args.get('pattern', '*'))
    if name == 'ls':
        return 'ls -- ' + q(path)
    raise ValueError(f'No IR adapter for tool {name!r}')


def build_rows(request, config):
    messages = request['messages']
    task = next((text(m) for m in messages if m['role'] == 'user'), '')
    tid = request['session_id']
    state = FeatureState(trajectory_id=tid)
    observer = GroupedObserver({**config, 'working_directory': request['cwd']})
    seq = -1
    latest = None
    completed = 0
    pending = {}
    adapted = set()

    def emit(kind, payload, cid=None, *, ir_payload=None):
        nonlocal seq, state, latest
        seq += 1
        event = CanonicalEvent(1, f'{tid}:{seq}', tid, seq, EventType(kind),
                               '1970-01-01T00:00:00+00:00', json.dumps(payload),
                               logical_call_id=cid,
                               attempt_id=f'{cid}:observed' if kind in ('api_attempt_started', 'api_completed', 'api_failed') else None)
        state = state.apply(event)
        observer.apply({'event_type': kind, 'event_seq': seq, 'trajectory_id': tid,
                        'logical_call_id': cid, 'payload': ir_payload or payload})
        latest = event

    emit('task_started', {'task_id': tid, 'task_text': task,
                         'model_id': request['model'], 'agent_id': 'pi'})
    for i, message in enumerate(messages):
        if message['role'] == 'assistant':
            cid = f'call:{completed}'
            prefix = messages[:i]
            emit('request_built', {'model': message.get('model', request['model']),
                                  'message_count': len(prefix)}, cid)
            emit('api_attempt_started', {}, cid)
            usage = message.get('usage') or {}
            # Pi splits cache tokens from input; training uses total provider input.
            inputs = usage.get('input')
            if inputs is not None:
                inputs += usage.get('cacheRead', 0) + usage.get('cacheWrite', 0)
            emit('api_completed', {'usage': {
                'input_tokens': inputs, 'output_tokens': usage.get('output'),
                'cached_input_tokens': usage.get('cacheRead'),
                'cache_write_input_tokens': usage.get('cacheWrite'),
                'reasoning_output_tokens': usage.get('reasoning') }}, cid)
            content = message.get('content', [])
            for block in content if isinstance(content, list) else []:
                if block.get('type') != 'toolCall':
                    continue
                name, args, call_id = block['name'], block['arguments'], block['id']
                command = shell_surface(name, args)
                payload = {'tool_call_id': call_id, 'tool_name': name, 'arguments': args}
                ir_payload = {**payload, 'tool_name': 'bash', 'arguments': {'command': command}}
                emit('tool_started', payload, cid, ir_payload=ir_payload)
                pending[call_id] = cid
                if name != 'bash':
                    adapted.add(name)
            completed += 1
        elif message['role'] == 'toolResult':
            call_id = message['toolCallId']
            if call_id not in pending:
                raise ValueError(f'Missing source action for tool result {call_id}')
            result = text(message)
            # Native Pi success is not an observed shell exit code.
            code = (message.get('details') or {}).get('exitCode')
            payload = {'tool_call_id': call_id, 'exit_code': code,
                       'result_text': result, 'result_chars': len(result),
                       'result_utf8_bytes': len(result.encode()), 'result_lines': len(result.splitlines()),
                       'result_fields': {'is_error': message.get('isError', False)}}
            emit('tool_failed' if message.get('isError') else 'tool_completed', payload, pending.pop(call_id))

    def row(position, observed):
        f = dict(observed.snapshot(latest).values)
        f.update(prediction_position=position, configured_call_limit=None,
                 completed_call_count=completed, remaining_call_allowance=None,
                 current_request_input_known=0.0,
                 request_logged_utf8_bytes=len(json.dumps(request['request_messages'], ensure_ascii=False).encode())
                 if position == 'call_start' else None)
        if position != 'call_start':
            for key in ['current_request_tokens_local', 'request_message_count', 'request_tool_count',
                        'request_content_chars', 'request_system_utf8_bytes', 'request_delta_tokens']:
                f[key] = None
        r = {'task_id': tid, 'point_id': f'{tid}:{seq}:{position}', 'position': position,
             'completed_calls': completed, 'task_text': task, 'features': f}
        r['ir_features'] = grouped_project(observer.snapshot(r), observer.config)
        return r

    remaining = row('task_start' if completed == 0 else 'task_update',
                    state._start_logical_call('snapshot-only') if completed else state)
    emit('request_built', {'model': request['model'], 'message_count': len(request['request_messages'])},
         f'call:{completed}')
    upcoming = row('call_start', state)
    return [upcoming, remaining], completed, sorted(adapted)


def forecast(request, pipeline, entry):
    started = time.perf_counter()
    rows, completed, adapted = build_rows(request, pipeline.config['ir_features'])
    pred = pipeline.predict(rows)
    def horizon(i):
        value = float(pred['final'][i])
        if not math.isfinite(value):
            raise ValueError('Non-finite token forecast')
        return {'tokens': round(max(0, value)),
                'token_interval': [round(float(pred['lower'][i])), round(float(pred['upper'][i]))],
                'requirements': None,
                'requirements_unavailable_reason': 'checkpoint_has_no_requirement_head'}
    return {'next_call': horizon(0), 'remaining_task': horizon(1),
            'completed_calls': completed, 'inference_ms': round((time.perf_counter()-started)*1000, 2),
            'checkpoint': entry['path'], 'checkpoint_sha256': entry['sha256'],
            'token_definition': 'provider_input_including_cache_plus_output; reasoning_in_output',
            'warnings': ['Cross-harness/model transfer: trained on Opus mini-SWE-agent traces.',
                         'No Pi call cap is configured; cap features remain missing.'],
            'adapted_native_tools': adapted}


if __name__ == '__main__':
    pipeline, entry = load_model()
    for line in sys.stdin:
        request = {}
        try:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError('Expected a JSON request object')
            request = parsed
            response = {'id': request['id'], 'result': forecast(request, pipeline, entry)}
        except Exception as exc:
            response = {'id': request.get('id'), 'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(response, allow_nan=False), flush=True)
