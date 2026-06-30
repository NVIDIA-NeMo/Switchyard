---
name: Bug report
about: Report a defect in Switchyard
title: "[bug] "
labels: bug
assignees: ''
---

## Symptom

What happened? One or two sentences.

## Reproduction

Minimal steps to reproduce. Include the command line, the inbound request shape, and the configured recipe.

```bash
# example
switchyard serve --config profiles.yaml --port 4000
curl -s http://localhost:4000/v1/chat/completions -d '{"model":"...","messages":[...]}'
```

## Expected vs. actual

- **Expected:** what should have happened.
- **Actual:** what did happen. Paste the error / log output.

## Environment

- Switchyard version (or commit SHA):
- Python version (`python --version`):
- OS / arch:
- Install path (`uv sync`, `pip install nemo-switchyard`, source build, etc.):
- Inbound format (Chat Completions / Anthropic Messages / Responses):
- Backend (OpenAI / Anthropic / NVIDIA Inference Hub / other):

## Additional context

Anything else that helps — config snippet, related issue, screenshot.
