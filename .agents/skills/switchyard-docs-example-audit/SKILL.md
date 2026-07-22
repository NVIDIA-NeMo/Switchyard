---
name: switchyard-docs-example-audit
description: Audit Switchyard documentation examples against source code and schemas. Use when reviewing Python imports and call shapes, CLI commands and defaults, profile configuration, environment variables, or HTTP request and response examples.
---

# Audit Switchyard Documentation Examples

Adapted from NVIDIA Tech Docs Skill Library's `doc-code-example-audit` workflow (Apache-2.0).

Read `AGENTS.md` and `docs/AGENTS.md` completely. This is read-only unless the user also asks to fix
the findings.

## Audit Checklist

### Python

- Verify imports use the canonical public package.
- Verify every referenced symbol exists and is publicly exported.
- Check constructor and method signatures against the current checkout.
- Mark illustrative types and pseudocode explicitly.

### CLI

- Read the owning parser definition or generated help.
- Check subcommand nesting, flag spelling, aliases, defaults, deprecations, and mutual exclusions.
- Confirm paths are interpreted as documented.

### Configuration

- Find the owning config model and loader.
- Check field names, nesting, accepted values, required fields, and defaults.
- Search other documentation for contradictory descriptions of the same field.

### HTTP and JSON

- Find the owning schema, wire type, translator, endpoint, and representative tests.
- Check field names, nesting, value types, and streaming versus non-streaming shape.
- Do not accept an example merely because it resembles an upstream provider format.

### Environment and Providers

- Confirm each environment variable is consumed by source.
- Verify precedence between explicit arguments, saved configuration, and environment fallbacks.
- Flag examples that imply credentials or live provider behavior without saying so.

## Findings

Report only verified mismatches. For each finding, include the documentation location, observed text,
authoritative source, user impact, and minimal correction. Separate confirmed defects from questions.
Say clearly when no material accuracy issues are found.
