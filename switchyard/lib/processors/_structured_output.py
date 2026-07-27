# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict-mode JSON-schema helpers for OpenAI Structured Outputs.

Routing processors build their ``response_format`` payload via
:func:`build_response_format` and rely on
:func:`to_strict_openai_schema` to translate a pydantic
``model_json_schema()`` into the subset OpenAI accepts under
``strict: true``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

# Keywords that pydantic emits but OpenAI Structured Outputs' strict subset
# does not accept. We drop them from the wire schema.
_STRICT_DROP_KEYS: frozenset[str] = frozenset(
    {
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "format",
        "pattern",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "title",
    },
)

# JSON Schema keywords whose value is a name→schema mapping rather than a
# normal schema-annotation dict. Their child keys are user-defined names
# (property names, definition names) and must never be filtered against
# ``_STRICT_DROP_KEYS`` — a Pydantic model with a field literally named
# ``title`` would otherwise lose that property because ``title`` is also a
# JSON Schema annotation we strip. Recurse into the values; leave the keys
# untouched.
_NAME_MAP_KEYS: frozenset[str] = frozenset(
    {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    },
)


def to_strict_openai_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Convert a pydantic JSON schema to the OpenAI Structured Outputs subset.

    Strict mode requires every object to set ``additionalProperties: false``
    and list ALL of its property keys under ``required`` (even ones that have
    defaults in the pydantic model). Several JSON-Schema keywords are not
    supported and must be stripped.
    """
    result = _strictify(model_cls.model_json_schema())
    assert isinstance(result, dict)
    return result


def _strictify(node: Any) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _STRICT_DROP_KEYS:
                continue
            if key in _NAME_MAP_KEYS and isinstance(value, dict):
                # ``properties`` / ``$defs`` / etc.: child keys are
                # user-defined property/definition names. Recurse into
                # the values but preserve every key verbatim.
                out[key] = {k: _strictify(v) for k, v in value.items()}
            else:
                out[key] = _strictify(value)
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(node, list):
        return [_strictify(item) for item in node]
    return node


def build_response_format(
    schema: type[BaseModel] | None,
    mode: Literal["json_schema", "json_object"],
) -> dict[str, Any]:
    """Build the ``response_format`` kwarg for ``acompletion(...)``.

    ``json_schema`` (preferred) returns the strict Structured Outputs
    payload built from ``schema``. Falls back to ``{"type":
    "json_object"}`` when ``mode="json_object"`` or no schema is
    supplied — useful for backends that don't advertise Structured
    Outputs support.
    """
    if mode == "json_object" or schema is None:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": to_strict_openai_schema(schema),
        },
    }


__all__ = [
    "build_response_format",
    "to_strict_openai_schema",
]
