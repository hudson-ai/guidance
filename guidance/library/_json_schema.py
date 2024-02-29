from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence, Literal

import guidance
from guidance.library import char_range, one_or_more, optional, zero_or_more

from .._grammar import GrammarFunction, select


@guidance(stateless=True)
def _gen_json_int(lm):
    pos_nonzero = char_range("1", "9") + zero_or_more(char_range("0", "9"))
    return lm + optional("-") + select(["0", pos_nonzero])


@guidance(stateless=True)
def _gen_json_number(lm):
    mantissa_int = _gen_json_int()
    mantissa_frac = "." + one_or_more(char_range("0", "9"))
    exponent = "e" + select(["", "+", "-"]) + one_or_more(char_range("0", "9"))

    return lm + mantissa_int + optional(mantissa_frac) + optional(exponent)


@guidance(stateless=True)
def _gen_json_string(lm):
    string_chars = select(
        [
            char_range("a", "z"),
            char_range("A", "Z"),
            char_range("0", "9"),
            *[c for c in "-_' ,.!?/[]{}():;"],
            "\\n",
            "\\t",
            "\\\\",
        ],
        recurse=True,
    )
    return lm + '"' + string_chars + '"'


@guidance(stateless=True)
def _gen_json_object(
    lm,
    *,
    properties: Mapping[str, Any],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
    mode: Literal["json", "python"]
):
    lm += "{"
    properties_added = 0
    for name, property_schema in properties.items():
        lm += '"' + name + '"'

        lm += ":"
        lm += _gen_json(
            json_schema=property_schema,
            definitions=definitions,
            mode=mode
        )
        properties_added += 1
        if properties_added < len(properties):
            lm += ","
    lm += "}"
    return lm


@guidance(stateless=True)
def _gen_json_array(
    lm,
    *,
    item_schema: Mapping[str, Any],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
    mode: Literal["json", "python"]
):
    lm += "["
    lm += optional(
        zero_or_more(_gen_json(json_schema=item_schema, definitions=definitions, mode=mode) + ",")
        + _gen_json(json_schema=item_schema, definitions=definitions, mode=mode)
    )
    lm += "]"
    return lm


@guidance(stateless=True)
def _process_anyOf(
    lm,
    *,
    anyof_list: Sequence[MutableMapping[str, Any]],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
    mode: Literal["json", "python"]
):
    options = [
        _gen_json(json_schema=item, definitions=definitions, mode=mode) for item in anyof_list
    ]
    return lm + select(options)


@guidance(stateless=True)
def _gen_json(
    lm,
    json_schema: Mapping[str, Any],
    definitions: Mapping[str, Callable[[], GrammarFunction]],
    mode: Literal["json", "python"],
):
    ANYOF_STRING = "anyOf"
    if ANYOF_STRING in json_schema:
        return lm + _process_anyOf(
            anyof_list=json_schema[ANYOF_STRING], definitions=definitions, mode=mode
        )

    REF_STRING = "$ref"
    object_schema = None
    if REF_STRING in json_schema:
        return lm + _get_definition(json_schema[REF_STRING], definitions)
    else:
        target_type = json_schema["type"]

    result = None
    if target_type == "null":
        if mode == "json":
            result = "null"
        elif mode == "python":
            return "None"
    elif target_type == "boolean":
        if mode == "json":
            result = select(["true", "false"])
        else:
            result = select(["True", "False"])
    elif target_type == "integer":
        result = _gen_json_int()
    elif target_type == "number":
        result = _gen_json_number()
    elif target_type == "string":
        result = _gen_json_string()
    elif target_type == "array":
        result = _gen_json_array(
            item_schema=json_schema["items"], definitions=definitions, mode=mode
        )
    elif target_type == "object":
        if object_schema is None:
            object_properties = json_schema["properties"]
        else:
            object_properties = object_schema["properties"]
        result = _gen_json_object(properties=object_properties, definitions=definitions, mode=mode)
    else:
        raise ValueError(f"Unsupported type in schema: {json_schema['type']}")

    return lm + result


@guidance(stateless=True)
def gen_json(lm, json_schema: Mapping[str, Any], name: Optional[str] = None, mode: Literal["json", "python"] = "json"):
    if mode not in ["json", "python"]:
        raise ValueError(f"Invalid value for 'mode'. Expected one of ['json', 'python'], got {mode!r}")
 
    _DEFS_KEY = "$defs"
    definitions = {}
    if _DEFS_KEY in json_schema:
        definitions = _build_definitions(json_schema[_DEFS_KEY], mode=mode)

    return lm + guidance.capture(_gen_json(json_schema, definitions, mode=mode), name=name)


def _build_definitions(
    raw_definitions: Mapping[str, Any], mode: Literal["json", "python"]
) -> Mapping[str, Callable[[], GrammarFunction]]:
    definitions = {}

    def build_definition(
        json_schema: Mapping[str, Any]
    ) -> Callable[[], GrammarFunction]:
        @guidance(stateless=True, dedent=False)
        def closure(lm):
            return lm + _gen_json(json_schema=json_schema, definitions=definitions, mode=mode)

        return closure

    definitions = {
        ref: build_definition(schema) for ref, schema in raw_definitions.items()
    }
    return definitions


@guidance(stateless=True)
def _get_definition(
    lm,
    reference: str,
    definitions: Mapping[str, Callable[[], GrammarFunction]],
):
    assert definitions is not None
    REF_START = "#/$defs/"
    assert reference.startswith(
        REF_START
    ), f"Reference {reference} must start with {REF_START}"

    target_name = reference[len(REF_START) :]
    definition = definitions[target_name]
    return lm + definition()
