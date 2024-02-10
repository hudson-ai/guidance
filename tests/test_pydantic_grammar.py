import json
import pytest
from types import NoneType
from pydantic import TypeAdapter, BaseModel, Field

from guidance._grammar import GrammarFunction
from guidance._pydantic_grammar import gen_type
from guidance._parser import EarleyCommitParser, ParserException


def to_compact_json(target: any) -> str:
    # See 'Compact Encoding':
    # https://docs.python.org/3/library/json.html
    # Since this is ultimately about the generated
    # output, we don't need to worry about pretty printing
    # and whitespace
    return json.dumps(target, separators=(",", ":"))


def check_string_with_grammar(input_string: str, grammar: GrammarFunction):
    parser = EarleyCommitParser(grammar)

    print(f"Checking {input_string}")
    for c in input_string:
        print(f"Working on: {c}")
        print(f"Valid next bytes: {parser.valid_next_bytes()}")
        next_byte = bytes(c, encoding="utf8")
        print(f"Consuming: {next_byte}")
        parser.consume_byte(next_byte)


@pytest.mark.parametrize(
    "simple_json_string",
    [
        '"with_underscore"',
        '"ALLCAPS"',
        '"with a space"',
        '"MiXeD cAsInG"',
        '"with-hyphen"',
        '"Mix case_underscore-hyphens"',
        '"with a comma, in the string"',
    ],
)
def test_string_schema(simple_json_string):
    type = str

    # First sanity check what we're setting up
    TypeAdapter(type).validate_json(simple_json_string)

    # Now set up the actual conversion
    grammar = gen_type(type)
    parser = EarleyCommitParser(grammar)

    for c in simple_json_string:
        print(f"Working on: {c}")
        print(f"Valid next bytes: {parser.valid_next_bytes()}")
        next_byte = bytes(c, encoding="utf8")
        print(f"Consuming: {next_byte}")
        parser.consume_byte(next_byte)


@pytest.mark.parametrize(
    "json_int",
    [to_compact_json(x) for x in [0, 1, 100, 9876543210, 99, 737, 858, -1, -10, -20]],
)
def test_integer_schema(json_int):
    type = int

    # First sanity check what we're setting up
    TypeAdapter(type).validate_json(json_int)

    grammar = gen_type(type)
    check_string_with_grammar(json_int, grammar)


def test_simple_object():
    class Schema(BaseModel):
        name: str
        productId: int = Field(description="The unique identifier for a product")

    target_obj = dict(name="my product", productId=123)

    # First sanity check what we're setting up
    Schema.model_validate(target_obj)

    target_string = to_compact_json(target_obj)
    grammar = gen_type(Schema)
    check_string_with_grammar(target_string, grammar)


def test_nested_object():
    class Info(BaseModel):
        a: int
        b: int

    class Schema(BaseModel):
        name: str
        info: Info

    target_obj = dict(name="my product", info=dict(a=1, b=2))

    # First sanity check what we're setting up
    Schema.model_validate(target_obj)

    target_string = to_compact_json(target_obj)
    grammar = gen_type(Schema)
    check_string_with_grammar(target_string, grammar)


@pytest.mark.parametrize("json_list", [to_compact_json(x) for x in [[], [0], [34, 56], [1, 2, 3], [9, 8, 7, 6]]])
def test_integer_list(json_list):
    type = list[int]

    # First sanity check what we're setting up
    TypeAdapter(type).validate_json(json_list)

    grammar = gen_type(type)
    check_string_with_grammar(json_list, grammar)


@pytest.mark.parametrize("json_list", [to_compact_json(x) for x in [[], ["a"], ["b c", "d, e"]]])
def test_string_list(json_list):
    type = list[str]

    # First sanity check what we're setting up
    TypeAdapter(type).validate_json(json_list)

    grammar = gen_type(type)
    check_string_with_grammar(json_list, grammar)


@pytest.mark.parametrize("target_list", [[], [dict(a=1)], [dict(a=2), dict(a=3)]])
def test_object_list(target_list):
    class Schema(BaseModel):
        a: int
    type = list[Schema]

    # First sanity check what we're setting up
    target_string = TypeAdapter(type).validate_python(target_list)

    target_string = to_compact_json(target_list)
    grammar = gen_type(type)
    check_string_with_grammar(target_string, grammar)


def test_object_containing_list():
    class Schema(BaseModel):
        a: str
        b: list[int]

    target_obj = {
        "a": "some lengthy string of characters",
        "b": [1, 2, 3, 2312, 123],
    }

    # First sanity check what we're setting up
    Schema.model_validate(target_obj)

    target_string = to_compact_json(target_obj)
    grammar = gen_type(Schema)
    check_string_with_grammar(target_string, grammar)


@pytest.mark.parametrize(
    ["bad_list", "unexpected_char"],
    [
        ("[,]", b","),
        ("[,1]", b","),
        ("[1,]", b"]"),
        ("[1,2,]", b"]"),
        ("[0,1,2,3,]", b"]"),
        ("[0,,1]", b","),
    ],
)
def test_bad_int_list(bad_list: str, unexpected_char):
    type = list[int]

    # First sanity check what we're setting up
    TypeAdapter(type).validate_python([1,2,3])

    grammar = gen_type(type)
    parser = EarleyCommitParser(grammar)
    with pytest.raises(ParserException) as pe:
        for c in bad_list:
            next_byte = bytes(c, encoding="utf8")
            print(f"Consuming: {next_byte}")
            parser.consume_byte(next_byte)
    assert pe.value.current_byte == unexpected_char


@pytest.mark.parametrize("target_bool", [True, False])
def test_boolean(target_bool):
    type = bool

    # First sanity check what we're setting up
    TypeAdapter(type).validate_python(target_bool)

    grammar = gen_type(type)
    target_string = to_compact_json(target_bool)
    check_string_with_grammar(target_string, grammar)


@pytest.mark.parametrize(
    "target_number",
    # It appears that Inf and NaN are not actually part of the JSON spec
    [0, 1, -1, 134, -234762, 0.1, 1.0, -10.33, 452.342, 1.23e23, -1.2e-22],
)
def test_number(target_number):
    type = float

    # First sanity check what we're setting up
    TypeAdapter(type).validate_python(target_number)

    grammar = gen_type(type)
    target_string = to_compact_json(target_number)
    check_string_with_grammar(target_string, grammar)


def test_null():
    type = NoneType

    target_obj = None

    # First sanity check what we're setting up
    TypeAdapter(type).validate_python(target_obj)

    grammar = gen_type(type)
    target_string = to_compact_json(target_obj)
    check_string_with_grammar(target_string, grammar)