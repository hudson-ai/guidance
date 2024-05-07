from typing import NamedTuple, Optional

from lark import Lark, Transformer

from .._grammar import Join, select, string
from .._guidance import guidance
from ._any_char import any_char
from ._any_char_but import any_char_but
from ._char_range import char_range
from ._char_set import char_set
from ._optional import optional
from ._zero_or_more import zero_or_more

grammar = """
?start      : expression

?expression : term
            | expression ("|" term)+      -> or_expr

?term       : factor
            | term factor+          -> concat_expr

?factor     : atom quantifier?      -> quantified_expr

?atom       : char | special_char
            | "."                   -> any_char
            | "[" CHAR_SET "]"      -> char_set
            | "[^" CHAR_SET "]"     -> any_char_but
            | "(" expression ")"

?quantifier : "*"                   -> zero_or_more
            | "+"                   -> one_or_more
            | "?"                   -> zero_or_one
            | "{" INT "}"           -> exact
            | "{" INT "," "}"       -> at_least
            | "{" INT "," INT "}"   -> range_quant

?special_char: "\\d"                 -> digit
            
CHAR_SET: /[^\[\]]+/
char: /[^|()[\\]{}.*+?\\-]/
INT: /\d+/
"""


# Define a Transformer to convert parse tree to AST
class RegexTransformer(Transformer):
    def or_expr(self, args):
        return select(args)

    def concat_expr(self, args):
        return Join(args)

    def quantified_expr(self, args):
        expr, *quant = args
        if quant:
            [quant] = quant
            return quantified_expr(expr, quant)
        return expr

    def any_char(self, _):
        return any_char()

    def any_char_but(self, args):
        return any_char_but(args)

    def char_set(self, args):
        [s] = args
        return char_set(s)

    def char(self, args):
        [s] = args
        return string(s)

    def digit(self, _):
        return char_range("0", "9")

    def zero_or_more(self, _):
        return Quantifier(0, None)

    def one_or_more(self, _):
        return Quantifier(1, None)

    def zero_or_one(self, _):
        return Quantifier(0, 1)

    def exact(self, args):
        [count] = args
        return Quantifier(int(count), int(count))

    def at_least(self, args):
        [count] = args
        return Quantifier(int(count), None)

    def range_quant(self, args):
        min, max = args
        return Quantifier(int(min), int(max))


class Quantifier(NamedTuple):
    min: int
    max: Optional[int]


@guidance(stateless=True)
def quantified_expr(lm, expr, quantifier: Quantifier):
    min, max = quantifier
    for _ in range(min):
        lm += expr
    if max is None:
        return lm + zero_or_more(expr)
    for _ in range(max - min):
        lm += optional(expr)
    return lm


# Create the parser
_parser = Lark(grammar, parser="lalr", transformer=RegexTransformer())


@guidance(stateless=True)
def regex(lm, pattern):
    try:
        return lm + _parser.parse(pattern)
    except:
        print(pattern)
        raise
