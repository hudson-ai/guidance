from lark import Lark, Transformer

grammar = """
?start      : expression

?expression : term 
            | expression ("|" term)+      -> or_expr

?term       : factor
            | term factor+          -> concat_expr

?factor     : atom quantifier?      -> quantified_expr

?atom       : char
            | "."                   -> any_char
            | "[" CHAR_SET "]"      -> char_set
            | "(" expression ")"
            | ESCAPED_CHAR          -> escaped_char

?quantifier : "*"                   -> zero_or_more
            | "+"                   -> one_or_more
            | "?"                   -> zero_or_one
            | "{" INT "}"           -> exact
            | "{" INT "," "}"       -> at_least
            | "{" INT "," INT "}"   -> range_quant

CHAR_SET: /[^\[\]]+/
ESCAPED_CHAR: /\\[tnrfaebBdDsSwW0-9]/
char: /[^|()[\\]{}.*+?\\-]/
INT: /\d+/
"""

# Define a Transformer to convert parse tree to AST
class RegexTransformer(Transformer):
    def or_expr(self, args):
        return OrExpression(args)

    def concat_expr(self, args):
        return ConcatExpression(args)

    def quantified_expr(self, args):
        expr, *quant = args
        if quant:
            return QuantifiedExpression(expr, quant[0])
        return expr

    def any_char(self, _):
        return AnyChar()

    def char_set(self, args):
        return CharSet(''.join(args))

    def char(self, args):
        return Character(args[0].value)

    def zero_or_more(self, _):
        return Quantifier(0, None)  # Equivalent to '*'

    def one_or_more(self, _):
        return Quantifier(1, None)  # Equivalent to '+'

    def zero_or_one(self, _):
        return Quantifier(0, 1)  # Equivalent to '?'

    def exact(self, args):
        count = int(args[0])
        return Quantifier(count, count)  # Equivalent to '{n}'

    def at_least(self, args):
        return Quantifier(int(args[0]), None)  # Equivalent to '{n,}'

    def range_quant(self, args):
        return Quantifier(int(args[0]), int(args[1]))  # Equivalent to '{n,m}'


# Define generic AST nodes
class OrExpression:
    def __init__(self, expressions):
        self.expressions = expressions

    def __repr__(self):
        return f"OrExpression({self.expressions})"

class ConcatExpression:
    def __init__(self, expressions):
        self.expressions = expressions

    def __repr__(self):
        return f"ConcatExpression({self.expressions})"

class QuantifiedExpression:
    def __init__(self, expression, quantifier):
        self.expression = expression
        self.quantifier = quantifier

    def __repr__(self):
        return f"QuantifiedExpression(expression={self.expression}, quantifier={self.quantifier})"

class AnyChar:
    def __repr__(self):
        return "AnyChar()"

class CharSet:
    def __init__(self, chars):
        self.chars = chars

    def __repr__(self):
        return f"CharSet(chars='{self.chars}')"

class Character:
    def __init__(self, char):
        self.char = char

    def __repr__(self):
        return f"Character(char='{self.char}')"

class EscapedChar:
    def __init__(self, char):
        self.char = char

    def __repr__(self):
        return f"EscapedChar(char='{self.char}')"

class Quantifier:
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __repr__(self):
        min_str = str(self.min) if self.min is not None else 'None'
        max_str = str(self.max) if self.max is not None else 'None'
        return f"Quantifier(min={min_str}, max={max_str})"


# Create the parser
parser = Lark(grammar, parser='lalr', transformer=RegexTransformer())

# Example usage
regex = "abcd" #"[abc]*|(d|e+f?)"
ast = parser.parse(regex)
print(ast)
