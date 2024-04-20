import pytest

from guidance import ebnf


class TestIntegerArithmetic:
    start = "expr"
    grammar_def = """
    expr    : expr "+" term     -> add
            | expr "-" term     -> sub
            | term

    term    : term "*" factor   -> mul
            | term "/" factor   -> div
            | factor

    factor  : integer
            | "(" expr ")"

    integer : DIGIT+
            | "-" integer       -> neg

    %import common.DIGIT
    """
    grammar = ebnf(grammar=grammar_def, start=start)

    def test_repr(self):
        # This test is just a reminder to reduce redundancies
        # in the graph and improve caching
        print(repr(self.grammar))
        assert len(repr(self.grammar).split("\n")) < 18

    @pytest.mark.parametrize(
        "matchstr", ["1+2+3+4", "1/2+3/4", "(1+2/3)*(4+(5+3/2))", "8/-4", "-9", "42"]
    )
    def test_good(self, matchstr):
        assert self.grammar.match(matchstr) is not None

    @pytest.mark.parametrize("matchstr", ["2+3+4(5)", "7(+8)", "8/*6"])
    def test_bad(self, matchstr):
        assert self.grammar.match(matchstr) is None
