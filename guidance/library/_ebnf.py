from collections import defaultdict
from typing import Callable

from lark import Lark
from lark.grammar import NonTerminal, Rule, Terminal

import guidance
from guidance import capture, regex, select
from guidance._grammar import GrammarFunction


class EBNF:
    def __init__(self, grammar: str, start: str):
        self.start = start
        self.parser = Lark(grammar, start=start)  # kwds?

        # grammars for nonterminals -- regex seems to be the simplest solution
        self.terminal_grammars: dict[str, GrammarFunction] = {
            terminal.name: regex(pattern=terminal.pattern.to_regexp())
            for terminal in self.parser.terminals
        }

        # Collect rules by nonterminal such that we can easily `select` over
        # the corresponding grammars
        self.rules_by_nonterminal: dict[str, list[Rule]] = defaultdict(list)
        for rule in self.parser.rules:
            self.rules_by_nonterminal[rule.origin.name].append(rule)

        # Callables to produce grammars for nonterminals -- I *think* they need to be callables
        # rather than literal grammars to avoid infinite recursion (taking advantage of late binding)
        # TODO: test this hypothesis
        self.nonterminal_grammar_callables: dict[str, Callable[[], GrammarFunction]] = (
            {}
        )

    def _build(self, name: str) -> Callable[[], GrammarFunction]:
        # No-arg function to be wrapped in guidance decorator.
        #   - Associated with exactly one nonterminal
        #   - Needs to be no-arg to allow for mutual recursion via `Placeholder`s
        #   - Wrap in guidance decorator later so that we can set the __name__ first
        def inner(lm):
            # Options to select over (one for each rule associated with a nonterminal)
            options = []
            for rule in self.rules_by_nonterminal[name]:
                # Form a `Join` over all terms in rule's expansion
                option = ""
                for term in rule.expansion:
                    if isinstance(term, Terminal):
                        grammar = self.terminal_grammars[term.name]
                    elif isinstance(term, NonTerminal):
                        grammar_callable = (
                            self.nonterminal_grammar_callables.setdefault(
                                term.name, self._build(term.name)
                            )
                        )
                        grammar = grammar_callable()
                    else:
                        raise RuntimeError("Something went very wrong")
                    option += grammar
                options.append(option)
            return lm + select(options)

        # Set name and wrap
        inner.__name__ = name
        return guidance(inner, stateless=True, dedent=False)

    def build(self) -> Callable[[], GrammarFunction]:
        # Trigger recursive build of grammar using start nonterminal
        return self._build(self.start)


@guidance(stateless=True)
def ebnf(lm, name=None, *, grammar: str, start: str):
    grammar_callable = EBNF(grammar, start).build()
    g = grammar_callable()
    return lm + capture(g, name=name)


def example():
    """
    Currently, `ebnf` relies on `regex`, which supports only a very limited regex syntax.
    Therefore we have to use a really simplified set of terminals here
    (e.g. DIGIT as opposed to NUMBER)
    """
    start = "sum"
    grammar_def = """
    sum     : product
            | sum "+" product   -> add
            | sum "-" product   -> subtract
    product : item
            | product "*" item  -> multiply
            | product "/" item  -> divide
    item    : DIGIT             -> number
            | "(" sum ")"       -> sum
    %import common.DIGIT
    """
    grammar = ebnf(grammar=grammar_def, start=start)

    assert grammar.match("1+2+3+4") is not None
    assert grammar.match("1/2+3/4") is not None
    assert grammar.match("(1+2/3)*(4+(5+3/2))") is not None
    assert grammar.match("2+3+4(5)") is None
    assert grammar.match("7(+8)") is None

    return grammar


if __name__ == "__main__":
    example()
