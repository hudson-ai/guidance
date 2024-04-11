from __future__ import annotations

from functools import cached_property
from typing import Callable, Dict

import interegular

import guidance
from guidance import any_char_but, optional, select
from guidance._grammar import GrammarFunction

from ._char_range import nice_char_group

# Aliases, purely to make later type annotations readable
State = int
TransitionKey = int


class FSM(interegular.FSM):

    @cached_property
    def grammars(self) -> Dict[TransitionKey, GrammarFunction]:
        """
        Interregular's FSM.alphabet.by_transition maps a TransitionKey to the set of
        all characters that induce that transition. We convert this set of characters
        to a GrammarFunction.
        """
        alphabet = {
            char
            for char in self.alphabet.keys()
            if char != interegular.fsm.anything_else
        }

        grammars: Dict[TransitionKey, GrammarFunction] = {}
        for transition_key, chars in self.alphabet.by_transition.items():
            if interegular.fsm.anything_else in chars:
                assert [interegular.fsm.anything_else] == chars
                grammars[transition_key] = any_char_but(alphabet)
            else:
                grammars[transition_key] = nice_char_group(chars)

        return grammars

    def to_grammar(self) -> GrammarFunction:
        # Mapping of states to GrammarFunctions corresponding to running FSM from the given state.
        # This lies in closure of `build_func` and inner `func` in order to allow `func` to be
        # zero-arg, enabling `Placeholder`-based (mutual) recursion.
        funcs: Dict[State, Callable[[], GrammarFunction]] = {}

        def build_func(state: State) -> Callable[[], GrammarFunction]:
            # Mapping of transition keys to next state. This lies in the closure of `func`.
            transition: Dict[TransitionKey, State] = self.map[state]

            # Zero-arg grammar function (will be wrapped in guidance decorator after definition)
            def func(lm):
                options = []
                for transition_key, next_state in transition.items():
                    # Start with the grammar specified by the transition key
                    option = self.grammars[transition_key]
                    # If next state has no transitions out of it, we know `build_func`
                    # will return the null grammar
                    if self.map[next_state]:
                        # If we've already built the next state's grammar function, get it,
                        # otherwise build it and add to `funcs``
                        next_func = funcs.setdefault(next_state, build_func(next_state))
                        # Call the grammar function corresponding to the next state
                        option += next_func()
                    options.append(option)

                # Select from possible next grammars
                next_grammar = select(options)
                if state in self.finals:
                    # Termination condition
                    next_grammar = optional(next_grammar)

                return lm + next_grammar

            # Set name for repr
            func.__name__ = f"fsm_{state}"
            # Wrap in guidance decorator
            func = guidance(func, stateless=True, dedent=False)
            return func

        func: Callable[[], GrammarFunction] = build_func(self.initial)
        return func()

    @classmethod
    def from_pattern(cls, pattern: str) -> FSM:
        fsm = interegular.parse_pattern(pattern).to_fsm()
        return cls(
            alphabet=fsm.alphabet,
            states=fsm.states,
            initial=fsm.initial,
            finals=fsm.finals,
            map=fsm.map,
        )
