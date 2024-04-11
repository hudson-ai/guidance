from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Callable

import interegular

import guidance
from guidance import any_char_but, optional, select
from guidance._grammar import GrammarFunction

from ._char_range import nice_char_group

# Aliases, purely to make later type annotations readable
State = int
TransitionKey = int


class FSM:
    def __init__(
        self,
        transition_map: Mapping[State, Mapping[TransitionKey, State]],
        initial: State,
        finals: Collection[State],
        grammars: Mapping[TransitionKey, GrammarFunction],
    ):
        self.transition_map = transition_map
        self.initial = initial
        self.finals = finals
        self.grammars = grammars

    def to_grammar(self) -> GrammarFunction:
        # Mapping of states to GrammarFunctions corresponding to running FSM from the given state.
        # This lies in closure of `build_func` and inner `func` in order to allow `func` to be
        # zero-arg, enabling `Placeholder`-based (mutual) recursion.
        funcs: Mapping[State, Callable[[], GrammarFunction]] = {}

        def build_func(state: State) -> Callable[[], GrammarFunction]:
            # Mapping of transition keys to next state. This lies in the closure of `func`.
            transition: Mapping[TransitionKey, State] = self.transition_map[state]

            # Zero-arg grammar function (will be wrapped in guidance decorator after definition)
            def func(lm):
                options = []
                for transition_key, next_state in transition.items():
                    # Start with the grammar specified by the transition key
                    option = self.grammars[transition_key]
                    # If next state has no transitions out of it, we know `build_func`
                    # will return the null grammar
                    if self.transition_map[next_state]:
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
    def from_interegular_fsm(cls, fsm: interegular.FSM) -> FSM:
        # TODO: consider making this the __init__, thinly wrapping interegular FSMs

        # Remove redundant states first
        fsm = fsm.reduce()
        alphabet = {
            char
            for char in fsm.alphabet.keys()
            if char != interegular.fsm.anything_else
        }
        grammars = {}
        for transition_key, chars in fsm.alphabet.by_transition.items():
            if interegular.fsm.anything_else in chars:
                assert [interegular.fsm.anything_else] == chars
                grammars[transition_key] = any_char_but(alphabet)
            else:
                grammars[transition_key] = nice_char_group(chars)

        return cls(
            transition_map=fsm.map,
            initial=fsm.initial,
            finals=fsm.finals,
            grammars=grammars,
        )

    @classmethod
    def from_pattern(cls, pattern: str) -> FSM:
        return cls.from_interegular_fsm(interegular.parse_pattern(pattern).to_fsm())


class Regex:
    # TODO: should be a subclass of FSM?

    def __init__(self, fsm: interegular.FSM):
        self.fsm = fsm

    @classmethod
    def from_pattern(cls, pattern: str) -> Regex:
        fsm = interegular.parse_pattern(pattern).to_fsm()
        return cls(fsm)

    def intersection(self, other: Regex) -> Regex:
        # TODO: make FSM a thin wrapper around interegular.FSM and put this method there
        fsm = self.fsm.intersection(other.fsm)
        return Regex(fsm)

    def __and__(self, other: Regex) -> Regex:
        return self.intersection(other)

    def to_fsm(self) -> FSM:
        return FSM.from_interegular_fsm(self.fsm)

    def to_grammar(self) -> GrammarFunction:
        return self.to_fsm().to_grammar()
