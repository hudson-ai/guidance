from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    List,
    Mapping,
    Set,
    TypeVar,
    Union,
)

# import interegular
import interegular.fsm

from guidance import any_char_but, guidance, optional, select
from guidance._grammar import Byte, GrammarFunction

from ._char_range import nice_char_group

# Aliases, purely to make later type annotations readable
State = int
TransitionKey = int


class _AnythingElse(Enum):
    # https://peps.python.org/pep-0484/#support-for-singleton-types-in-unions
    SENTINEL = 0


_anything_else = _AnythingElse.SENTINEL


class Alphabet(Dict[Union[str, _AnythingElse], TransitionKey]):

    @property
    def strs(self) -> Set[str]:
        return {b for b in self if b is not _anything_else}

    @property
    def by_transition(self) -> Dict[TransitionKey, List[Union[str, _AnythingElse]]]:
        by_transition = defaultdict(list)
        for b, t in self.items():
            by_transition[t].append(b)
        return by_transition

    def to_grammars(self) -> Dict[TransitionKey, GrammarFunction]:
        grammars: Dict[TransitionKey, GrammarFunction] = {}
        all_strs = self.strs
        for transition_key, chars in self.by_transition.items():
            if _anything_else in chars:
                # TODO: make class immutable and do this check at instantiation?
                assert chars == [_anything_else]
                grammars[transition_key] = any_char_but(forbidden=all_strs)
            else:
                grammars[transition_key] = nice_char_group(chars=chars)
        return grammars

    def to_interegular(self) -> interegular.fsm.Alphabet:
        alphabet: dict[Union[str, interegular.fsm.anything_else], TransitionKey] = {}
        for b, transition_key in self.items():
            if isinstance(b, str):
                s = b
            elif isinstance(b, _AnythingElse):
                s = interegular.fsm.anything_else
            else:
                raise TypeError(
                    f"Key {b} in alphabet is not of type Union[str, AnythingElse]: got {type(b)}"
                )
            alphabet[s] = transition_key
        return interegular.fsm.Alphabet(alphabet)

    @classmethod
    def from_interegular(
        cls, interegular_alphabet: interegular.fsm.Alphabet
    ) -> Alphabet:
        alphabet: Dict[Union[str, _AnythingElse], TransitionKey] = {}
        for chr, t in interegular_alphabet.items():
            if isinstance(chr, str):
                alphabet[chr] = t
            elif chr is interegular.fsm.anything_else:
                alphabet[_anything_else] = t
            else:
                raise TypeError(
                    f"Key {chr} in interegular alphabet is not of type Union[str, _AnythingElseCls]: got {type(chr)}"
                )
        return cls(alphabet)


class DFA:
    def __init__(
        self,
        alphabet: Mapping[Union[str, _AnythingElse], TransitionKey],
        states: Collection[State],
        initial: State,
        finals: Collection[State],
        map: Mapping[State, Mapping[TransitionKey, State]],
    ):
        self.alphabet = Alphabet(alphabet)
        self.states = states
        self.initial = initial
        self.finals = finals
        self.map = map

    def to_interegular(self) -> interegular.FSM:
        return interegular.FSM(
            alphabet=self.alphabet.to_interegular(),
            states=self.states,
            initial=self.initial,
            finals=self.finals,
            map=self.map,
        )

    @classmethod
    def from_interegular(cls, fsm: interegular.FSM):
        return cls(
            alphabet=Alphabet.from_interegular(fsm.alphabet),
            states=fsm.states,
            initial=fsm.initial,
            finals=fsm.finals,
            map=fsm.map,
        )

    @classmethod
    def from_pattern(cls, pattern: str) -> DFA:
        fsm = interegular.parse_pattern(pattern).to_fsm()
        return cls.from_interegular(fsm)

    def to_grammar(self) -> GrammarFunction:
        # Mapping of states to GrammarFunctions corresponding to running FSM from the given state.
        # This lies in closure of `build_func` and inner `func` in order to allow `func` to be
        # zero-arg, enabling `Placeholder`-based (mutual) recursion.
        funcs: Dict[State, Callable[[], GrammarFunction]] = {}
        grammars = self.alphabet.to_grammars()

        def build_func(state: State) -> Callable[[], GrammarFunction]:
            # Mapping of transition keys to next state. This lies in the closure of `func`.
            transition: Mapping[TransitionKey, State] = self.map[state]

            # Zero-arg grammar function (will be wrapped in guidance decorator after definition)
            def func(lm):
                options = []
                for transition_key, next_state in transition.items():
                    # Start with the grammar specified by the transition key
                    option = grammars[transition_key]
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
            func = guidance(func, stateless=True, dedent=False, cache=True)
            return func

        func: Callable[[], GrammarFunction] = build_func(self.initial)
        return func()
