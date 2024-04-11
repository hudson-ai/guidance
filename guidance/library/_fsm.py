from collections.abc import Collection, Mapping
from typing import Callable

import guidance
from guidance import optional, select
from guidance._grammar import GrammarFunction

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
