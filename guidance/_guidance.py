import functools
import inspect
from typing import Any

from ._grammar import DeferredReference, RawFunction, Terminal, string
from ._utils import strip_multiline_string_indents
from .models import Model


def guidance(
    f = None,
    *,
    stateless = False,
    cache = False,
    dedent = True,
    model = Model,
):
    """Decorator used to define guidance grammars"""
    return _decorator(f, stateless=stateless, cache=cache, dedent=dedent, model=model)


_null_grammar = string("")


def _decorator(f, *, stateless, cache, dedent, model):

    # if we are not yet being used as a decorator, then save the args
    if f is None:
        return functools.partial(
            _decorator, stateless=stateless, cache=cache, dedent=dedent, model=model
        )

    # if we are being used as a decorator then return the decorated function
    else:

        # this strips out indentation in multiline strings that aligns with the current python indentation
        if dedent is True or dedent == "python":
            f = strip_multiline_string_indents(f)

        # we cache if requested
        if cache:
            f = functools.cache(f)

        placeholders = {}
        # Remove the first argument from the wrapped function
        signature = inspect.signature(f)
        params = list(signature.parameters.values())
        params.pop(0)
        signature = signature.replace(parameters=params)

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            arguments = normalize_args_kwargs(signature, args, kwargs)

            # make a stateless grammar if we can
            if stateless is True or (
                callable(stateless) and stateless(*args, **kwargs)
            ):

                # if we have a (deferred) reference set, then we must be in a recursive definition and so we return the reference
                try:
                    return placeholders[arguments]
                except (KeyError, TypeError):
                    pass

                # otherwise we call the function to generate the grammar

                # set a DeferredReference for recursive calls (only if we don't have arguments that might make caching a bad idea)
                try:
                    placeholders[arguments] = DeferredReference()
                except TypeError:
                    pass

                try:
                    # call the function to get the grammar node
                    node = f(_null_grammar, *args, **kwargs)
                except:
                    raise
                else:
                    if not isinstance(node, (Terminal, str)):
                        node.name = f.__name__
                    # set the reference value with our generated node
                    try:
                        placeholder = placeholders[arguments]
                    except TypeError:
                        pass
                    else:
                        placeholder.value = node
                finally:
                    try:
                        del placeholders[arguments]
                    except TypeError:
                        pass
                return node

            # otherwise must be stateful (which means we can't be inside a select() call)
            else:
                return RawFunction(f, args, kwargs)


        # attach the signature to the wrapped function
        wrapped.__signature__ = signature

        # attach this as a method of the model class (if given)
        # if model is not None:
        #     setattr(model, f.__name__, f)

        return wrapped


def normalize_args_kwargs(signature: inspect.Signature, args, kwargs) -> tuple[tuple[str, Any], ...]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return tuple(bound.arguments.items())
