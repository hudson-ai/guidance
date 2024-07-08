import numpy as np

from ._model import Engine, Model, Chat
from ._byte_tokenizer import ByteTokenizer

class MockEngine(Engine):
    def __init__(self, tokenizer, byte_pattern, compute_log_probs):
        super().__init__(tokenizer, compute_log_probs=compute_log_probs)

        self._valid_mask = np.zeros(len(tokenizer.tokens))
        for i, t in enumerate(tokenizer.tokens):
            try:
                t.decode("utf8")
                self._valid_mask[i] = 1.0
            except:
                pass
        self.called_temperatures = []

        # allow for strings to be passed
        if isinstance(byte_pattern, str):
            byte_pattern = byte_pattern.encode("utf8")
        self.byte_pattern = byte_pattern

        # seed the random number generator
        self._rand_generator = np.random.default_rng(seed=42)

    def get_logits(self, token_ids, forced_bytes, current_temp):
        """Pretends to compute the logits for the given token state."""
        return (
            self._rand_generator.standard_normal(len(self.tokenizer.tokens))
            * self._valid_mask
        )

    def get_next_token(self, prompt_tokens: list[int], token_mask: bytes, temperature: float) -> int:
        self.called_temperatures.append(temperature)
        if self.byte_pattern is not None:
            # If we have a byte pattern, we should use it to determine the next token
            # (forced even if invalid) otherwise, we will fall back to random valid tokens

            # build the byte string from the tokens
            bstr = self.tokenizer.decode(prompt_tokens)

            if self.byte_pattern.startswith(bstr) and len(bstr) < len(self.byte_pattern):
                # TODO: only tokenize head to avoid repeated work
                next_token = self.tokenizer.encode(self.byte_pattern[len(bstr) :])[0]
                # Note: may force a bad token -- we will get an exception from the engine
                return next_token

            # Better MockException?
            raise Exception("Byte pattern does not match")

        return super().get_next_token(prompt_tokens, token_mask, temperature)


class Mock(Model):
    def __init__(
        self,
        byte_pattern=None,
        echo=True,
        compute_log_probs=False,
    ):
        """Build a new Mock model object that represents a model in a given state."""

        # TODO: allow other tokenizers? May be useful for testing
        tokenizer = ByteTokenizer()
        engine = MockEngine(tokenizer, byte_pattern, compute_log_probs)

        super().__init__(engine, echo=echo)


class MockChat(Mock, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
