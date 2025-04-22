from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Union
from pydantic import BaseModel, Field

from ...trace import CaptureOutput

class CaptureVar(TypedDict):
    value: str
    log_prob: Optional[float]

class UsageMetrics(BaseModel):
    completion_tokens: int = 0
    prompt_tokens: int = 0

    completion_tokens_details: "CompletionTokensDetails" = Field(
        default_factory=lambda: CompletionTokensDetails()
    )
    prompt_tokens_details: "PromptTokensDetails" = Field(
        default_factory=lambda: PromptTokensDetails()
    )

class CompletionTokensDetails(BaseModel):
    fast_forward_tokens: int = 0

    def __add__(self, other: "CompletionTokensDetails") -> "CompletionTokensDetails":
        return CompletionTokensDetails(
            fast_forward_tokens=self.fast_forward_tokens + other.fast_forward_tokens
        )
class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0

    def __add__(self, other: "PromptTokensDetails") -> "PromptTokensDetails":
        return PromptTokensDetails(
            cached_tokens=self.cached_tokens + other.cached_tokens
        )

class State(ABC):
    def __init__(self) -> None:
        self.captures: dict[str, Union[CaptureVar, list[CaptureVar]]] = {}
        self.active_role: Optional[str] = None
        self.usage: UsageMetrics = UsageMetrics()

    @abstractmethod
    def __str__(self) -> str:
        pass

    def apply_capture(
        self, name: str, value: Optional[str], log_prob=Optional[float], is_append: bool = False
    ) -> CaptureOutput:
        if value is None:
            # A "reset" signal
            self.captures.pop(name)
        else:
            var = CaptureVar(value=value, log_prob=log_prob)
            if is_append:
                vars = self.captures.get(name, [])
                if not isinstance(vars, list):
                    vars = [vars]
                vars.append(var)
                self.captures[name] = vars
            else:
                self.captures[name] = var

        return CaptureOutput(
            name=name,
            value=value,
            log_probs=log_prob or float("nan"),
            is_append=is_append,
        )
