"""DSLError: a failure that carries the reason code that becomes the trial's verdict."""

from alpha_factory.contracts.evaluation import ReasonCode


class DSLError(Exception):
    def __init__(self, reason: ReasonCode, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message

    def __str__(self) -> str:
        return f"{self.reason.value}: {self.message}"
