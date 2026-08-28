from dataclasses import dataclass


LABELS = ("technology", "business", "science", "campus", "other")


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str

    def __post_init__(self) -> None:
        if not self.id or not self.text.strip():
            raise ValueError("id and non-empty text are required")
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    abstain: bool = False

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"unknown prediction label: {self.label}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True)
class EvaluationRow:
    example_id: str
    condition: str
    expected: str
    predicted: str
    confidence: float
    abstain: bool
    correct: bool
