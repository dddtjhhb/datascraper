from typing import Protocol

from .schema import Prediction


class Model(Protocol):
    def predict(self, text: str) -> Prediction: ...


class KeywordBaseline:
    """Transparent CPU-only baseline used to validate the evaluation system."""

    KEYWORDS = {
        "technology": {"software", "computer", "ai", "chip", "internet", "database"},
        "business": {"company", "market", "revenue", "stock", "bank", "investment"},
        "science": {"research", "scientist", "study", "laboratory", "physics", "biology"},
        "campus": {"student", "campus", "university", "course", "professor", "library"},
    }

    def __init__(self, abstain_threshold: float = 0.45):
        self.abstain_threshold = abstain_threshold

    def predict(self, text: str) -> Prediction:
        tokens = {token.strip(".,:;!?()[]\"'").lower() for token in text.split()}
        scores = {label: len(tokens & words) for label, words in self.KEYWORDS.items()}
        label, score = max(scores.items(), key=lambda item: item[1])
        total = sum(scores.values())
        if score == 0:
            return Prediction("other", 0.25, True)
        confidence = min(0.95, 0.40 + 0.15 * score + 0.05 * score / max(total, 1))
        return Prediction(label, confidence, confidence < self.abstain_threshold)
