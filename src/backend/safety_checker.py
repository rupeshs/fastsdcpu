from typing import Any

from transformers import pipeline

from constants import SAFETY_CHECKER_MODEL


class SafetyChecker:
    """A class to check if an image is NSFW or not."""

    def __init__(
        self,
        mode_id: str = SAFETY_CHECKER_MODEL,
    ):
        self.classifier = pipeline(
            "image-classification",
            model=mode_id,
        )

    def is_safe(
        self,
        image: Any,
    ) -> bool:
        pred = self.classifier(image)
        scores = {label["label"]: label["score"] for label in pred}
        nsfw_score = scores.get("nsfw", 0)
        normal_score = scores.get("normal", 0)
        print(f"NSFW score: {nsfw_score}, Normal score: {normal_score}")
        return normal_score > nsfw_score
