"""Inference du modele entraine sur une image ou un dossier d'images."""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DefectDetector:
    """Encapsule le modele YOLO entraine pour l'inference."""

    def __init__(self, weights_path: str, confidence: float = 0.4) -> None:
        self.model = YOLO(weights_path)
        self.confidence = confidence

    def predict(self, image_path: str):
        results = self.model.predict(source=image_path, conf=self.confidence, verbose=False)
        return results[0]

    def predict_batch(self, folder: str) -> dict:
        predictions = {}
        for image_path in Path(folder).glob("*.jpg"):
            predictions[image_path.name] = self.predict(str(image_path))
        logger.info("Inference realisee sur %d images", len(predictions))
        return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference sur des images de defauts industriels")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    args = parser.parse_args()

    detector = DefectDetector(args.weights)
    result = detector.predict(args.source)
    logger.info("%d defauts detectes sur %s", len(result.boxes), args.source)


if __name__ == "__main__":
    main()
