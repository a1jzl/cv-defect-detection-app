"""Evaluation du modele entraine : mAP, precision, rappel par classe."""

import argparse
import logging

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation du modele de detection de defauts")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/dataset.yaml")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data)

    logger.info("mAP@0.5: %.3f", metrics.box.map50)
    logger.info("mAP@0.5:0.95: %.3f", metrics.box.map)
    logger.info("Precision moyenne: %.3f", metrics.box.mp)
    logger.info("Rappel moyen: %.3f", metrics.box.mr)


if __name__ == "__main__":
    main()
