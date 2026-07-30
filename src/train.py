"""Fine-tuning d'un modele YOLOv8 pre-entraine sur le dataset de defauts."""

import argparse
import logging

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning YOLOv8 sur le dataset de defauts industriels")
    parser.add_argument("--data", type=str, required=True, help="Chemin du fichier dataset.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    parser.add_argument("--img-size", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(args.base_model)
    logger.info("Modele de base charge: %s", args.base_model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        project="runs/train",
        name="defect_detection",
    )
    logger.info("Entrainement termine. Resultats: %s", results)


if __name__ == "__main__":
    main()
