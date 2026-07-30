"""Application Gradio de demonstration du modele de detection de defauts."""

import gradio as gr
from PIL import Image

from src.inference import DefectDetector

WEIGHTS_PATH = "runs/train/defect_detection/weights/best.pt"

detector = DefectDetector(WEIGHTS_PATH)


def detect_defects(image: Image.Image) -> Image.Image:
    result = detector.model.predict(source=image, conf=0.4, verbose=False)[0]
    annotated = result.plot()
    return Image.fromarray(annotated[..., ::-1])


demo = gr.Interface(
    fn=detect_defects,
    inputs=gr.Image(type="pil", label="Image de la piece industrielle"),
    outputs=gr.Image(type="pil", label="Defauts detectes"),
    title="Detection de defauts industriels",
    description="Deposez une image de piece pour detecter automatiquement les defauts de surface.",
)

if __name__ == "__main__":
    demo.launch()
