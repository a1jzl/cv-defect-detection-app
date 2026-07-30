# CV Defect Detection App

Detection de defauts sur des pieces industrielles par vision par ordinateur, en s'appuyant sur le transfer learning avec YOLO, et deployee dans une application web interactive permettant de tester le modele sur de nouvelles images.

## 1. Contexte et objectif

Le controle qualite visuel est un cas d'usage frequent de la vision par ordinateur en industrie : identifier automatiquement des defauts (rayures, fissures, pieces manquantes) sur des images de production. Ce projet met en place un pipeline complet, du fine-tuning d'un modele de detection d'objets pre-entraine jusqu'a une interface de test accessible sans code.

## 2. Jeu de donnees

Dataset public de defauts de surface industriels (par exemple NEU Surface Defect Database ou un dataset Kaggle equivalent de controle qualite), annote au format YOLO (boites englobantes par classe de defaut).

## 3. Approche technique

Le modele part d'une architecture YOLOv8 pre-entrainee sur COCO, puis est affine (fine-tuning) sur le dataset de defauts industriels. Cette approche de transfer learning permet d'obtenir de bonnes performances avec un nombre d'images d'entrainement limite, ce qui correspond a une contrainte realiste en contexte industriel.

## 4. Stack technique

- Python 3.11
- ultralytics (YOLOv8) pour la detection d'objets
- OpenCV pour le pretraitement des images
- Gradio pour l'interface web interactive de demonstration
- Docker pour la containerisation du service d'inference

## 5. Structure du repo

```
cv-defect-detection-app/
  data/                      dataset annote au format YOLO (non versionne)
  src/
    prepare_dataset.py        conversion et validation des annotations
    train.py                    fine-tuning du modele YOLOv8
    evaluate.py                   calcul des metriques (mAP, precision, rappel)
    inference.py                    inference sur une image ou un dossier
  app/
    demo.py                      interface Gradio de demonstration
  Dockerfile
  requirements.txt
  .github/workflows/ci.yml
```

## 6. Installation et utilisation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/prepare_dataset.py --input data/raw
python src/train.py --epochs 100 --data data/dataset.yaml
python src/evaluate.py --weights runs/train/best.pt
python app/demo.py
```

## 7. Metriques suivies

Le script d'evaluation calcule le mAP@0.5, la precision et le rappel par classe de defaut, avec une matrice de confusion pour identifier les confusions frequentes entre types de defauts.

## 8. Interface de demonstration

L'application Gradio permet de deposer une image de piece industrielle et visualise en temps reel les defauts detectes avec leur boite englobante et leur score de confiance.

## 9. Limites et pistes d'amelioration

- Le dataset public utilise est plus petit et moins varie qu'un jeu de donnees industriel reel, ce qui limite la generalisation du modele.
- La detection de tres petits defauts pourrait beneficier d'une strategie de decoupage d'image en tuiles (tiling) avant inference.
- Le pipeline ne couvre pas encore le reentrainement continu a partir des nouvelles images annotees en production.

## 10. Auteur

Projet realise dans le cadre d'une recherche d'alternance en data/IA.
