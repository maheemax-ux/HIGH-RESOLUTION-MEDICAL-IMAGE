# High-Resolution Medical Image Augmentation

A complete educational project for applying conservative augmentation to high-resolution medical images and optionally training a baseline classifier.

## 1. Project goal
Increase training diversity while preserving medically meaningful structures.

## 2. Important safety principle
Augmentation is not a substitute for clinical validation. Every transformation should be justified for the modality and anatomy.

## 3. Dataset structure

dataset/
- train/
  - class_1/
  - class_2/
- val/
  - class_1/
  - class_2/
- test/
  - class_1/
  - class_2/

Use a patient-level split when multiple images belong to the same patient.

## 4. Installation

python -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate      # Windows

pip install -r requirements.txt

## 5. Augmentation

python train.py

Put one sample image at:
sample_images/sample.jpg

The script produces an original image and augmented examples.

## 6. Optional classifier

python train_model.py

The classifier is an educational baseline. For real research, consider transfer learning, class imbalance handling, calibration, external validation and explainability.

## 7. Evaluation

python evaluate.py

Report:
- Accuracy
- Sensitivity/Recall
- Specificity
- Precision
- F1-score
- ROC-AUC
- Confusion matrix

## 8. Medical considerations

Do not use transformations that can change disease appearance or anatomical meaning. Horizontal flipping may be inappropriate when left/right information matters. Keep validation and test images unaugmented.

This project is for education/research prototyping and is not a medical diagnostic system.
