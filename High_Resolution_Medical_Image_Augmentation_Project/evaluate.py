
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

model = tf.keras.models.load_model("best_medical_model.keras")

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=16,
    color_mode="grayscale",
    shuffle=False
)

y_true = np.concatenate([y.numpy() for _, y in test_ds])
pred = model.predict(test_ds)

if pred.shape[-1] == 1:
    y_prob = pred.ravel()
    y_pred = (y_prob >= 0.5).astype(int)
else:
    y_prob = pred[:, 1]
    y_pred = np.argmax(pred, axis=1)

print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=test_ds.class_names))

try:
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))
except ValueError:
    print("ROC-AUC could not be calculated for this test split.")
