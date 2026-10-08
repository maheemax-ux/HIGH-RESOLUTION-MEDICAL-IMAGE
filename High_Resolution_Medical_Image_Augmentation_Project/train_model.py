"""
Optional baseline classifier for the augmented dataset.

This example uses TensorFlow/Keras. It assumes:
dataset/train/<class_name>/*.jpg
dataset/val/<class_name>/*.jpg
dataset/test/<class_name>/*.jpg

For a serious medical project, keep the test set completely untouched.
Augmentation should be applied to training images only.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    seed=SEED,
)

num_classes = len(train_ds.class_names)

# Keras-side augmentation for the baseline.
# Keep this conservative and clinically justified.
augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.08),
    layers.RandomTranslation(0.03, 0.03),
    layers.RandomContrast(0.10),
], name="medical_augmentation")

model = models.Sequential([
    layers.Input(shape=(224, 224, 1)),
    augmentation,
    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.30),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(
        1 if num_classes == 2 else num_classes,
        activation="sigmoid" if num_classes == 2 else "softmax"
    )
])

loss = "binary_crossentropy" if num_classes == 2 else "sparse_categorical_crossentropy"

model.compile(
    optimizer="adam",
    loss=loss,
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "best_medical_model.keras",
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)

print("Training complete.")
