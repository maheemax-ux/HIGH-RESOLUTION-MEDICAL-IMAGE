"""
Medical Image Augmentation Project
-----------------------------------
Generic high-resolution image augmentation pipeline.

Expected dataset:
dataset/
    train/
        class_1/
        class_2/
    val/
        class_1/
        class_2/
    test/
        class_1/
        class_2/

For a binary medical classification task, replace class_1/class_2
with meaningful labels such as normal/abnormal.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

IMAGE_SIZE = (512, 512)
SEED = 42

# Conservative medical-image augmentation.
# Do NOT blindly use horizontal flips when anatomical laterality matters.
train_transform = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.08,
        rotate_limit=0,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.4
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.10,
        contrast_limit=0.10,
        p=0.4
    ),
    A.OneOf([
        A.GaussNoise(std_range=(0.01, 0.03), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.Sharpen(p=1.0)
    ], p=0.20),
])

val_transform = A.Compose([])


def load_image(path, image_size=IMAGE_SIZE):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return image


def augment_image(path, output_dir, n=5):
    os.makedirs(output_dir, exist_ok=True)
    image = load_image(path)

    # Save original
    cv2.imwrite(os.path.join(output_dir, "original.png"), image)

    for i in range(n):
        augmented = train_transform(image=image)["image"]
        out = os.path.join(output_dir, f"augmented_{i+1}.png")
        cv2.imwrite(out, augmented)


def show_examples(path, n=6):
    image = load_image(path)
    images = [("Original", image)]

    for i in range(n - 1):
        aug = train_transform(image=image)["image"]
        images.append((f"Augmented {i+1}", aug))

    plt.figure(figsize=(12, 8))
    for i, (title, img) in enumerate(images):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample = "sample_images/sample.jpg"

    if os.path.exists(sample):
        show_examples(sample)
        augment_image(sample, "outputs/augmented", n=10)
        print("Augmented images saved to outputs/augmented/")
    else:
        print(
            "Add a medical image at sample_images/sample.jpg "
            "or change the sample path in this file."
        )
