# -*- coding: utf-8 -*-
"""Show train/test CIFAR-10 examples with predictions and true labels."""

import os
import pickle
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("USERPROFILE", str(ROOT))
os.environ.setdefault("HOME", str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from utils import load_dataset, prep_pixels


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

OUTPUT_DIR = ROOT / "runs" / "plots" / "prediction_examples"


def load_cifar10_from_archive():
    archive_path = ROOT / ".keras" / "datasets" / "cifar-10-batches-py.tar.gz"
    if not archive_path.exists():
        raise FileNotFoundError(f"CIFAR-10 archive was not found: {archive_path}")

    def load_batch_from_tar(tar, member_name):
        member = tar.getmember(f"cifar-10-batches-py/{member_name}")
        with tar.extractfile(member) as f:
            batch = pickle.load(f, encoding="bytes")
        images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[b"labels"])
        return images, labels

    train_images = []
    train_labels = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for index in range(1, 6):
            images, labels = load_batch_from_tar(tar, f"data_batch_{index}")
            train_images.append(images)
            train_labels.append(labels)
        testX, testY = load_batch_from_tar(tar, "test_batch")

    X_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    trainX, valX, trainY, valY = train_test_split(
        X_train,
        y_train,
        test_size=5000,
        random_state=1234,
    )
    return trainX, to_categorical(trainY), valX, to_categorical(valY), testX, to_categorical(testY)


def load_dataset_safely():
    try:
        return load_dataset()
    except PermissionError:
        return load_cifar10_from_archive()


def labels_from_one_hot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=1)


def predict_labels(model, images):
    probabilities = model.predict(images, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    return predicted_labels, confidences


def select_correct_and_incorrect_examples(normalized_images, one_hot_labels, model, per_group=4):
    true_labels = labels_from_one_hot(one_hot_labels)
    predicted_labels, confidences = predict_labels(model, normalized_images)
    correct_indices = np.where(predicted_labels == true_labels)[0][:per_group]
    incorrect_indices = np.where(predicted_labels != true_labels)[0][:per_group]

    if len(correct_indices) < per_group:
        raise ValueError(f"Only found {len(correct_indices)} correct examples.")
    if len(incorrect_indices) < per_group:
        raise ValueError(f"Only found {len(incorrect_indices)} incorrect examples.")

    selected_indices = np.concatenate([correct_indices, incorrect_indices])
    return selected_indices, predicted_labels, true_labels, confidences


def plot_examples(raw_images, normalized_images, one_hot_labels, model, title, output_path, per_group=4):
    selected_indices, predicted_labels, true_labels, confidences = select_correct_and_incorrect_examples(
        normalized_images,
        one_hot_labels,
        model,
        per_group=per_group,
    )

    cols = 4
    rows = 2
    plt.figure(figsize=(cols * 3.2, rows * 3.4))

    for plot_index, data_index in enumerate(selected_indices):
        ax = plt.subplot(rows, cols, plot_index + 1)
        ax.imshow(raw_images[data_index])
        ax.axis("off")

        pred = CLASS_NAMES[predicted_labels[data_index]]
        true = CLASS_NAMES[true_labels[data_index]]
        color = "green" if predicted_labels[data_index] == true_labels[data_index] else "red"
        row_label = "correct" if plot_index < per_group else "incorrect"
        ax.set_title(
            f"{row_label}\npred: {pred} ({confidences[data_index]:.2f})\ntrue: {true}",
            color=color,
            fontsize=10,
        )

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "runs" / "models" / "best_model_seed_0"
    model = tf.keras.models.load_model(model_path)

    trainX, trainY, valX, valY, testX, testY = load_dataset_safely()
    trainX_norm, valX_norm, testX_norm = prep_pixels(trainX, valX, testX)

    train_path = OUTPUT_DIR / "train_prediction_examples_seed_0.png"
    test_path = OUTPUT_DIR / "test_prediction_examples_seed_0.png"
    plot_examples(
        trainX,
        trainX_norm,
        trainY,
        model,
        "Train examples: prediction vs true label (best model seed=0)",
        train_path,
    )
    plot_examples(
        testX,
        testX_norm,
        testY,
        model,
        "Test examples: prediction vs true label (best model seed=0)",
        test_path,
    )

    print("Created files:")
    print(train_path)
    print(test_path)


if __name__ == "__main__":
    main()
