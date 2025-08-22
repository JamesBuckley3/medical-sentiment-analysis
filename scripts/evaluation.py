"""
This script defines a flexible `evaluate_model` function that calculates and displays key classification metrics and a confusion matrix for a trained model. It is designed to work with both scikit-learn pipelines and Hugging Face `Trainer` objects.

The process involves the following steps:
1.  **Conditional Prediction Logic**:
    -   The function first checks if the provided `model` is a Hugging Face `Trainer` instance. If so, it uses `model.predict()` on the `tokenized_datasets` to get predictions and labels.
    -   If not a `Trainer`, it checks for `X_test` and `y_test` and uses the standard scikit-learn `model.predict()` and `model.predict_proba()` methods.
    -   A `ValueError` is raised if neither condition is met, guiding the user on the correct inputs.
2.  **Calculate and Print Metrics**:
    -   Overall metrics, including **accuracy** and **F1 score**, are calculated and printed. The F1 score is specifically calculated for the 'positive' class.
    -   **Per-class metrics** (**precision**, **recall**, and **F1 score**) are computed for each label and printed individually.
3.  **Plot Confusion Matrix**:
    -   A **confusion matrix** is calculated to show the number of correct and incorrect predictions for each class.
    -   A visual plot of the confusion matrix is then displayed using `matplotlib`, providing a clear overview of the model's performance.
4.  **Return Values**: The function returns the true labels (`y_true`), predicted labels (`y_pred`), and optionally the prediction probabilities (`probs`), which can be used for further analysis.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from transformers import Trainer


def evaluate_model(
    model,
    X_test=None,
    y_test=None,
    tokenized_datasets=None,
    label_names=None,
    return_probs=False,
):
    """
    Works for both:
    - Hugging Face Trainer (requires tokenized_datasets)
    - scikit-learn pipeline (requires X_test, y_test)
    Returns (y_true, y_pred, probs or None)
    """
    # Case 1: Hugging Face Trainer
    if isinstance(model, Trainer) and tokenized_datasets is not None:
        preds = model.predict(tokenized_datasets["test"])
        y_true = preds.label_ids
        y_pred = np.argmax(preds.predictions, axis=-1)
        probs = preds.predictions if return_probs else None

    # Case 2: scikit-learn pipeline
    elif X_test is not None and y_test is not None:
        y_true = y_test
        y_pred = model.predict(X_test)
        probs = model.predict_proba(X_test) if return_probs else None

    else:
        raise ValueError(
            "Invalid inputs. Provide either (Trainer + tokenized_datasets) or (sklearn model + X_test, y_test)."
        )

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(
        y_true,
        y_pred,
        pos_label=(
            label_names.index("positive") if isinstance(label_names[0], str) else 1
        ),
    )
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Per-class metrics
    precision, recall, f1s, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(label_names))
    )
    for i, label in enumerate(label_names):
        print(f"\nClass: {label}")
        print(f" Precision: {precision[i]:.4f}")
        print(f" Recall:    {recall[i]:.4f}")
        print(f" F1 Score:  {f1s[i]:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

    return y_true, y_pred, probs
