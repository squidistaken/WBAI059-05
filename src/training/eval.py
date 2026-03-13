from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from src.data.data import AGNews, AGNewsWord2Vec
from rich.table import Table
from rich.panel import Panel
from src.const import DEBUG, LOGGER, DEVICE
from src.utils.error_analysis_pipeline import ErrorAnalysisPipeline
from typing import Any
import torch
import numpy as np


def evaluate_model(
    model: Any, ds: AGNews | AGNewsWord2Vec, use_test: bool = False
) -> None:
    """Evaluate a trained model on the dev set and display results.

    Args:
        model (Any): The model.
        ds (AGNews | AGNewsWord2Vec): The dataset.
        use_test (bool, optional): Whether to use the test set for evaluation.
                                   Defaults to False.
    """
    split_name = "Test" if use_test else "Dev"

    LOGGER.info(f"Evaluating {model.__class__.__name__} on {split_name} set")

    if isinstance(model, torch.nn.Module):
        # If the model is from PyTorch, run batched inference for prediction.
        split_key = "test" if use_test else "dev"
        torch_ds = ds.get_torch_dataset(split_key)
        y = ds.y_test if use_test else ds.y_dev

        model.eval()

        preds = []

        with torch.no_grad():
            for i in range(0, len(torch_ds.X), 64):
                batch = torch_ds.X[i : i + 64].float().to(DEVICE)
                logits = model(batch)

                preds.append(torch.argmax(logits, dim=1).cpu().numpy() + 1)

        y_pred = np.concatenate(preds)
        X = torch_ds.X
    else:
        # Else, predict using Sklearn on TF-IDF arrays.
        X, y = (ds.X_dev, ds.y_dev) if not use_test else (ds.X_test, ds.y_test)
        y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")

    panel = Panel(
        f"{model.__class__.__name__} Results on {split_name} Set",
        style="bold green",
    )

    LOGGER.log_and_print(panel)

    if DEBUG:
        print(y_pred.shape, y.shape, X.shape)

    table = Table(title="Evaluation Metrics")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Accuracy", f"{acc:.4f}")
    table.add_row("F1 Score (weighted)", f"{f1:.4f}")
    LOGGER.log_and_print(table)

    cm = confusion_matrix(y, y_pred)

    # Make Confusion Matrix readable by mapping label indices to class names.
    label_mapping = ds.label_mapping
    cm_table = Table(title="Confusion Matrix (with Class Names)")

    cm_table.add_column("Predicted \\ Actual", style="bold white")

    for i in range(cm.shape[0]):
        cm_table.add_column(label_mapping[i + 1], style="magenta")

    for i in range(cm.shape[0]):
        row = [label_mapping[i + 1]] + [
            str(cm[i, j]) for j in range(cm.shape[1])
        ]

        cm_table.add_row(*row)

    LOGGER.log_and_print(cm_table)


def analyze_model_errors(
    model: Any,
    ds: AGNews,
    split: str = "dev",
    min_examples: int = 10,
) -> None:
    """
    Run error analysis on model predictions.

    Args:
        model: Trained sklearn model or PyTorch nn.Module
        ds: AGNews dataset
        split: Dataset split to analyze ('dev' or 'test')
        min_examples: Minimum number of examples to show per error type
    """
    LOGGER.info(
        f"Running error analysis for {model.__class__.__name__} "
        f"on {split.upper()} set"
    )
    pipeline = ErrorAnalysisPipeline()
    pipeline.run(
        model=model,
        ds=ds,
        split=split,
        min_examples=min_examples,
        wrap_width=80,
        show_full_text=True,
    )
