from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
from rich.table import Table
from rich.panel import Panel
import textwrap
from src.const import LOGGER, DEVICE
from src.data.data import AGNews, AGNewsWord2Vec
import torch


class ErrorAnalyzer:
    """Class to analyze and display misclassifications for a trained modal."""

    def __init__(
        self,
        model: Any,
        ds: AGNews | AGNewsWord2Vec,
        min_examples: int = 10,
        show_full_text: bool = False,
        wrap_width: int = 80,
    ) -> None:
        """Initialize the class.

        Args:
            model (Any): The trained model.
            ds (AGNews): The AGNews dataset object.
            min_examples (int, optional): The minimum number of examples to
                                          show for each error type. Defaults to
                                          10.
            show_full_text (bool, optional): Whether to show the full text of
                                             misclassified examples. Defaults
                                             to False.
            wrap_width (int, optional): The width to wrap text at. Defaults to
                                        80.
        """
        self.model = model
        self.ds = ds
        self.min_examples = min_examples
        self.show_full_text = show_full_text
        self.wrap_width = wrap_width
        self.X = None
        self.y = None
        self.df = None
        self.predictions = None
        self.confidence = None
        self.misclassifications = {}
        self.error_stats = {}
        self.current_split = None

    def analyze(self, split: str = "dev") -> Dict[Tuple[int, int], List[Dict]]:
        """Analyze misclassifications in a given dataset split.

        Args:
            split (str, optional): The dataset split to analyze. Defaults to
                                   "dev".

        Returns:
            Dict[Tuple[int, int], List[Dict]]: A dictionary mapping
                                               (predicted_label, actual_label)
                                               tuples to lists of misclassified
                                               examples.
        """
        self._load_split(split)
        self._generate_predictions()
        self._extract_misclassifications()
        self._compute_statistics()

        return self.misclassifications

    def _load_split(self, split: str) -> None:
        """Load the dataset split into X, y, and df attributes.

        Args:
            split (str): The dataset split to load.

        Raises:
            ValueError: If the split is unknown.
        """
        self.current_split = split

        match split:
            case "dev":
                self.y, self.df = self.ds.y_dev, self.ds.dev_df
                self.X = self.ds.X_dev if hasattr(self.ds, "X_dev") else None

            case "test":
                self.y, self.df = self.ds.y_test, self.ds.test_df
                self.X = self.ds.X_test if hasattr(self.ds, "X_test") else None

            case _:
                raise ValueError(f"Unknown split: {split}")

    def _generate_predictions(self) -> None:
        """Generate predictions and confidence scores for the split."""
        if isinstance(self.model, torch.nn.Module):
            # If the model is from PyTorch, run batched inference.
            self.model.eval()

            preds = []
            probs = []
            torch_ds = self.ds.get_torch_dataset(self.current_split)

            with torch.no_grad():
                for i in range(0, len(torch_ds.X), 64):
                    batch = torch_ds.X[i : i + 64].float().to(DEVICE)
                    logits = self.model(batch)

                    preds.append(torch.argmax(logits, dim=1).cpu().numpy() + 1)
                    probs.append(
                        torch.nn.functional.softmax(logits, dim=1)
                        .cpu()
                        .numpy()
                    )

            self.predictions = np.concatenate(preds)
            self.confidence = np.max(np.concatenate(probs), axis=1)
        else:
            # Else, analyze errors using SKLearn on TF-IDF arrays
            self.predictions = self.model.predict(self.X)

            if hasattr(self.model, "predict_proba"):
                self.confidence = np.max(
                    self.model.predict_proba(self.X), axis=1
                )

            elif hasattr(self.model, "decision_function"):
                decisions = self.model.decision_function(self.X)
                decisions = (
                    np.max(np.abs(decisions), axis=1)
                    if len(decisions.shape) > 1
                    else np.abs(decisions)
                )
                self.confidence = 1 / (1 + np.exp(-decisions))

            else:
                self.confidence = None

    def _extract_misclassifications(self) -> None:
        """Extract misclassified examples and group them by (predicted_label,
        actual_label)."""
        misclass_mask = self.predictions != self.y
        indices = np.where(misclass_mask)[0]
        label_map = self.ds.label_mapping
        texts = self.df["text"].to_list()
        errors = defaultdict(list)

        for idx in indices:
            actual, pred = self.y[idx], self.predictions[idx]

            errors[(pred, actual)].append(
                {
                    "text": texts[idx],
                    "actual_label": actual,
                    "predicted_label": pred,
                    "actual_class": label_map[actual],
                    "predicted_class": label_map[pred],
                    "confidence": (
                        self.confidence[idx]
                        if self.confidence is not None
                        else None
                    ),
                    "index": idx,
                }
            )

        self.misclassifications = dict(errors)

    def _compute_statistics(self) -> None:
        """Compute error statistics."""
        total = len(self.y)
        errors = sum(len(v) for v in self.misclassifications.values())

        self.error_stats = {
            "total_samples": total,
            "total_errors": errors,
            "error_rate": errors / total if total > 0 else 0,
            "accuracy": (total - errors) / total if total > 0 else 0,
            "error_types": len(self.misclassifications),
        }

    def display_summary(self, split: str = "dev") -> None:
        """Display error summary in a panel."""
        s = self.error_stats
        split_name = split.upper()
        panel = Panel(
            f"[bold cyan]Total Samples:[/][white] {s['total_samples']}[/]\n"
            f"[bold red]Total Errors:[/][white] {s['total_errors']}[/]\n"
            f"[bold yellow]Error Rate:[/][white] {s['error_rate']:.2%}[/]\n"
            f"[bold green]Accuracy:[/][white] {s['accuracy']:.2%}[/]",
            title=f"{self.model.__class__.__name__} Error Summary ({split_name})",
        )

        LOGGER.log_and_print(panel)

    def display_error_matrix(self) -> None:
        """Display a confusion matrix of errors in a table."""
        labels = self.ds.label_mapping
        n = len(labels)
        table = Table(title="Confusion Matrix (Errors Only)")

        table.add_column("Pred \\ Actual", style="bold white")

        for i in range(1, n + 1):
            table.add_column(labels[i], style="magenta")

        for p in range(1, n + 1):
            row = [labels[p]]

            for a in range(1, n + 1):
                count = len(self.misclassifications.get((p, a), []))
                row.append(str(count) if count > 0 else "-")

            table.add_row(*row)

        LOGGER.log_and_print(table)

    def display_error_group(
        self, pred: int, actual: int, examples: list, limit: int
    ) -> None:
        """Display a group of misclassified examples for a specific (predicted_label, actual_label) pair."""
        labels = self.ds.label_mapping
        title = f"Predicted: [red]{labels[pred]}[/] | Actual: [green]{labels[actual]}[/] ({len(examples)} cases)"
        table = Table(title=title, show_lines=True)

        table.add_column("Idx", width=6)
        table.add_column("Text")

        if self.confidence is not None:
            table.add_column("Conf", width=10)

        for ex in examples[:limit]:
            wrapped_text = textwrap.fill(ex["text"], width=self.wrap_width)
            row = [str(ex["index"]), wrapped_text]

            if self.confidence is not None:
                row.append(f"{ex['confidence']:.4f}")

            table.add_row(*row)

        LOGGER.log_and_print(table)

    def display_hardest_cases(self, min_examples: int = 10) -> None:
        """Get the hardest cases (lowest confidence predictions).

        Args:
            min_examples (int, optional): The minimum number of examples to display. Defaults to 10.

        Returns:
            None
        """
        if self.confidence is None:
            warning_panel = Panel(
                "[yellow]Warning: Confidence scores not available.[/yellow]",
            )
            LOGGER.log_and_print(warning_panel)
            return None

        # Flatten all misclassifications into a single list.
        all_errors = [
            ex
            for examples in self.misclassifications.values()
            for ex in examples
        ]
        hardest = sorted(
            all_errors,
            key=lambda x: (
                x["confidence"] if x["confidence"] is not None else 1.0
            ),
        )[:min_examples]

        table = Table(
            title="[bold red]Top 10 Lowest Confidence Errors[/bold red]",
            show_lines=True,
        )

        table.add_column("Pred \\ Actual", style="yellow", width=20)
        table.add_column("Text", style="white")
        table.add_column("Conf", style="red", width=8)

        for ex in hardest:
            wrapped = textwrap.fill(ex["text"], width=self.wrap_width)
            path = f"{ex['actual_class']} \\ {ex['predicted_class']}"

            table.add_row(path, wrapped, f"{ex['confidence']:.4f}")

        LOGGER.log_and_print(table)
