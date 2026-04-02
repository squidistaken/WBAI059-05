from src.const import CONSOLE
from src.training.analyze_errors import ErrorAnalyzer
from typing import Any


class ErrorAnalysisPipeline:
    """Utility class to run the full error analysis pipeline for a model and dataset."""

    def __init__(self):
        """Initialize the pipeline."""
        self.console = CONSOLE

    def run(
        self,
        model: Any,
        ds: Any,
        split: str = "dev",
        min_examples: int = 10,
        wrap_width: int = 80,
        show_full_text: bool = True,
    ) -> None:
        """Run the error analysis pipeline.

        Args:
            model (Any): The trained model to analyze.
            ds (Any): The dataset object.
            split (str, optional): The dataset split to analyze. Defaults to
                                   "dev".
            min_examples (int, optional): The minimum number of examples to
                                          show for each error type. Defaults to
                                          10.
            wrap_width (int, optional): The width to wrap text at. Defaults to
                                        80.
            show_full_text (bool, optional): Whether to show the full text of
                                             misclassified examples. Defaults
                                             to True.
        """
        analyzer = ErrorAnalyzer(
            model,
            ds,
            min_examples=min_examples,
            wrap_width=wrap_width,
            show_full_text=show_full_text,
        )

        analyzer.analyze(split=split)
        analyzer.display_summary(split=split)
        analyzer.display_error_matrix()

        for (pred, actual), examples in sorted(
            analyzer.misclassifications.items(), key=lambda x: -len(x[1])
        ):
            analyzer.display_error_group(pred, actual, examples, min_examples)

        analyzer.display_hardest_cases(min_examples=min_examples)
