from src.data.agnews import AGNews
from src.training.eval import evaluate_model, analyze_model_errors
from src.training.train import train_model, get_model
from src.training.gridsearch import svm_gridsearch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
import numpy as np
from rich.panel import Panel
from src.const import DATA_DIR, RETRAIN_MODEL, LOGGER
from src.utils.ui import cli_menu
from typing import Optional


class Assignment1Showcase:
    """Class for assignment 1 showcase."""

    ds = AGNews(path=DATA_DIR)
    use_updated_models = True

    def __call__(self, choice: Optional[int] = None) -> None:
        """Call the showcase.

        Args:
            choice (Optional[int], optional): The functionality to showcase.
                                              Defaults to None.
        """
        if choice is not None:
            if choice == 1:
                self.train_and_evaluate()

            elif choice == 2:
                self.grid_search()

            elif choice == 3:
                self.analyze_errors()

            return

        cli_menu(
            "Select a functionality to showcase:",
            {
                "Train and Evaluate Baseline Models": self.train_and_evaluate,
                "Perform SVM Grid Search": self.grid_search,
                "Analyze Errors on Models": self.analyze_errors,
                "Back to Main Menu": lambda: LOGGER.log_and_print(
                    Panel(
                        "[bold yellow]Returning to Main Menu...[/bold yellow]"
                    )
                ),
            },
        )

    def train_and_evaluate(self) -> None:
        """Train baseline models and evaluate them on dev/test sets."""

        # Logistic Regression model.
        if RETRAIN_MODEL:
            panel = Panel(
                "Training: Logistic Regression Model...",
                style="bold yellow",
            )
            LOGGER.log_and_print(panel)
            logreg_model = train_model(
                LogisticRegression(max_iter=1000),
                self.ds,
                assignment=1,
            )

        else:
            logreg_model = get_model(
                LogisticRegression(max_iter=1000),
                self.ds,
                assignment=1,
            )

        # SVM model.
        if RETRAIN_MODEL:
            panel = Panel("Training: SVM...", style="bold yellow")
            LOGGER.log_and_print(panel)
            svm_model = (
                train_model(
                    LinearSVC(C=0.1),
                    self.ds,
                    assignment=1,
                )
                if self.use_updated_models
                else train_model(
                    SVC(kernel="linear", C=0.1),
                    self.ds,
                    assignment=1,
                )
            )

        else:
            svm_model = (
                get_model(
                    LinearSVC(C=0.1),
                    self.ds,
                    assignment=1,
                )
                if self.use_updated_models
                else get_model(
                    SVC(kernel="linear", C=0.1),
                    self.ds,
                    assignment=1,
                )
            )

        cli_menu(
            "Evaluate on which set?",
            {
                "Dev Set": lambda: (
                    evaluate_model(logreg_model, self.ds),
                    evaluate_model(svm_model, self.ds),
                ),
                "Test Set": lambda: (
                    evaluate_model(logreg_model, self.ds, use_test=True),
                    evaluate_model(svm_model, self.ds, use_test=True),
                ),
                "Back to Menu": lambda: None,
            },
        )

    def grid_search(self) -> None:
        """Perform SVM grid search to find the best hyperparameters."""
        panel = Panel(
            "WARNING: Running SVM Grid Search can take a long time.",
            style="bold red",
        )
        LOGGER.log_and_print(panel)
        param_grid = {"C": np.logspace(-3, 3, 7), "kernel": ["linear"]}

        svm_gridsearch(
            ds=self.ds,
            param_grid=param_grid,
            eval=True,
            assignment=1,
        )

    def analyze_errors(self):
        """Analyze model errors."""
        logreg_model = get_model(
            LogisticRegression(max_iter=1000),
            self.ds,
            assignment=1,
        )

        svm_model = (
            get_model(
                LinearSVC(C=0.1),
                self.ds,
                assignment=1,
            )
            if self.use_updated_models
            else get_model(
                SVC(kernel="linear", C=0.1),
                self.ds,
                assignment=1,
            )
        )

        cli_menu(
            "Analyze errors for which split?",
            {
                "Dev Set": lambda: (
                    analyze_model_errors(
                        logreg_model, self.ds, split="dev", min_examples=10
                    ),
                    analyze_model_errors(
                        svm_model, self.ds, split="dev", min_examples=10
                    ),
                ),
                "Test Set": lambda: (
                    analyze_model_errors(
                        logreg_model, self.ds, split="test", min_examples=10
                    ),
                    analyze_model_errors(
                        svm_model, self.ds, split="test", min_examples=10
                    ),
                ),
                "Back to Menu": lambda: None,
            },
        )
