from src.data.agnews2vec import AGNewsWord2Vec, AGNewsWord2VecDataset
from src.models.cnn import CNNClassifier
from src.models.lstm import LSTMClassifier
from src.training.trainer import Trainer
from src.utils.output import get_output_path
from src.const import CONSOLE, DATA_DIR, RETRAIN_MODEL, LOGGER, DEVICE
from src.training.eval import evaluate_model, analyze_model_errors
from rich.panel import Panel
from src.utils.ui import cli_menu
from rich.progress import track
from src.utils.data import get_available_vram
from typing import Optional, Union
import torch
from pathlib import Path


class Assignment2Showcase:
    """Class for assignment 2 showcase."""

    ds = AGNewsWord2Vec(path=DATA_DIR)

    def __call__(self, choice: Optional[int] = None) -> None:
        """Call the showcase.

        Args:
            choice (Optional[int], optional): The functionality to showcase.
                                              Defaults to None.
        """
        if choice is not None:
            if choice == 1:
                self.word_similarity()

            elif choice == 2:
                self.train_and_evaluate_cnn()

            elif choice == 3:
                self.train_and_evaluate_lstm()

            elif choice == 4:
                self.analyze_errors()

            elif choice == 5:
                self.ablation_study("max_sequence_length")

            return

        cli_menu(
            "Select a functionality to showcase:",
            {
                "Examine Word Similarity": self.word_similarity,
                "Train and Evaluate CNN Model": self.train_and_evaluate_cnn,
                "Train and Evaluate LSTM Model": self.train_and_evaluate_lstm,
                "Analyze Errors": self.analyze_errors,
                "Ablation Study on Sequence Length": lambda: self.ablation_study(
                    "max_sequence_length"
                ),
                "Back to Main Menu": lambda: LOGGER.log_and_print(
                    Panel(
                        "[bold yellow]Returning to Main Menu...[/bold yellow]"
                    )
                ),
            },
        )

    def word_similarity(self) -> None:
        """Showcase word similarity functionality."""
        while True:
            word = CONSOLE.input(
                "Enter a word to find its nearest neighbors (x to exit): "
            ).strip()

            if word.lower() == "x":
                LOGGER.log_and_print(
                    Panel(
                        "[bold yellow]Exiting Word Similarity Showcase...[/bold yellow]"
                    )
                )
                break

            neighbors = self.ds.nearest_neighbors(word, top_n=10)

            if neighbors:
                panel = Panel(
                    f"Nearest neighbors for [bold green]{word}[/bold green]:\n\n"
                    + "\n".join(
                        [
                            f"{neighbor[0]} (similarity: {neighbor[1]:.4f})"
                            for neighbor in neighbors
                        ]
                    ),
                    style="bold blue",
                )

                LOGGER.log_and_print(panel)

            else:
                panel = Panel(
                    f"No neighbors found for [bold red]{word}[/bold red]. It may not be in the vocabulary.",
                    style="bold red",
                )

                LOGGER.log_and_print(panel)

    def _get_or_train_cnn_model(self) -> CNNClassifier:
        """Get a trained CNN model or train a new one if it does not exist.

        Returns:
            CNNClassifier: The trained CNN model.
        """
        output_dir = get_output_path(assignment=2)
        model_path = output_dir / "cnn_model.pt"

        # Setup Configuration for CNN.
        cnn_config = {
            "embedding_dim": 100,
            "num_classes": 4,
            "num_filters": 100,
            "filter_sizes": [3, 4, 5, 6, 7],
            "dropout": 0.5,
        }

        cnn_model = CNNClassifier(config=cnn_config).to(DEVICE)

        if RETRAIN_MODEL or not model_path.exists():
            self._train_model(
                cnn_model, "CNN", output_dir, model_path, max_seq_len=256
            )

        else:
            LOGGER.info(f"Loading CNN model from {model_path}")

            try:
                cnn_model.load_state_dict(
                    torch.load(
                        model_path, map_location=DEVICE, weights_only=True
                    )
                )

            except RuntimeError as e:
                LOGGER.log_and_print(
                    Panel(f"[bold red]Error loading model: {e}[/bold red]")
                )
                LOGGER.log_and_print(
                    Panel(f"[bold yellow]Training new model...[/bold yellow]")
                )
                self._train_model(
                    cnn_model, "CNN", output_dir, model_path, max_seq_len=256
                )

        cnn_model.eval()

        return cnn_model

    def _get_or_train_lstm_model(self) -> LSTMClassifier:
        """Get a trained LSTM model or train a new one if it does not exist.

        Returns:
            LSTMClassifier: The trained LSTM model.
        """
        output_dir = get_output_path(assignment=2)
        model_path = output_dir / "lstm_model.pt"

        # Setup Configuration for LSTM.
        lstm_config = {
            "vocab_size": 100,
            "num_classes": 4,
            "num_filters": 100,
            "dropout": 0.5,
        }

        lstm_model = LSTMClassifier(config=lstm_config).to(DEVICE)

        if RETRAIN_MODEL or not model_path.exists():
            self._train_model(lstm_model, "LSTM", output_dir, model_path)

        else:
            LOGGER.info(f"Loading LSTM model from {model_path}")

            try:
                lstm_model.load_state_dict(
                    torch.load(
                        model_path, map_location=DEVICE, weights_only=True
                    )
                )

            except RuntimeError as e:
                LOGGER.log_and_print(
                    Panel(f"[bold red]Error loading model: {e}[/bold red]")
                )
                LOGGER.log_and_print(
                    Panel(f"[bold yellow]Training new model...[/bold yellow]")
                )

                self._train_model(lstm_model, "LSTM", output_dir, model_path)

        lstm_model.eval()

        return lstm_model

    def train_and_evaluate_cnn(self, split: Optional[int] = None) -> None:
        """Train and evaluate a CNN model.

        Args:
            split (Optional[int], optional): The split to evaluate on. Defaults
                                             to None.
        """
        cnn_model = self._get_or_train_cnn_model()

        if split is not None:
            if split == 1:
                evaluate_model(cnn_model, self.ds, use_test=False)

            elif split == 2:
                evaluate_model(cnn_model, self.ds, use_test=True)

            return

        cli_menu(
            "Evaluate CNN on which set?",
            {
                "Dev Set": lambda: evaluate_model(
                    cnn_model, self.ds, use_test=False
                ),
                "Test Set": lambda: evaluate_model(
                    cnn_model, self.ds, use_test=True
                ),
                "Back to Menu": lambda: None,
            },
        )

    def analyze_errors(self) -> None:
        """Run error analysis on the CNN and LSTM models."""
        cnn_model = self._get_or_train_cnn_model()
        lstm_model = self._get_or_train_lstm_model()

        cli_menu(
            "Analyze errors for which split?",
            {
                "Dev Set": lambda: (
                    analyze_model_errors(
                        cnn_model, self.ds, split="dev", min_examples=10
                    ),
                    analyze_model_errors(
                        lstm_model, self.ds, split="dev", min_examples=10
                    ),
                ),
                "Test Set": lambda: (
                    analyze_model_errors(
                        cnn_model, self.ds, split="test", min_examples=10
                    ),
                    analyze_model_errors(
                        lstm_model, self.ds, split="test", min_examples=10
                    ),
                ),
                "Back to Menu": lambda: None,
            },
        )

    def train_and_evaluate_lstm(self, split: Optional[int] = None) -> None:
        """Train and evaluate an LSTM model.

        Args:
            split (Optional[int], optional): The split to evaluate on. Defaults
                                             to None.
        """
        lstm_model = self._get_or_train_lstm_model()

        if split is not None:
            if split == 1:
                evaluate_model(lstm_model, self.ds, use_test=False)

            elif split == 2:
                evaluate_model(lstm_model, self.ds, use_test=True)

            return

        cli_menu(
            "Evaluate CNN on which set?",
            {
                "Dev Set": lambda: evaluate_model(
                    lstm_model, self.ds, use_test=False
                ),
                "Test Set": lambda: evaluate_model(
                    lstm_model, self.ds, use_test=True
                ),
                "Back to Menu": lambda: None,
            },
        )

    def _train_model(
        self,
        model: Union[CNNClassifier, LSTMClassifier],
        model_name: str,
        output_dir: Path,
        model_path: Path,
        plot_path: Optional[Path] = None,
        max_seq_len: Optional[int] = None,
    ) -> Trainer:
        """Train a model.

        Args:
            model (Union[CNNClassifier, LSTMClassifier]): The model to train.
            model_name (str): The name of the model.
            output_dir (Path): The output directory.
            model_path (Path): The path to save the model to.
            plot_path (Optional[Path], optional): The path to save the plot to.
                                                  Defaults to None.
            max_seq_len (Optional[int], optional): The maximum length of the
                                                   sequence. Defaults to None.

        Returns:
            Trainer: The trainer instance.
        """
        LOGGER.log_and_print(
            Panel(
                f"[bold yellow]Training {model_name} Classifier...[/bold yellow]"
            )
        )

        data_kwargs = {}
        if max_seq_len is not None:
            data_kwargs["max_length"] = max_seq_len

        if get_available_vram() > 16.0:
            train_data = self.ds.get_torch_dataset("train", **data_kwargs)
            dev_data = self.ds.get_torch_dataset("dev", **data_kwargs)
            test_data = self.ds.get_torch_dataset("test", **data_kwargs)

        else:
            LOGGER.log_and_print(
                Panel(
                    f"[bold red]Warning: Available VRAM is low ({get_available_vram():.2f} GB). Using (slow) memory efficient preprocessing for training.[/bold red]"
                )
            )

            train_data = AGNewsWord2VecDataset(
                path=DATA_DIR, split="train", **data_kwargs
            )
            dev_data = AGNewsWord2VecDataset(
                path=DATA_DIR, split="test", **data_kwargs
            )
            test_data = AGNewsWord2VecDataset(
                path=DATA_DIR, split="test", **data_kwargs
            )

        trainer = Trainer(
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            batch_size=64,
        )

        trainer.train(
            num_epochs=20, learning_rate=1e-3, early_stopping=True, patience=3
        )

        if plot_path is None:
            plot_path = (
                output_dir / f"{model_name.lower()}_training_history.png"
            )

        trainer.plot_history(show=False, save_path=str(plot_path))
        trainer.save_model(model_path)

        acc = trainer.evaluate(
            lambda pred, labels: torch.sum(
                pred == (torch.argmax(labels, dim=1) + 1)
            )
            / len(labels),
            use_predict=True,
        )

        LOGGER.log_and_print(
            Panel(
                f"[bold green]Training complete!\nHistory plot saved to {plot_path}\nModel saved to {model_path}\nAccuracy:{acc:.2f}[/bold green]"
            )
        )

        return trainer

    def ablation_study(self, parameter: str = "max_sequence_length") -> None:
        """Run an ablation study on a parameter.

        Args:
            parameter (str): The parameter to study. Defaults to
                            "max_sequence_length".
        """
        if parameter == "max_sequence_length":
            results = {
                "cnn": {"test": {}, "dev": {}},
                "lstm": {"test": {}, "dev": {}},
            }
            vals = [16, 32, 64, 128, 256]

            for max_length in track(
                vals,
                description="Running ablation study on max sequence length...",
            ):
                cnn_model = CNNClassifier(
                    config={
                        "embedding_dim": 100,
                        "num_classes": 4,
                        "num_filters": 100,
                        "filter_sizes": [3, 4, 5, 6, 7],
                        "dropout": 0.5,
                    }
                ).to(DEVICE)
                lstm_model = LSTMClassifier(
                    config={
                        "embedding_dim": 100,
                        "hidden_dim": 100,
                        "num_classes": 4,
                        "num_layers": 4,
                        "dropout": 0.5,
                    }
                ).to(DEVICE)

                # CNN training and logging.
                trainer = self._train_model(
                    model=cnn_model,
                    model_name="CNN",
                    output_dir=get_output_path(assignment=2),
                    model_path=get_output_path(assignment=2)
                    / f"cnn_model_maxlen_{max_length}.pt",
                    plot_path=get_output_path(assignment=2)
                    / f"cnn_training_history_maxlen_{max_length}.png",
                    max_seq_len=max_length,
                )
                accuracy = lambda pred, labels: torch.sum(
                    pred == (torch.argmax(labels, dim=1) + 1)
                ) / len(labels)
                acc_cnn_test = trainer.evaluate(
                    accuracy,
                    use_predict=True,
                    use_test=True,
                )
                acc_cnn_dev = trainer.evaluate(
                    accuracy,
                    use_predict=True,
                    use_test=False,
                )

                # LSTM training and logging.
                trainer = self._train_model(
                    model=lstm_model,
                    model_name="LSTM",
                    output_dir=get_output_path(assignment=2),
                    model_path=get_output_path(assignment=2)
                    / f"lstm_model_maxlen_{max_length}.pt",
                    plot_path=get_output_path(assignment=2)
                    / f"lstm_training_history_maxlen_{max_length}.png",
                    max_seq_len=max_length,
                )
                acc_lstm_test = trainer.evaluate(
                    accuracy,
                    use_predict=True,
                    use_test=True,
                )
                acc_lstm_dev = trainer.evaluate(
                    accuracy,
                    use_predict=True,
                    use_test=False,
                )
                results["cnn"]["test"][max_length] = acc_cnn_test
                results["cnn"]["dev"][max_length] = acc_cnn_dev
                results["lstm"]["test"][max_length] = acc_lstm_test
                results["lstm"]["dev"][max_length] = acc_lstm_dev

            LOGGER.log_and_print(
                Panel(
                    "[bold cyan]Ablation Study Results on Test:[/bold cyan]\n"
                    + "\n".join(
                        [
                            f"Max Sequence Length: {val} - CNN Accuracy: {results['cnn']['test'][val]:.4f}, LSTM Accuracy: {results['lstm']['test'][val]:.4f}"
                            for val in vals
                        ]
                    )
                    + "\n\n"
                    + "[bold cyan]Ablation Study Results on Dev:[/bold cyan]\n"
                    + "\n".join(
                        [
                            f"Max Sequence Length: {val} - CNN Accuracy: {results['cnn']['dev'][val]:.4f}, LSTM Accuracy: {results['lstm']['dev'][val]:.4f}"
                            for val in vals
                        ]
                    ),
                    style="bold green",
                )
            )

            with open(
                get_output_path(assignment=2) / "ablation_study_results.csv",
                "w",
            ) as f:
                f.write(
                    "model,max_sequence_length,accuracy_test,accuracy_dev\n"
                )

                for val in vals:
                    f.write(
                        f"cnn,{val},{results['cnn']['test'][val]:.4f},{results['cnn']['dev'][val]:.4f}\n"
                    )
                    f.write(
                        f"lstm,{val},{results['lstm']['test'][val]:.4f},{results['lstm']['dev'][val]:.4f}\n"
                    )

        else:
            LOGGER.log_and_print(
                Panel(
                    f"Ablation study for parameter '{parameter}' is not implemented yet."
                )
            )
