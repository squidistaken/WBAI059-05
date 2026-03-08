from src.data.data import AGNewsWord2Vec, AGNewsWord2VecDataset
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
from typing import Optional
import torch

from pathlib import Path


class Assignment2Showcase:
    ds = AGNewsWord2Vec(path=DATA_DIR)

    def __call__(self, choice: Optional[int] = None):
        if choice is not None:
            if choice == 1:
                self.word_similarity()
            elif choice == 2:
                self.train_and_evaluate_cnn()
            elif choice == 3:
                self.analyze_cnn_errors()
            elif choice == 4:
                self.lstm_placeholder()
            elif choice == 5:
                self.ablation_study("max_sequence_length")
            return
        cli_menu(
            "Select a functionality to showcase:",
            {
                "Examine Word Similarity": self.word_similarity,
                "Train and Evaluate CNN Model": self.train_and_evaluate_cnn,
                "Analyze Errors on CNN Model": self.analyze_cnn_errors,
                "Train and Evaluate LSTM Model (Not Implemented)": self.lstm_placeholder,
                "Ablation Study on Sequence Length (Only CNN for Now)": lambda: self.ablation_study(
                    "max_sequence_length"
                ),
                "Back to Main Menu": lambda: LOGGER.log_and_print(
                    Panel("[bold yellow]Returning to Main Menu...[/bold yellow]")
                ),
            },
        )

    def word_similarity(self):
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

            neighbors = self.ds.nearest_neighbors(word, topn=10)
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

    def get_or_train_cnn_model(self):
        """Get the trained CNN model or train a new one if it doesn't exist."""
        output_dir = get_output_path(assignment=2)
        model_path = output_dir / "cnn_model.pt"

        # Setup Configuration for CNN
        cnn_config = {
            "embedding_dim": 100,
            "num_classes": 4,
            "num_filters": 100,
            "filter_sizes": [3, 4, 5, 6, 7],
            "dropout": 0.5,
        }

        cnn_model = CNNClassifier(config=cnn_config).to(DEVICE)

        if RETRAIN_MODEL or not model_path.exists():
            self._train_cnn(cnn_model, output_dir, model_path)
        else:
            LOGGER.info(f"Loading CNN model from {model_path}")
            try:
                cnn_model.load_state_dict(
                    torch.load(model_path, map_location=DEVICE, weights_only=True)
                )
            except RuntimeError as e:
                LOGGER.log_and_print(
                    Panel(f"[bold red]Error loading model: {e}[/bold red]")
                )
                LOGGER.log_and_print(
                    Panel(f"[bold yellow]Training new model...[/bold yellow]")
                )
                self._train_cnn(cnn_model, output_dir, model_path)
        cnn_model.eval()

        return cnn_model
    
    def get_or_train_lstm_model(self):
        """Get the trained CNN model or train a new one if it doesn't exist."""
        output_dir = get_output_path(assignment=2)
        model_path = output_dir / "lstm_model.pt"

        # Setup Configuration for LSTM

        lstm_config = {
            "vocab_size": 100,
            "num_classes": 4,
            "num_filters": 100,
            "dropout": 0.5,
        }

        lstm_model = LSTMClassifier(config=lstm_config).to(DEVICE)

        if RETRAIN_MODEL or not model_path.exists():
            self._train_cnn(lstm_model, output_dir, model_path)
        else:
            LOGGER.info(f"Loading LSTM model from {model_path}")
            try:
                lstm_model.load_state_dict(
                    torch.load(model_path, map_location=DEVICE, weights_only=True)
                )
            except RuntimeError as e:
                LOGGER.log_and_print(Panel(f"[bold red]Error loading model: {e}[/bold red]"))
                LOGGER.log_and_print(Panel(f"[bold yellow]Training new model...[/bold yellow]"))
                self._train_cnn(lstm_model, output_dir, model_path)
        lstm_model.eval()

        return lstm_model

    def train_and_evaluate_cnn(self, split: int | None = None):
        """Train (if needed) and evaluate the CNN model."""
        cnn_model = self.get_or_train_cnn_model()
        if split is not None:
            if split == 1:
                evaluate_model(cnn_model, self.ds, use_test=False)
            elif split == 2:
                evaluate_model(cnn_model, self.ds, use_test=True)
            return
        # Evaluate the Model
        cli_menu(
            "Evaluate CNN on which set?",
            {
                "Dev Set": lambda: evaluate_model(cnn_model, self.ds, use_test=False),
                "Test Set": lambda: evaluate_model(cnn_model, self.ds, use_test=True),
                "Back to Menu": lambda: None,
            },
        )

    def analyze_cnn_errors(self):
        """Run error analysis on the CNN model."""
        cnn_model = self.get_or_train_cnn_model()

        # Run Error Analysis
        cli_menu(
            "Analyze CNN errors for which split?",
            {
                "Dev Set": lambda: analyze_model_errors(
                    cnn_model, self.ds, split="dev", min_examples=10
                ),
                "Test Set": lambda: analyze_model_errors(
                    cnn_model, self.ds, split="test", min_examples=10
                ),
                "Back to Menu": lambda: None,
            },
        )

    def lstm_placeholder(self):
        # Placeholder for LSTM training/evaluation
        lstm_model = self.get_or_train_lstm_model()

        # Evaluate the Model
        cli_menu(
            "Evaluate CNN on which set?",
            {
                "Dev Set": lambda: evaluate_model(lstm_model, self.ds, use_test=False),
                "Test Set": lambda: evaluate_model(lstm_model, self.ds, use_test=True),
                "Back to Menu": lambda: None,
            },
        )

    def _train_cnn(
        self,
        cnn_model: CNNClassifier,
        output_dir: Path,
        model_path: Path,
        plot_path: Optional[Path] = None,
        max_series_len: int = 256,
        return_trainer: bool = True,
    ) -> Trainer | None:
        LOGGER.log_and_print(
            Panel("[bold yellow]Training CNN Classifier...[/bold yellow]")
        )

        if get_available_vram() > 16.0:
            train_data = self.ds.get_torch_dataset("train", max_length=max_series_len)
            dev_data = self.ds.get_torch_dataset("dev", max_length=max_series_len)
        else:
            LOGGER.log_and_print(
                Panel(
                    f"[bold red]Warning: Available VRAM is low ({get_available_vram():.2f} GB). Using (slow) memory efficient preprocessing for training.[/bold red]"
                )
            )
            train_data = AGNewsWord2VecDataset(
                path=DATA_DIR, split="train", max_length=max_series_len
            )
            dev_data = AGNewsWord2VecDataset(
                path=DATA_DIR, split="test", max_length=max_series_len
            )
        trainer = Trainer(
            model=cnn_model,
            train_data=train_data,
            eval_data=dev_data,
            batch_size=64,
        )

        # Train the CNN Model
        trainer.train(
            num_epochs=1, learning_rate=1e-3, early_stopping=True, patience=5
        )

        # Plot and save Training History
        if plot_path is None:
            plot_path = output_dir / "cnn_training_history.png"
        trainer.plot_history(show=False, save_path=str(plot_path))

        # Save the trained model
        trainer.save_model(model_path)
        
        acc = trainer.evaluate(lambda pred, labels: torch.sum(pred == (torch.argmax(labels, dim=1) + 1)) / len(labels), use_predict=True)

        LOGGER.log_and_print(
            Panel(
                f"[bold green]Training complete!\nHistory plot saved to {plot_path}\nModel saved to {model_path}\nAccuracy:{acc:.2f}[/bold green]"
            )
        )
        if return_trainer:
            return trainer

    def _train_lstm(self, lstm_model: LSTMClassifier, output_dir: Path, model_path: Path) -> None: 
        #TODO: reduce code duplication with _train_cnn (reason for seperate function is mainly to set lr and stuff differently for lstm, should be refactored)
        LOGGER.log_and_print(
            Panel("[bold yellow]Training LSTM Classifier...[/bold yellow]")
        )
        
        if get_available_vram() > 16.0:
            train_data = self.ds.get_torch_dataset("train")
            dev_data = self.ds.get_torch_dataset("dev")
        else:
            LOGGER.log_and_print(
                Panel(
                    f"[bold red]Warning: Available VRAM is low ({get_available_vram():.2f} GB). Using (slow) memory efficient preprocessing for training.[/bold red]"
                )
            )
            train_data = AGNewsWord2VecDataset(path=DATA_DIR, split="train")
            dev_data = AGNewsWord2VecDataset(path=DATA_DIR, split="test")
        trainer = Trainer(
            model=lstm_model,
            train_data=train_data,
            eval_data=dev_data,
            batch_size=64,
        )

        # Train the LSTM Model
        trainer.train(num_epochs=50, learning_rate=1e-7, early_stopping=True, patience=5)

        # Plot and save Training History
        plot_path = output_dir / "lstm_training_history.png"
        trainer.plot_history(show=False, save_path=str(plot_path))

        # Save the trained model
        trainer.save_model(model_path)

        LOGGER.log_and_print(
            Panel(
                f"[bold green]Training complete!\nHistory plot saved to {plot_path}\nModel saved to {model_path}[/bold green]"
            )
        )

    
    def ablation_study(self, parameter: str):
        # Placeholder for ablation study functionality

        if parameter == "max_sequence_length":
            # Example of how to modify the dataset for different max sequence lengths
            results = {}
            for max_length in track(
                [32, 64, 128, 256, 512],
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

                trainer = self._train_cnn(
                    cnn_model,
                    output_dir=get_output_path(assignment=2),
                    model_path=get_output_path(assignment=2)
                    / f"cnn_model_maxlen_{max_length}.pt",
                    plot_path=get_output_path(assignment=2) / f"cnn_training_history_maxlen_{max_length}.png",
                    max_series_len=max_length,
                )
                acc = trainer.evaluate(
                    lambda pred, labels: torch.sum(pred == labels) / len(labels),
                    use_predict=True,
                )
                results[max_length] = acc

            LOGGER.log_and_print(
                Panel(
                    "[bold cyan]Ablation Study Results:[/bold cyan]\n"
                    + "\n".join(
                        [
                            f"Max Length: {length:3d} | Accuracy: {acc:.4f}"
                            for length, acc in results.items()
                        ]
                    ),
                    style="bold green",
                )
            )

        else:
            LOGGER.log_and_print(
                Panel(
                    f"Ablation study for parameter '{parameter}' is not implemented yet."
                )
            )
