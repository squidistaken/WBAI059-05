from src.data.data import AGNewsWord2Vec, AGNewsWord2VecDataset
from src.models.cnn import CNNClassifier
from src.training.trainer import Trainer
from src.utils.output import get_output_path
from src.const import CONSOLE, DATA_DIR, RETRAIN_MODEL, LOGGER, DEVICE
from src.training.eval import evaluate_model, analyze_model_errors
from rich.panel import Panel
from src.utils.ui import cli_menu
from src.utils.data import get_available_vram
from typing import Optional
import torch


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
            return None
        else:
            cli_menu(
                "Select a functionality to showcase:",
                {
                    "Examine Word Similarity": self.word_similarity,
                    "Train and Evaluate CNN Model": self.train_and_evaluate_cnn,
                    "Analyze Errors on CNN Model": self.analyze_cnn_errors,
                    "Train and Evaluate LSTM Model (Not Implemented)": self.lstm_placeholder,
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
            "filter_sizes": [3, 4, 5],
            "dropout": 0.5,
        }

        cnn_model = CNNClassifier(config=cnn_config).to(DEVICE)

        if RETRAIN_MODEL or not model_path.exists():
            LOGGER.log_and_print(
                Panel("[bold yellow]Training CNN Classifier...[/bold yellow]")
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
                model=cnn_model,
                train_data=train_data,
                eval_data=dev_data,
                batch_size=64,
            )

            # Train the CNN Model
            trainer.train(num_epochs=50, learning_rate=1e-3, early_stopping=True, patience=3)

            # Plot and save Training History
            plot_path = output_dir / "cnn_training_history.png"
            trainer.plot_history(show=False, save_path=str(plot_path))

            # Save the trained model
            trainer.save_model(model_path)

            LOGGER.log_and_print(
                Panel(
                    f"[bold green]Training complete!\nHistory plot saved to {plot_path}\nModel saved to {model_path}[/bold green]"
                )
            )
        else:
            LOGGER.info(f"Loading CNN model from {model_path}")
            cnn_model.load_state_dict(
                torch.load(model_path, map_location=DEVICE, weights_only=True)
            )
            cnn_model.eval()

        return cnn_model

    def train_and_evaluate_cnn(self):
        """Train (if needed) and evaluate the CNN model."""
        cnn_model = self.get_or_train_cnn_model()

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
        panel = Panel(
            "LSTM training/evaluation functionality is not implemented yet.",
            style="bold yellow",
        )
        LOGGER.log_and_print(panel)
    
    
   

