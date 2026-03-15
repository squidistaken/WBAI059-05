from src.data.agnews2trans import AGNews2Trans, AGNews2TransDataset
from src.const import DATA_DIR, LOGGER, DEVICE, RETRAIN_MODEL
from src.utils.output import get_output_path
from src.utils.ui import cli_menu
from src.utils.data import get_available_vram
from typing import Optional
from rich.panel import Panel
from src.models.distilbert import DistilBERTClassifer
from src.training.trainer import Trainer
import torch
from pathlib import Path
from src.training.eval import evaluate_model


class Assignment3Showcase:
    """Class for assignment 3 showcase."""

    ds = AGNews2Trans(path=DATA_DIR)

    # TODO: Implement robustness/slice evaluations.
    def __call__(self, choice: Optional[int] = None) -> None:
        """Call the showcase.


        Args:
            choice (Optional[int], optional): The functionality to showcase.
                                              Defaults to None.
        """
        if choice is not None:
            if choice == 1:
                self.distilbert()

            if choice == 2:
                self.analyze_errors()

            return

        cli_menu(
            "Select a functionality to showcase:",
            {
                "Finetune and Evaluate DistilBERT": self.distilbert,
                "Analyze Errors": self.analyze_errors,
                "Back to Main Menu": lambda: LOGGER.log_and_print(
                    Panel(
                        "[bold yellow]Returning to Main Menu...[/bold yellow]"
                    )
                ),
            },
        )

    def distilbert(self) -> None:
        """Finetune and evaluate DistilBERT."""
        output_dir = get_output_path(assignment=3)
        model_path = output_dir / "distilbert_model.pt"

        if RETRAIN_MODEL or not model_path.exists():
            trainer = self._train_model(model_path=model_path)
            model = trainer.load_model(model_path)

        else:
            LOGGER.info(f"Loading LSTM model from {model_path}")
            model = DistilBERTClassifer().to(DEVICE)

            model.load_state_dict(
                torch.load(model_path, map_location=DEVICE, weights_only=True)
            )

        cli_menu(
            "Evaluate CNN on which set?",
            {
                "Dev Set": lambda: evaluate_model(
                    model, self.ds, use_test=False
                ),
                "Test Set": lambda: evaluate_model(
                    model, self.ds, use_test=True
                ),
                "Back to Menu": lambda: None,
            },
        )

    def _train_model(self, model_path: Path) -> Trainer:
        """Train the DistilBERT model.

        Args:
            model_path (Path): The path to save the model.

        Returns:
            Trainer: A trainer containing the trained model.
        """
        if get_available_vram() > 8.0:
            train_data = self.ds.get_torch_dataset("train")
            dev_data = self.ds.get_torch_dataset("dev")
            test_data = self.ds.get_torch_dataset("test")
            batch_size = 64

        else:
            LOGGER.log_and_print(
                Panel(
                    f"[bold red]Warning: Available VRAM is low ({get_available_vram():.2f} GB). Using (slow) memory efficient preprocessing for training.[/bold red]"
                )
            )

            train_data = AGNews2TransDataset(split="train")
            dev_data = AGNews2TransDataset(split="dev")
            test_data = AGNews2TransDataset(split="test")
            batch_size = 32

        model = DistilBERTClassifer().to(DEVICE)
        trainer = Trainer(
            model=model,
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            batch_size=batch_size,
        )

        trainer.train(
            num_epochs=20,
            learning_rate=2e-5,
            optimizer=torch.optim.AdamW,
            early_stopping=True,
            patience=3,
        )

        plot_path = get_output_path(3) / "distilbert_training_history.png"

        trainer.plot_history(show=False, save_path=str(plot_path))
        trainer.save_model(model_path)

        return trainer

    def analyze_errors(self): ...
