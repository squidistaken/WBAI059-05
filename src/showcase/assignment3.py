from src.data.agnews2trans import AGNews2Trans, AGNews2TransDataset
from src.data.transformed_dataset import TransformedDataset
from src.const import LOGGER, DEVICE, RETRAIN_MODEL
from src.utils.output import get_output_path
from src.utils.ui import cli_menu
from src.utils.data import get_available_vram
from typing import Optional
from rich.panel import Panel
from src.models.distilbert import DistilBERTClassifer
from src.training.trainer import Trainer
import torch
from pathlib import Path
from src.training.eval import analyze_model_errors, evaluate_model
from src.utils.robustness import keyword_masking, split_length_buckets


class Assignment3Showcase:
    """Class for assignment 3 showcase."""

    ds = AGNews2Trans()

    # TODO: Implement robustness/slice evaluations.
    def __call__(self, choice: Optional[int] = None) -> None:
        """Call the showcase.

        Args:
            choice (Optional[int], optional): The functionality to showcase.
                                              Defaults to None.
        """
        if choice is not None:
            if choice == 1:
                self.finetune_distilbert()

            elif choice == 2:
                self.robustness_evaluation()

            elif choice == 3:
                self.analyze_errors()

            return

        cli_menu(
            "Select a functionality to showcase:",
            {
                "Finetune and Evaluate DistilBERT": self.finetune_distilbert,
                "Robustness Evaluation": self.robustness_evaluation,
                "Analyze Errors": self.analyze_errors,
                "Back to Main Menu": lambda: LOGGER.log_and_print(
                    Panel(
                        "[bold yellow]Returning to Main Menu...[/bold yellow]"
                    )
                ),
            },
        )

    def robustness_evaluation(self) -> None:
        cli_menu(
            "Select a robustness evaluation to run:",
            {
                "Keyword Masking Evaluation": self.keyword_mask_evaluation,
                "Length Bucket Evaluation": self.length_bucket_evaluation,
                "Back to Menu": lambda: None,
            },
        )

    def keyword_mask_evaluation(self) -> None:
        """Evaluate the robustness of the DistilBERT model through a keyword masking procedure."""
        model = self._get_or_finetune_dilstilbert()

        kv = self.ds.tokenizer.get_vocab()

        LOGGER.log_and_print(
            Panel("Evaluating Keyword Masking Procedure...", style="bold blue")
        )

        cli_menu(
            "Evaluate Keyword Masking on which set?",
            {
                "Dev Set": lambda: evaluate_model(
                    model,
                    self.ds,
                    transform=lambda item: keyword_masking(
                        item,
                        kv,
                    ),
                ),
                "Test Set": lambda: evaluate_model(
                    model,
                    self.ds,
                    transform=lambda item: keyword_masking(
                        item,
                        kv,
                    ),
                    use_test=True,
                ),
                "Back to Menu": lambda: None,
            },
        )

    def length_bucket_evaluation(self) -> None:
        """Evaluate the robustness of the DistilBERT model through a length bucket evaluation."""
        model = self._get_or_finetune_dilstilbert()

        def _length_bucket_eval(split: str) -> None:
            torch_ds = self.ds.get_torch_dataset(split, transform_fn=None)
            buckets = split_length_buckets(
                torch_ds.X, torch_ds.y, bucket_size=50
            )

            for bucket_name, bucket_data in buckets.items():
                LOGGER.log_and_print(
                    Panel(
                        f"Evaluating on Length Bucket: {bucket_name} with {len(bucket_data)} samples.",
                        style="bold blue",
                    )
                )
                LOGGER.debug(
                    f"Bucket {bucket_name} has {len(bucket_data)} examples."
                )
                evaluate_model(model, bucket_data, use_test=(split == "test"))

        cli_menu(
            "Evaluate Length Buckets on which set?",
            {
                "Dev Set": lambda: _length_bucket_eval("dev"),
                "Test Set": lambda: _length_bucket_eval("test"),
                "Back to Menu": lambda: None,
            },
        )

    def finetune_distilbert(self) -> None:
        """Finetune and evaluate DistilBERT."""
        model = self._get_or_finetune_dilstilbert()

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

    def _get_or_finetune_dilstilbert(self) -> DistilBERTClassifer:
        """Get a finetune DistilBERT model or train a new one if it does not
        exist.

        Returns:
            DistilBERTClassifer: The finetuned DistilBERT model.
        """
        output_dir = get_output_path(assignment=3)
        model_path = output_dir / "distilbert_model.pt"

        model = DistilBERTClassifer().to(DEVICE)

        if RETRAIN_MODEL or not model_path.exists():
            self._train_model(model_path)
        else:
            LOGGER.info(f"Loading DistilBERT model from {model_path}")

            try:
                model.load_state_dict(
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
                self._train_model(model_path)

        model.eval()

        return model

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

    def analyze_errors(self) -> None:
        """Run error analysis on the DistilBERT model."""
        model = self._get_or_finetune_dilstilbert()

        cli_menu(
            "Analyze errors for which split?",
            {
                "Dev Set": lambda: (
                    analyze_model_errors(
                        model, self.ds, split="dev", min_examples=10
                    ),
                ),
                "Test Set": lambda: (
                    analyze_model_errors(
                        model, self.ds, split="test", min_examples=10
                    ),
                ),
                "Back to Menu": lambda: None,
            },
        )
