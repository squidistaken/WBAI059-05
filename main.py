import argparse
from rich.panel import Panel
from src.const import DEBUG, LOGGER
from src.utils.ui import cli_menu
import os
import sys
from typing import Optional


def main():
    """Run main program."""
    LOGGER.info("Starting NLP Pipeline...")

    if DEBUG:
        LOGGER.log_and_print("=== MAIN FUNCTION STARTED ===")
        LOGGER.log_and_print(f"Current working directory: {os.getcwd()}")
        LOGGER.log_and_print(f"Script arguments: {sys.argv}")
        LOGGER.log_and_print("Loading AG News dataset...")

    # Set up parser arguments for instantaneous calling.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--assignment", type=int, choices=[1, 2], help="Assignment number"
    )
    parser.add_argument(
        "--functionality",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        help="Functionality number",
    )

    args = parser.parse_args()

    # Define subfunctions per assignment to call the appropriate showcase
    # based on the choice. This avoids loading additional modules if not needed
    # when running specific functionalities directly from the command line.
    def assignment_1(choice: Optional[int] = None) -> None:
        """Set up the assignment 1 showcase.

        Args:
            choice (Optional[int], optional): The functionality to showcase.
                                              Defaults to None.
        """
        from src.showcase.assignment1 import Assignment1Showcase

        if choice not in [1, 2, 3]:
            Assignment1Showcase()()
        else:
            Assignment1Showcase()(choice=choice)

    def assignment_2(choice: Optional[int] = None) -> None:
        """Set up the assignment 2 showcase.

        Args:
            choice (Optional[int], optional): The functionality to showcase.
                                              Defaults to None.
        """
        from src.showcase.assignment2 import Assignment2Showcase

        if choice not in [1, 2, 3, 4, 5]:
            Assignment2Showcase()()
        else:
            Assignment2Showcase()(choice=choice)

    def assignment_3(choice: Optional[int] = None) -> None:
        """Set up the assignment 3 showcase.

        Args:
            choice (Optional[int], optional): The functionality to showcase.
                                              Defaults to None.
        """
        from src.showcase.assignment3 import Assignment3Showcase

        # TODO: Implement assignment 3.
        Assignment3Showcase()()

    if args.assignment and args.functionality:
        if args.assignment == 1:
            assignment_1(choice=args.functionality)
        elif args.assignment == 2:
            assignment_2(choice=args.functionality)
    else:
        panel = Panel("AG News NLP Pipeline", style="bold blue")

        LOGGER.log_and_print(panel)

        while True:
            cli_menu(
                "Select an assignment to showcase different functionalities:",
                {
                    "Assignment 1 - Dataset Showcase & Baseline Models": (
                        assignment_1
                    ),
                    "Assignment 2 - CNN & LSTM": (assignment_2),
                    "Assignment 3 - Transformers (Not Implemented)": (
                        assignment_3
                    ),
                    "Exit": lambda: exit(0),
                },
            )


if __name__ == "__main__":
    panel = Panel("Starting NLP Pipeline...", style="bold green")
    LOGGER.log_and_print(panel)
    main()
