import argparse
from rich.panel import Panel
from src.const import  DEBUG, LOGGER
from src.utils.ui import cli_menu
import os
import sys

def main():
    """Run main pipeline."""
    LOGGER.info("Starting NLP Pipeline...")

    if DEBUG:
        print("=== MAIN FUNCTION STARTED ===")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script arguments: {sys.argv}")
        print("Loading AG News dataset...")

    # Parser for allow running specific functionalities directly from command
    # line without going through menus.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--assignment", type=int, choices=[1, 2], help="Assignment number"
    )
    parser.add_argument(
        "--functionality",
        type=int,
        choices=[1, 2, 3, 4],
        help="Functionality number",
    )

    args = parser.parse_args()
    
    # define functions for each assignment to call the appropriate showcase based on the choice
    # avoid loading additional modules if not needed when running specific functionalities directly from command line
    def assignment1(choice=None):
        from src.showcase.assignment1 import Assignment1Showcase
        if choice == 1:
            Assignment1Showcase()(choice=1)
        elif choice == 2:
            Assignment1Showcase()(choice=2)
        elif choice == 3:
            Assignment1Showcase()(choice=3)
        else:
            Assignment1Showcase()()
    
    def assignment2(choice=None):
        from src.showcase.assignment2 import Assignment2Showcase
        if choice == 1:
            Assignment2Showcase()(choice=1)
        elif choice == 2:
            Assignment2Showcase()(choice=2)
        elif choice == 3:
            Assignment2Showcase()(choice=3)
        elif choice == 4:
            Assignment2Showcase()(choice=4)
        else:
            Assignment2Showcase()()
            
    def assignment3(choice=None):
        from src.showcase.assignment3 import Assignment3Showcase
        Assignment3Showcase()()

    if args.assignment and args.functionality:
        if args.assignment == 1:
            assignment1(choice=args.functionality)
        elif args.assignment == 2:
            assignment2(choice=args.functionality)
    else:
        panel = Panel("AG News NLP Pipeline", style="bold blue")
        LOGGER.log_and_print(panel)
        while True:
            cli_menu(
                "Select an assignment to showcase different functionalities:",
                {
                    "Assignment 1 - Dataset Showcase & Baseline Models": (
                        assignment1
                    ),
                    "Assignment 2 - CNN & LSTM": (
                        assignment2
                    ),
                    "Assignment 3 - Transformers (Not Implemented)": (
                        assignment3
                    ),
                    "Exit": lambda: exit(0),
                },
            )


if __name__ == "__main__":
    panel = Panel("Starting NLP Pipeline...", style="bold green")
    LOGGER.log_and_print(panel)
    main()
