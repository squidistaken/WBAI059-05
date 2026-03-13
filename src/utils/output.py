from pathlib import Path
from src.const import MODEL_DIR


def get_output_path(assignment: int) -> Path:
    """Get the output path for a specific assignment.

    Args:
        assignment: The assignment number.

    Returns:
        Path to the output directory.
    """

    output_dir = MODEL_DIR / f"assignment_{assignment}"
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir
