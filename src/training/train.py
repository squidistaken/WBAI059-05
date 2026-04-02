from sklearn.base import BaseEstimator
from src.data.agnews import AGNews
from src.utils.output import get_output_path
from src.const import LOGGER
import pickle


def train_model(
    model: BaseEstimator,
    ds: AGNews,
    save: bool = True,
    assignment: int = 1,
):
    """Train a model on the AG News dataset.

    Args:
        model: The sklearn model to train
        ds: The AGNews dataset
        save: Whether to save the model
        assignment: Assignment number (for saving to assignment directory)
    """
    LOGGER.info(f"Training {model.__class__.__name__}...")
    model.fit(ds.X_train, ds.y_train)
    LOGGER.info(f"Training complete for {model.__class__.__name__}")

    if save:
        output_dir = get_output_path(assignment=assignment)
        model_path = output_dir / f"{model.__class__.__name__}.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        LOGGER.info(f"Model saved to {model_path}")
    return model


def get_model(
    model: BaseEstimator,
    ds: AGNews,
    assignment: int = 1,
) -> BaseEstimator:
    """Load a trained model from disk.

    Args:
        model: The sklearn model to load
        ds: The AGNews dataset
        assignment: Assignment number (for loading from assignment directory)
    """
    output_dir = get_output_path(assignment=assignment)
    model_path = output_dir / f"{model.__class__.__name__}.pkl"

    if not model_path.exists():
        LOGGER.info(
            f"Model {model_path.name} not found. Training new model..."
        )
        model = train_model(model, ds, assignment=assignment)
    else:
        LOGGER.info(f"Loading {model.__class__.__name__} from {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    LOGGER.info(f"Successfully loaded {model.__class__.__name__}")
    return model
