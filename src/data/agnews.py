from src.data.download import download_ag_news
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from src.const import DEBUG, LOGGER
from pathlib import Path
from typing import Union
from rich.panel import Panel
import polars as pl


class AGNews:
    """Class to handle loading and vectorizing the AG News dataset."""

    def __init__(
        self, path: Union[Path, str] = "data/ag_news", verbose: bool = True
    ) -> None:
        """Initialise the class and load/vectorize the dataset.

        Args:
            path (Union[Path, str], optional): The filepath. Defaults to
                                               "data/ag_news".
            verbose (bool, optional): Whether to be verbose in logging.
                                      Defaults to True.
        """
        self.path = Path(path)
        self.verbose = verbose

        if self.verbose:
            LOGGER.log_and_print(
                Panel("Initializing AGNews dataset...", style="bold yellow")
            )

        if len(list(self.path.glob("*.csv"))) < 3:
            if DEBUG:
                LOGGER.debug("CSV files not found, downloading dataset...")

            download_ag_news()

        self._load_data()
        self._vectorize()

    def _load_data(self) -> None:
        """Load the data from disk into memory."""
        self.train_df = pl.read_csv(self.path / "train.csv")
        self.dev_df = pl.read_csv(self.path / "dev.csv")
        self.test_df = pl.read_csv(self.path / "test.csv")

        # Combine columns into a single text column.
        self.train_df = self.train_df.with_columns(
            pl.concat_str(
                [pl.col("title"), pl.col("description")], separator=" "
            ).alias("text")
        )
        self.dev_df = self.dev_df.with_columns(
            pl.concat_str(
                [pl.col("title"), pl.col("description")], separator=" "
            ).alias("text")
        )
        self.test_df = self.test_df.with_columns(
            pl.concat_str(
                [pl.col("title"), pl.col("description")], separator=" "
            ).alias("text")
        )

        # Move target arrays to load_data so both TF-IDF and Embeddings have
        # access.
        self.y_train = self.train_df["label"].to_numpy()
        self.y_dev = self.dev_df["label"].to_numpy()
        self.y_test = self.test_df["label"].to_numpy()

        if DEBUG:
            LOGGER.debug(f"Sample of loaded data:\n {self.train_df.head()}")

    def _vectorize(self, max_features=5000) -> None:
        """Vectorize the text data using TF-IDF.

        Args:
            max_features (int, optional): The maximum number of features.
                                          Defaults to 5000.
        """
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features
        )
        self.X_train = self.vectorizer.fit_transform(
            self.train_df["text"].to_list()
        )
        self.X_dev = self.vectorizer.transform(self.dev_df["text"].to_list())
        self.X_test = self.vectorizer.transform(self.test_df["text"].to_list())

        if DEBUG:
            LOGGER.debug("=== VECTORIZED DATA SAMPLE ===")
            LOGGER.debug(f"X_train shape: {self.X_train.shape}")
            LOGGER.debug(f"y_train shape: {self.y_train.shape}")
            LOGGER.debug(f"X_dev shape {self.X_dev.shape}")
            LOGGER.debug(f"y_dev shape: {self.y_dev.shape}")
            LOGGER.debug(f"X_test shape: {self.X_test.shape}")
            LOGGER.debug(f"y_test shape: {self.y_test.shape}")

    def _normalize(self) -> None:
        """Normalise the data."""
        # When with_mean = False, the data is sparse.
        self.scaler = StandardScaler(with_mean=False)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_dev = self.scaler.transform(self.X_dev)
        self.X_test = self.scaler.transform(self.X_test)

    @property
    def label_mapping(self):
        """Return a mapping of label indices to class names."""
        return {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
