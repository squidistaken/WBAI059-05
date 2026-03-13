from src.data.download import download_ag_news
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import polars as pl
from pathlib import Path
from src.const import DEBUG, LOGGER, RANDOM_SEED, MODEL_DIR, DATA_DIR
from rich.panel import Panel
from rich.progress import track
import numpy as np
import torch
from gensim.models import Word2Vec, KeyedVectors
from gensim.utils import simple_preprocess
from src.utils.data import TorchDataset
from src.utils.singleton import SingletonMeta
from torch.utils.data import Dataset
from torch import as_tensor, Tensor
from torch.nn.functional import one_hot
from typing import Literal, Union, Any


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
                LOGGER.log_and_print(
                    "CSV files not found, downloading dataset..."
                )

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
            LOGGER.log_and_print(
                f"Sample of loaded data:\n {self.train_df.head()}"
            )

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
            LOGGER.log_and_print("=== VECTORIZED DATA SAMPLE ===")
            LOGGER.log_and_print(f"X_train shape: {self.X_train.shape}")
            LOGGER.log_and_print(f"y_train shape: {self.y_train.shape}")
            LOGGER.log_and_print(f"X_dev shape {self.X_dev.shape}")
            LOGGER.log_and_print(f"y_dev shape: {self.y_dev.shape}")
            LOGGER.log_and_print(f"X_test shape: {self.X_test.shape}")
            LOGGER.log_and_print(f"y_test shape: {self.y_test.shape}")

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


class AGNewsWord2Vec(AGNews, metaclass=SingletonMeta):
    """PyTorch Dataset wrapper class for the AG News dataset."""

    def __init__(
        self, path: Union[Path, str, None] = None, verbose: bool = True
    ) -> None:
        """Initialise the class and load/vectorize the dataset.

        Args:
            path (Union[Path, str, None], optional): The filepath. Defaults to
                                                     None.
            verbose (bool, optional): Whether to be verbose in logging.
                                      Defaults to True.
        """
        self.path = Path(path) if path is not None else DATA_DIR
        self.verbose = verbose

        if self.verbose:
            LOGGER.log_and_print(
                Panel(
                    "Initializing AGNewsWord2Vec dataset...",
                    style="bold yellow",
                )
            )

        if len(list(self.path.glob("*.csv"))) < 3:
            if DEBUG:
                LOGGER.info("CSV files not found, downloading dataset...")

            download_ag_news()

        self._load_data()
        self._embeddings()

    def _get_word2vec(self) -> KeyedVectors:
        """Retrieve or train a Word2Vec model.

        Returns:
            KeyedVectors: The trained KeyedVectors from the model.
        """
        model_path = MODEL_DIR / "ag_news_word2vec.kv"

        if model_path.exists():
            if self.verbose:
                LOGGER.log_and_print(
                    Panel(
                        "Loading pre-trained Word2Vec model...",
                        style="bold yellow",
                    )
                )

            kv = KeyedVectors.load(str(model_path), mmap="r")

        else:
            if self.verbose:
                LOGGER.log_and_print(
                    Panel("Training Word2Vec model...", style="bold yellow")
                )

            w2v = Word2Vec(
                sentences=self.train_df["tokens"].to_list(),
                vector_size=100,
                window=5,
                min_count=3,
                workers=4,
                sg=1,
                negative=10,
                epochs=10,
                seed=RANDOM_SEED,
            )
            kv = w2v.wv

            model_path.parent.mkdir(parents=True, exist_ok=True)
            kv.save(str(model_path))

        # Ensure a padding token exists for downstream sequence processing.
        if "<PAD>" not in kv:
            try:
                kv.add_vectors(["<PAD>"], [np.zeros(100)])

            except AttributeError:
                kv.add_vector("<PAD>", np.zeros(100))

        return kv

    def _embeddings(self) -> None:
        """Tokenise and map words to the Word2Vec model."""
        self.train_df = self.train_df.with_columns(
            pl.col("text")
            .map_elements(
                lambda x: simple_preprocess(x), return_dtype=list[str]
            )
            .alias("tokens")
        )
        self.dev_df = self.dev_df.with_columns(
            pl.col("text")
            .map_elements(
                lambda x: simple_preprocess(x), return_dtype=list[str]
            )
            .alias("tokens")
        )
        self.test_df = self.test_df.with_columns(
            pl.col("text")
            .map_elements(
                lambda x: simple_preprocess(x), return_dtype=list[str]
            )
            .alias("tokens")
        )
        self.kv = self._get_word2vec()

        # Apply embeddings to the datasets.
        self.train_df = self.train_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [
                    self.kv[word] for word in tokens if word in self.kv
                ],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )
        self.dev_df = self.dev_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [
                    self.kv[word] for word in tokens if word in self.kv
                ],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )
        self.test_df = self.test_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [
                    self.kv[word] for word in tokens if word in self.kv
                ],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )

    def _pad_sequences(
        self, sequences: list[Union[np.ndarray, Any]], max_length: int
    ) -> np.ndarray:
        """Convert a list of variable-length embedding lists into a fixed-size
        array.

        Args:
            sequences (list[Union[np.ndarray, Any]]): The list of the shape
                                                      `(num_samples,
                                                      variable_seq_len, 100)`.
            max_length (int): The target length for the sequence dimension.

        Returns:
            np.ndarray: An array of shape `(num_samples, max_length, 100)`.
        """
        num_samples = len(sequences)
        vector_size = (
            self.kv.vector_size if hasattr(self.kv, "vector_size") else 100
        )

        # Pre-allocate contiguous memory for performance.
        X = np.zeros((num_samples, max_length, vector_size), dtype=np.float32)
        seq_wrap = (
            track(sequences, description="Padding sequences")
            if self.verbose
            else sequences
        )

        for i, seq in enumerate(seq_wrap):
            if seq is not None:
                length = min(len(seq), max_length)

                if length > 0:
                    X[i, :length, :] = np.array(seq[:length], dtype=np.float32)

        return X

    def get_torch_dataset(
        self, split: Literal["train", "dev", "test"], max_length: int = 256
    ) -> TorchDataset:
        """Get a TorchDataset for a given split.

        Args:
            split (Literal['train', 'dev', 'test']): The split.
            max_length (int, optional): The maximum sequence length. Defaults
                                        to 256.

        Raises:
            ValueError: If an invalid split name is used.

        Returns:
            TorchDataset: _description_
        """
        if split == "train":
            df = self.train_df
        elif split == "dev":
            df = self.dev_df
        elif split == "test":
            df = self.test_df
        else:
            raise ValueError(
                "Invalid split name. Use 'train', 'dev', or 'test'."
            )

        X = self._pad_sequences(df["embeddings"], max_length)

        # Convert labels into a one-hot tensor.
        y = one_hot(
            as_tensor(df["label"].to_numpy() - 1), num_classes=4
        ).float()

        return TorchDataset(X, y)

    def nearest_neighbors(self, word: str, top_n=5) -> list[tuple[str, float]]:
        """Find the nearest neighbors of a word in the Word2Vec embedding
        space.

        Args:
            word (str): The target word.
            topn (int, optional): The number of neighbours to view. Defaults to
                                  5.

        Returns:
            list[tuple[str, float]]: A list of (word, similarity_score) tuples,
                                     or an empty list if the word is not in the
                                     vocabulary.
        """
        if word in self.kv:
            return self.kv.most_similar(word, topn=top_n)

        return []


class AGNewsWord2VecDataset(Dataset):
    """PyTorch Dataset wrapper class for the AG News dataset with Word2Vec
    embeddings."""

    def __init__(
        self,
        path: Union[Path, str, None] = None,
        split: Literal["train", "dev", "test"] = "train",
        verbose: bool = True,
        max_length: int = 256,
    ) -> None:
        """Initialise the class and load/vectorize the dataset.

        Args:
            path (Union[Path, str, None], optional): The filepath. Defaults to
                                                     None.
            split (Literal['train', 'dev', 'test'], optional): The split.
                                                               Defaults to
                                                               "train".
            verbose (bool, optional): Whether to be verbose in logging.
                                      Defaults to True.
            max_length (int, optional): The fixed length for the padding
                                        embedding sequence. Defaults to 256.
        """
        self.ds = AGNewsWord2Vec(path=path, verbose=verbose)
        self.df = {
            "train": self.ds.train_df,
            "dev": self.ds.dev_df,
            "test": self.ds.test_df,
        }[split]
        self._max_length = max_length

    def __len__(self) -> int:
        """Get the total number of samples in the split.

        Returns:
            int: The total number of samples in the split.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            tuple[Tensor, Tensor]: A tuple `(embedding_tensor, one_hot_label)`.
        """
        seq = self.df["embeddings"][idx]

        # Initialize zero-pad buffer (max_length, vector_dim).
        padded_embedding = torch.zeros(
            (self._max_length, 100), dtype=torch.float32
        )

        if seq is not None and len(seq) > 0:
            seq_arr = np.array(seq, dtype=np.float32)
            length = min(len(seq_arr), self._max_length)
            padded_embedding[:length, :] = torch.from_numpy(seq_arr[:length])

        label = self.df["label"][idx]
        one_hot_label = torch.zeros(4, dtype=torch.float32)
        one_hot_label[label - 1] = 1.0

        return padded_embedding, one_hot_label
