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
from typing import Literal


class AGNews:
    """Class to handle loading and vectorizing the AG News dataset."""

    def __init__(self, path: Path | str = "data/ag_news", verbose: bool = True) -> None:
        """Initialize the class and load/vectorize the dataset."""
        self.path = Path(path)
        self.verbose = verbose
        if self.verbose:
            LOGGER.log_and_print(
                Panel("Initializing AGNews dataset...", style="bold yellow")
            )

        if len(list(self.path.glob("*.csv"))) < 3:
            if DEBUG:
                LOGGER.info("CSV files not found, downloading dataset...")
            download_ag_news()
        self._load_data()
        self._vectorize()

    def _load_data(self) -> None:
        """Load the data from disk into memory."""
        self.train_df = pl.read_csv(self.path / "train.csv")
        self.dev_df = pl.read_csv(self.path / "dev.csv")
        self.test_df = pl.read_csv(self.path / "test.csv")

        # Combine columns into a single text column
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

        # Move target arrays to load_data so both TF-IDF and Embeddings have access
        self.y_train = self.train_df["label"].to_numpy()
        self.y_dev = self.dev_df["label"].to_numpy()
        self.y_test = self.test_df["label"].to_numpy()

        if DEBUG:
            print("Sample of loaded data:")
            print(self.train_df.head())

    def _vectorize(self, max_features=5000) -> None:
        """Vectorize the text data using TF-IDF."""
        self.vectorizer = TfidfVectorizer(
            stop_words="english", max_features=max_features
        )

        self.X_train = self.vectorizer.fit_transform(self.train_df["text"].to_list())
        self.X_dev = self.vectorizer.transform(self.dev_df["text"].to_list())
        self.X_test = self.vectorizer.transform(self.test_df["text"].to_list())

        if DEBUG:
            print("Sample of vectorized data:")
            print("X_train shape:", self.X_train.shape)
            print("y_train shape:", self.y_train.shape)
            print("X_dev shape:", self.X_dev.shape)
            print("y_dev shape:", self.y_dev.shape)
            print("X_test shape:", self.X_test.shape)
            print("y_test shape:", self.y_test.shape)

    def _normalize(self) -> None:
        self.scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse data
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_dev = self.scaler.transform(self.X_dev)
        self.X_test = self.scaler.transform(self.X_test)

    @property
    def label_mapping(self):
        """Return the mapping of label indices to class names."""
        return {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


class AGNewsWord2Vec(AGNews, metaclass=SingletonMeta):
    """PyTorch Dataset wrapper for the AG News dataset."""

    def __init__(self, path: Path | str | None = None, verbose: bool = True) -> None:
        self.path = Path(path) if path is not None else DATA_DIR
        self.verbose = verbose
        if self.verbose:
            LOGGER.log_and_print(
                Panel("Initializing AGNewsWord2Vec dataset...", style="bold yellow")
            )

        if len(list(self.path.glob("*.csv"))) < 3:
            if DEBUG:
                LOGGER.info("CSV files not found, downloading dataset...")
            download_ag_news()
        self._load_data()
        self._embeddings()

    def _get_word2vec(self) -> KeyedVectors:
        model_path = MODEL_DIR / "ag_news_word2vec.kv"
        if model_path.exists():
            if self.verbose:
                LOGGER.log_and_print(
                    Panel("Loading pre-trained Word2Vec model...", style="bold yellow")
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

        # Adding via batching instead of looping.
        if "<PAD>" not in kv:
            try:
                kv.add_vectors(["<PAD>"], [np.zeros(100)])
            except AttributeError:
                kv.add_vector("<PAD>", np.zeros(100))
        return kv

    def _embeddings(self):
        self.train_df = self.train_df.with_columns(
            pl.col("text")
            .map_elements(lambda x: simple_preprocess(x), return_dtype=list[str])
            .alias("tokens")
        )
        self.dev_df = self.dev_df.with_columns(
            pl.col("text")
            .map_elements(lambda x: simple_preprocess(x), return_dtype=list[str])
            .alias("tokens")
        )
        self.test_df = self.test_df.with_columns(
            pl.col("text")
            .map_elements(lambda x: simple_preprocess(x), return_dtype=list[str])
            .alias("tokens")
        )

        self.kv = self._get_word2vec()

        # Apply embeddings to the datasets.
        self.train_df = self.train_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [self.kv[word] for word in tokens if word in self.kv],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )
        self.dev_df = self.dev_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [self.kv[word] for word in tokens if word in self.kv],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )
        self.test_df = self.test_df.with_columns(
            pl.col("tokens")
            .map_elements(
                lambda tokens: [self.kv[word] for word in tokens if word in self.kv],
                return_dtype=pl.List(pl.Array(float, shape=(100,))),
            )
            .alias("embeddings")
        )

    def _pad_sequences(self, sequences, max_length):
        #  Pre-allocate contiguous memory instead of manipulating Python lists
        num_samples = len(sequences)
        vector_size = self.kv.vector_size if hasattr(self.kv, "vector_size") else 100

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

    def get_torch_dataset(self, split=Literal["train", "dev", "test"], max_length=256):
        if split == "train":
            df = self.train_df
        elif split == "dev":
            df = self.dev_df
        elif split == "test":
            df = self.test_df
        else:
            raise ValueError("Invalid split name. Use 'train', 'dev', or 'test'.")

        X = self._pad_sequences(df["embeddings"], max_length)

        # Pull numeric array for memory-efficient one_hot conversion
        y = one_hot(as_tensor(df["label"].to_numpy() - 1), num_classes=4).float()

        return TorchDataset(X, y)

    def nearest_neighbors(self, word, topn=5):
        """Find the nearest neighbors of a word in the embedding space."""
        if word in self.kv:
            return self.kv.most_similar(word, topn=topn)
        else:
            return []


class AGNewsWord2VecDataset(Dataset):
    """PyTorch Dataset wrapper for the AG News dataset with Word2Vec embeddings."""

    def __init__(
        self,
        path: Path | str | None = None,
        split=Literal["train", "dev", "test"],
        verbose: bool = True,
    ) -> None:
        self.ds = AGNewsWord2Vec(path=path, verbose=verbose)
        self.df = {
            "train": self.ds.train_df,
            "dev": self.ds.dev_df,
            "test": self.ds.test_df,
        }[split]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return a single sample from the dataset."""
        seq = self.df["embeddings"][idx]

        # Zero-pad buffer
        padded_embedding = torch.zeros((256, 100), dtype=torch.float32)

        if seq is not None and len(seq) > 0:
            seq_arr = np.array(seq, dtype=np.float32)
            length = min(len(seq_arr), 256)
            padded_embedding[:length, :] = torch.from_numpy(seq_arr[:length])

        label = self.df["label"][idx]
        one_hot_label = torch.zeros(4, dtype=torch.float32)
        one_hot_label[label - 1] = 1.0

        return padded_embedding, one_hot_label
