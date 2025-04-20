import logging
import time
from numbers import Integral
from pathlib import Path

import chromadb
from sklearn.base import BaseEstimator, _fit_context
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class SemanticRetriever(BaseEstimator):
    """Retrieve the k-nearest neighbors using a semantic embedding.

    The index is built using the ChromaDB library.

    Parameters
    ----------
    embedding : transformer
        An embedding following the scikit-learn transformer API.

    persist_directory : str or pathlib.Path, default="./chroma_db"
        Directory where ChromaDB will persist the database.

    top_k : int, default=1
        Number of documents to retrieve.

    Attributes
    ----------
    X_fit_ : list of str or dict
        The input data.

    X_embedded_ : ndarray of shape (n_sentences, n_features)
        The embedded data.

    chroma_client_ : chromadb.PersistentClient
        The ChromaDB client for persistent storage.

    collection_ : chromadb.Collection
        The collection to retrieve the k-nearest neighbors.
    """

    _parameter_constraints = {
        "embedding": [HasMethods(["fit_transform", "transform"])],
        "persist_directory": [str, Path],
        "top_k": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(self, *, embedding, persist_directory="./chroma_db", top_k=1):
        self.embedding = embedding
        self.persist_directory = persist_directory
        self.top_k = top_k

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        """Embed the sentences and create the collection.

        Parameters
        ----------
        X : list of str or dict
            The input data.

        y : None
            This parameter is ignored.

        Returns
        -------
        self
            The fitted estimator.
        """
        self.X_fit_ = X
        start = time.time()
        self.X_embedded_ = self.embedding.fit_transform(X)

        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        self.chroma_client_ = chromadb.PersistentClient(path=str(persist_path))

        collection_name = "semantic_retriever"
        try:
            self.collection_ = self.chroma_client_.get_collection(collection_name)
            self.collection_.delete(where={})
        except:
            self.collection_ = self.chroma_client_.create_collection(collection_name)

        ids = [str(i) for i in range(len(X))]
        if isinstance(X[0], dict):
            documents = [item["text"] for item in X]
        else:
            documents = X

        batch_size = 1_000
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection_.add(
                ids=ids[i:batch_end],
                embeddings=self.X_embedded_[i:batch_end].tolist(),
                documents=documents[i:batch_end],
            )

        logger.info(f"Index created in {time.time() - start:.2f}s")
        return self

    def query(self, query):
        """Retrieve the most relevant documents for the query.

        Parameters
        ----------
        query : str
            The input data.

        Returns
        -------
        list of str or dict
            The list of the most relevant document from the training set.
        """
        check_is_fitted(self, ["X_fit_", "collection_"])
        if not isinstance(query, str):
            raise TypeError(f"query should be a string, got {type(query)}.")

        start = time.time()
        X_embedded = self.embedding.transform([query])

        results = self.collection_.query(
            query_embeddings=X_embedded.tolist(), n_results=self.top_k
        )

        indices = [int(id) for id in results["ids"][0]]

        logger.info(f"Semantic search done in {time.time() - start:.2f}s")

        if isinstance(self.X_fit_[0], dict):
            return [
                {
                    "source": self.X_fit_[neighbor]["source"],
                    "text": self.X_fit_[neighbor]["text"],
                }
                for neighbor in indices
            ]
        else:  # isinstance(self.X_fit_[0], str)
            return [self.X_fit_[neighbor] for neighbor in indices]

    def __getstate__(self):
        """Custom state for pickling."""
        state = self.__dict__.copy()
        if "chroma_client_" in state:
            del state["chroma_client_"]
        if "collection_" in state:
            del state["collection_"]
        return state

    def __setstate__(self, state):
        """Restore state when unpickling."""
        self.__dict__.update(state)
        if hasattr(self, "X_fit_"):
            persist_path = Path(self.persist_directory)
            self.chroma_client_ = chromadb.PersistentClient(path=str(persist_path))
            try:
                self.collection_ = self.chroma_client_.get_collection(
                    "semantic_retriever"
                )
            except ValueError as e:
                logger.error(f"Failed to reconnect to ChromaDB collection: {e}")
                raise RuntimeError(
                    "ChromaDB collection not found. The model may need to be refit."
                ) from e
