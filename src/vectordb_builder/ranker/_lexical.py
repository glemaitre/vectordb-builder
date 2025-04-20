import logging
import time
from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, _fit_context, clone
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class BM25Retriever(TransformerMixin, BaseEstimator):
    """Retrieve the k-nearest neighbors using a lexical search based on BM25.

    Parameters
    ----------
    count_vectorizer : transformer, default=None
        A count vectorizer to compute the count of terms in documents. If None, a
        :class:`sklearn.feature_extraction.text.CountVectorizer` is used.

    b : float, default=0.75
        The parameter of the BM25 formula.

    k1 : float, default=1.6
        The parameter of the BM25 formula.

    Attributes
    ----------
    X_fit_ : list of str or dict
        The input data.

    X_counts_ : sparse matrix of shape (n_documents, n_features)
        The count of terms in documents.

    count_vectorizer_ : transformer
        The count vectorizer used to compute the count of terms in documents.

    n_terms_by_document_ : ndarray of shape (n_sentences,)
        The number of terms by document.

    averaged_document_length_ : float
        The average number of terms by document.

    idf_ : ndarray of shape (n_features,)
        The inverse document frequency.
    """

    _parameter_constraints = {
        "count_vectorizer": [HasMethods(["fit_transform", "transform"]), None],
        "b": [Interval(Real, 0.0, 1.0, closed="left")],
        "k1": [Interval(Real, 0.0, 10.0, closed="left")],
    }

    def __init__(self, *, count_vectorizer=None, b=0.75, k1=1.6):
        self.count_vectorizer = count_vectorizer
        self.b = b
        self.k1 = k1

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        """Compute the vocabulary and the idf.

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

        if isinstance(X[0], dict):
            X = [x["text"] for x in X]

        start = time.time()
        if self.count_vectorizer is None:
            self.count_vectorizer_ = CountVectorizer().fit(X)
        else:
            self.count_vectorizer_ = clone(self.count_vectorizer).fit(X)

        self.X_counts_ = self.count_vectorizer_.transform(X)
        self.n_terms_by_document_ = self.X_counts_.sum(axis=1).A1
        self.averaged_document_length_ = self.n_terms_by_document_.mean()

        # compute idf
        n_documents = len(self.X_fit_)
        n_documents_by_term = self.X_counts_.sum(axis=0).A1
        numerator = n_documents - n_documents_by_term + 0.5
        denominator = n_documents_by_term + 0.5
        self.idf_ = np.log(numerator / denominator + 1)
        self.idf_[self.idf_ < 0] = 0.25 * np.mean(self.idf_)

        logger.info(f"BM25Retriever fitted in {time.time() - start:.2f}s")
        return self

    def transform(self, X):
        """Retrieve the most relevant documents for the query.

        Parameters
        ----------
        X : str
            The input data.

        Returns
        -------
        list of str or dict
            The list of the most relevant document from the training set.
        """
        check_is_fitted(self, "X_fit_")
        if not isinstance(X, str):
            raise TypeError(f"X should be a string, got {type(X)}.")
        start = time.time()
        query_terms_indices = self.count_vectorizer_.transform([X]).indices
        counts_query_in_X_fit = self.X_counts_[:, query_terms_indices].toarray()
        idf = self.idf_[query_terms_indices]
        numerator = counts_query_in_X_fit * (self.k1 + 1)
        denominator = counts_query_in_X_fit + self.k1 * (
            1
            - self.b
            + self.b
            * (
                self.n_terms_by_document_.reshape(-1, 1)
                / self.averaged_document_length_
            )
        )
        scores = (idf * numerator / denominator).sum(axis=1)
        logger.info(f"BM25Retriever scored in {time.time() - start:.2f}s")
        return scores

    def query(self, query, top_k=1):
        """Retrieve the most relevant documents for the query.

        Parameters
        ----------
        query : str
            The query.

        top_k : int, default=1
            Number of documents to retrieve.

        Returns
        -------
        list of str or dict
            The list of the most relevant document from the training set.
        """
        start = time.time()
        scores = self.transform(query)
        indices = scores.argsort()[::-1][: top_k]
        logger.info(f"BM25Retriever scored and retrieved in {time.time() - start:.2f}s")
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
