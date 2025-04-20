from pathlib import Path

import pytest
import joblib

from vectordb_builder.ranker import SemanticRetriever
from vectordb_builder.embedding import SentenceTransformer


@pytest.mark.parametrize(
    "input_texts, output",
    [
        (
            [
                {"source": "source 1", "text": "xxx"},
                {"source": "source 2", "text": "yyy"},
            ],
            [{"source": "source 1", "text": "xxx"}],
        ),
        (["xxx", "yyy"], ["xxx"]),
    ],
)
def test_semantic_retriever(input_texts, output, tmp_path):
    """Check that the SemanticRetriever wrapper works as expected"""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    retriever = SemanticRetriever(
        embedding=embedder, persist_directory=tmp_path, top_k=1
    ).fit(input_texts)
    assert retriever.query("xx") == output


def test_semantic_retriever_error(tmp_path):
    """Check that we raise an error when the input is not a string at inference time."""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    input_texts = [{"source": "source 1", "text": "xxx"}]
    retriever = SemanticRetriever(
        embedding=embedder, persist_directory=tmp_path
    ).fit(input_texts)
    with pytest.raises(TypeError):
        retriever.query(["xxxx"])


@pytest.mark.parametrize(
    "input_texts",
    [
        [
            {"source": "source 1", "text": "xxx"},
            {"source": "source 2", "text": "yyy"},
        ],
        ["xxx", "yyy"],
    ],
)
def test_semantic_retriever_max_documents_at_fit(input_texts, tmp_path):
    """Check that return at max the number of documents in the training set."""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    retriever = SemanticRetriever(
        embedding=embedder, persist_directory=tmp_path, top_k=20
    ).fit(input_texts)
    assert len(retriever.query("xx")) == len(input_texts)


def test_semantic_retriever_pathlib_support(tmp_path):
    """Check that the SemanticRetriever accepts Path objects for persist_directory."""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    path_object = Path(tmp_path) / "chroma_path_test"

    input_texts = ["xxx", "yyy"]
    retriever = SemanticRetriever(
        embedding=embedder, persist_directory=path_object, top_k=1
    ).fit(input_texts)

    assert retriever.query("xx") == ["xxx"]
    assert path_object.exists()


def test_semantic_retriever_pickle_unpickle(tmp_path):
    """Check that the SemanticRetriever can be pickled and unpickled using joblib."""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    persist_dir = Path(tmp_path) / "chroma_db_pickle_test"

    input_texts = ["text sample one", "text sample two", "completely different content"]
    retriever = SemanticRetriever(
        embedding=embedder, persist_directory=persist_dir, top_k=1
    ).fit(input_texts)

    query_result_before = retriever.query("sample")

    pickle_path = Path(tmp_path) / "semantic_retriever.joblib"
    joblib.dump(retriever, pickle_path)

    unpickled_retriever = joblib.load(pickle_path)

    query_result_after = unpickled_retriever.query("sample")
    assert query_result_before == query_result_after

    assert unpickled_retriever.persist_directory == persist_dir


def test_semantic_retriever_top_k_parameter(tmp_path):
    """Check that the top_k parameter works as expected."""
    cache_folder_path = (
        Path(__file__).parent.parent.parent / "embedding" / "tests" / "data"
    )
    model_name_or_path = "sentence-transformers/paraphrase-albert-small-v2"

    embedder = SentenceTransformer(
        model_name_or_path=model_name_or_path,
        cache_folder=str(cache_folder_path),
        show_progress_bar=False,
    )

    input_texts = [
        "first document",
        "second document",
        "third document",
        "fourth document",
        "fifth document"
    ]

    # Test with different top_k values
    retriever_top_1 = SemanticRetriever(
        embedding=embedder, persist_directory=tmp_path, top_k=1
    ).fit(input_texts)
    
    retriever_top_3 = SemanticRetriever(
        embedding=embedder, persist_directory=tmp_path / "top_3", top_k=3
    ).fit(input_texts)
    
    retriever_top_5 = SemanticRetriever(
        embedding=embedder, persist_directory=tmp_path / "top_5", top_k=5
    ).fit(input_texts)

    result_top_1 = retriever_top_1.query("document")
    result_top_3 = retriever_top_3.query("document")
    result_top_5 = retriever_top_5.query("document")

    assert len(result_top_1) == 1
    assert len(result_top_3) == 3
    assert len(result_top_5) == 5
