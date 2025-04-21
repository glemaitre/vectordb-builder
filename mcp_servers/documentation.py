import joblib
from mcp.server.fastmcp import FastMCP
from sentence_transformers import CrossEncoder
from vectordb_builder.ranker import RetrieverReranker

import configuration as conf

mcp = FastMCP("documentation")

api_semantic_retriever = joblib.load(conf.API_SEMANTIC_RETRIEVER_PATH)
api_lexical_retriever = joblib.load(conf.API_LEXICAL_RETRIEVER_PATH)
user_guide_semantic_retriever = joblib.load(conf.USER_GUIDE_SEMANTIC_RETRIEVER_PATH)
user_guide_lexical_retriever = joblib.load(conf.USER_GUIDE_LEXICAL_RETRIEVER_PATH)
gallery_semantic_retriever = joblib.load(conf.GALLERY_SEMANTIC_RETRIEVER_PATH)
gallery_lexical_retriever = joblib.load(conf.GALLERY_LEXICAL_RETRIEVER_PATH)
cross_encoder = CrossEncoder(
    model_name_or_path=conf.CROSS_ENCODER_PATH, device=conf.DEVICE
)
retriever = RetrieverReranker(
    retrievers=[
        api_semantic_retriever.set_params(top_k=conf.API_SEMANTIC_TOP_K),
        api_lexical_retriever.set_params(top_k=conf.API_LEXICAL_TOP_K),
        user_guide_semantic_retriever.set_params(top_k=conf.USER_GUIDE_SEMANTIC_TOP_K),
        user_guide_lexical_retriever.set_params(top_k=conf.USER_GUIDE_LEXICAL_TOP_K),
        gallery_semantic_retriever.set_params(top_k=conf.GALLERY_SEMANTIC_TOP_K),
        gallery_lexical_retriever.set_params(top_k=conf.GALLERY_LEXICAL_TOP_K),
    ],
    cross_encoder=cross_encoder,
    threshold=conf.CROSS_ENCODER_THRESHOLD,
    min_top_k=conf.CROSS_ENCODER_MIN_TOP_K,
    max_top_k=conf.CROSS_ENCODER_MAX_TOP_K,
)


@mcp.tool()
def search_in_index(query: str) -> str:
    """Get most relevant chunks from the scikit-learn documentation for a given query.

    Args:
        query: The query to search for.

    Returns:
        A string containing the different chunks from the scikit-learn documentation.
    """
    results = retriever.query(query)
    if isinstance(results[0], dict):
        return "\n".join([result["text"] for result in results])
    return "\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
