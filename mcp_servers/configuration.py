DEVICE = "cpu"

# Path to store the retriever once trained
API_SEMANTIC_RETRIEVER_PATH = "../models/api_semantic_retrieval.joblib"
API_CHROMA_PATH = "../models/api_semantic_retrieval.chroma"
API_SEMANTIC_TOP_K = 5
API_LEXICAL_RETRIEVER_PATH = "../models/api_lexical_retrieval.joblib"
API_LEXICAL_TOP_K = 5
USER_GUIDE_SEMANTIC_RETRIEVER_PATH = "../models/user_guide_semantic_retrieval.joblib"
USER_GUIDE_CHROMA_PATH = "../models/user_guide_semantic_retrieval.chroma"
USER_GUIDE_SEMANTIC_TOP_K = 5
USER_GUIDE_LEXICAL_RETRIEVER_PATH = "../models/user_guide_lexical_retrieval.joblib"
USER_GUIDE_LEXICAL_TOP_K = 5
GALLERY_SEMANTIC_RETRIEVER_PATH = "../models/gallery_semantic_retrieval.joblib"
GALLERY_CHROMA_PATH = "../models/gallery_semantic_retrieval.chroma"
GALLERY_SEMANTIC_TOP_K = 5
GALLERY_LEXICAL_RETRIEVER_PATH = "../models/gallery_lexical_retrieval.joblib"
GALLERY_LEXICAL_TOP_K = 5
CROSS_ENCODER_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_THRESHOLD = 2.0
CROSS_ENCODER_MIN_TOP_K = 3
CROSS_ENCODER_MAX_TOP_K = 20

# Parameter for the reranker
CROSS_ENCODER_PATH = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_THRESHOLD = 2.0
CROSS_ENCODER_MIN_TOP_K = 3
CROSS_ENCODER_MAX_TOP_K = 20
