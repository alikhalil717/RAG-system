import os
import sys
from weaviate.classes.config import DataType, Property
#  Base paths :

if '__file__' in globals():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
else:

    if os.path.exists("data") and os.path.exists("app"):
        BASE_DIR = os.getcwd()
    else:
        parent = os.path.dirname(os.getcwd())
        if os.path.exists(os.path.join(parent, "data")) and os.path.exists(os.path.join(parent, "app")):
            BASE_DIR = parent
        else:
            BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DATA_PATH = os.path.join(DATA_DIR, "pdfs")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
CHUNKS_CACHE_PATH = os.path.join(CHUNKS_DIR, "chunked_data.joblib")

# Weaviate embedded persistence
WEAVIATE_PERSISTENCE_PATH = os.path.join(BASE_DIR, ".collections")

# Models cache directory
MODELS_CACHE_PATH = os.path.join(BASE_DIR, "models", ".models")

# Create directories if they don't exist
os.makedirs(PDF_DATA_PATH, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(WEAVIATE_PERSISTENCE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODELS_CACHE_PATH), exist_ok=True)


# Weaviate collection & module settings
COLLECTION_NAME = "collection"

# Inference server endpoints (Flask app)
EMBEDDING_INFERENCE_URL = "http://127.0.0.1:5000/"
RERANKER_INFERENCE_URL = "http://127.0.0.1:5000/"

# Weaviate embedded environment variables
WEAVIATE_ENV_VARS = {
    "ENABLE_API_BASED_MODULES": "true",
    "ENABLE_MODULES": "text2vec-transformers,reranker-transformers",
    "TRANSFORMERS_INFERENCE_API": EMBEDDING_INFERENCE_URL,
    "RERANKER_INFERENCE_API": RERANKER_INFERENCE_URL,
}


#  Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Embedding model
#
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DEVICE = "cpu"

#  Reranker model settings
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_DEVICE = "cpu"

#  Retrieval & search defaults
DEFAULT_TOP_K = 5
DEFAULT_HYBRID_ALPHA = 0.5

#  Flask server settings

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False

#collection property
PROPERTIES = [
    Property(name="source", data_type=DataType.TEXT),
    Property(name="chunk", data_type=DataType.TEXT),
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="main_vector", data_type=DataType.NUMBER, vector_index_type="hnsw", vector_index_config={"efConstruction": 128, "M": 64}),
]