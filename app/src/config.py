import os
import sys
from tokenize import Single
from weaviate.classes.config import Configure, DataType, Property
from dotenv import load_dotenv

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

PROJECT_ROOT = os.path.dirname(BASE_DIR)
try:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except Exception:
    pass

DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DATA_PATH = os.path.join(DATA_DIR, "pdfs")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
CHUNKS_CACHE_PATH = os.path.join(CHUNKS_DIR, "chunked_data.joblib")

WEAVIATE_PERSISTENCE_PATH = os.path.join(BASE_DIR, ".collections")

MODELS_CACHE_PATH = os.path.join(BASE_DIR, "models", ".models")


def _get_local_model_path(model_dir_name: str, default_model_name: str) -> str:
    model_root = os.path.join(MODELS_CACHE_PATH, model_dir_name)
    snapshots_dir = os.path.join(model_root, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshots = sorted(
            entry for entry in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, entry))
        )
        if snapshots:
            return os.path.join(snapshots_dir, snapshots[-1])
    return default_model_name

os.makedirs(PDF_DATA_PATH, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(WEAVIATE_PERSISTENCE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODELS_CACHE_PATH), exist_ok=True)


COLLECTION_NAME = os.getenv("COLLECTION_NAME", "collection")

EMBEDDING_INFERENCE_URL = os.getenv("EMBEDDING_INFERENCE_URL", "http://127.0.0.1:5000/")
RERANKER_INFERENCE_URL = os.getenv("RERANKER_INFERENCE_URL", "http://127.0.0.1:5000/")

WEAVIATE_ENV_VARS = {
    "ENABLE_API_BASED_MODULES": "true",
    "ENABLE_MODULES": "text2vec-transformers ,reranker-transformers",
    "TRANSFORMERS_INFERENCE_API": EMBEDDING_INFERENCE_URL,
    "RERANKER_INFERENCE_API": RERANKER_INFERENCE_URL,
}


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

SOURCE_PROPERTIES=['chunk_size' ,'chunk_overlap' ,'separators']
EMBEDDING_MODEL_NAME = _get_local_model_path(
    "models--BAAI--bge-base-en-v1.5",
    "BAAI/bge-base-en-v1.5",
)
EMBEDDING_DEVICE = "cpu"
VECTORIZER_CONFIG = Configure.Vectors.text2vec_transformers(
    name="default",
    source_properties=["chunk"],
    vectorize_collection_name=False,
    inference_url=EMBEDDING_INFERENCE_URL,
)


RERANKER_MODEL_NAME = _get_local_model_path(
    "models--cross-encoder--ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
RERANKER_DEVICE = "cpu"
RERANKER_CONFIG = Configure.Reranker.transformers()


DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "20"))
DEFAULT_HYBRID_ALPHA = float(os.getenv("DEFAULT_HYBRID_ALPHA", "0.5"))
RERANKER_PROPERTY = ["chunk"]


FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"

WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))

PROPERTIES = [
    Property(name="source", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="book", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="chapter", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="page", data_type=DataType.INT, skip_vectorization=True),
    Property(name="chunk", data_type=DataType.TEXT),
    Property(name="chunk_index", data_type=DataType.INT),
    Property(name="has_code", data_type=DataType.BOOL, skip_vectorization=True),
    Property(name="code_language", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="content_type", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="file_path", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="chapter_number", data_type=DataType.NUMBER, skip_vectorization=True),
    Property(name="code_identifiers", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="http_route", data_type=DataType.TEXT, skip_vectorization=True),
    Property(name="is_full_file_snapshot", data_type=DataType.BOOL, skip_vectorization=True),
    Property(name="file_version_index", data_type=DataType.INT, skip_vectorization=True),
    Property(name="part_index", data_type=DataType.INT, skip_vectorization=True),
    Property(name="part_total", data_type=DataType.INT, skip_vectorization=True),
    Property(name="is_latest_file_version", data_type=DataType.BOOL, skip_vectorization=True),
]

LLM_NAME = os.getenv("LLM_NAME", "deepseek-coder")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
if not LLM_API_KEY:
    print("WARNING: LLM_API_KEY is not set. Add it to your .env file.")
LLM_URL = os.getenv("LLM_URL", "https://api.deepseek.com")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "3000"))







