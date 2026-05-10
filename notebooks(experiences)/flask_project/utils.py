import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.device("cpu")

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
import json
from typing import Union
import contextlib

print("Loading embedding model (CPU mode)...")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", cache_folder=".models", device="cpu")
print("Embedding model loaded on CPU.")



def generate_embedding(texts):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = embedding_model.encode(texts).tolist()
    return embeddings

def rerank(query, documents):
    pairs = [(query, doc) for doc in documents]
    scores = reranker.compute_score(pairs)
    if hasattr(scores, 'tolist'):
        scores = scores.tolist()
    return scores

@contextlib.contextmanager
def suppress_weaviate_logs():
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(original_stdout_fd, 1)
        os.dup2(original_stderr_fd, 2)
        os.close(devnull)
        os.close(original_stdout_fd)
        os.close(original_stderr_fd)

def print_object_properties(obj: Union[dict, list]) -> None:
    t = ''
    if isinstance(obj, dict):
        for x, y in obj.items():
            if x in ['article_content', 'chunk']:
                t += f'{x}: {str(y)[:100]}...(truncated)\n'
            elif x == 'main_vector':
                t += f'{x}: {str(y)[:30]}...(truncated)\n'
            else:
                t += f'{x}: {y}\n'
    else:
        for l in obj:
            for x, y in l.items():
                if x in ['article_content', 'chunk']:
                    t += f'{x}: {str(y)[:100]}...(truncated)\n'
                elif x == 'main_vector':
                    t += f'{x}: {str(y)[:30]}...(truncated)\n'
                else:
                    t += f'{x}: {y}\n'
            t += "\n\n"
    print(t)