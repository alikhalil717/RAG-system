import os
from config import *
import torch
from sentence_transformers import SentenceTransformer
from typing import Union ,List
import contextlib
import hashlib

__embedding_model = None
_FALLBACK_EMBEDDING_DIM = 384

def get_embedding_model():
    global __embedding_model
    if __embedding_model is None:
        try:
            __embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                cache_folder=MODELS_CACHE_PATH,
                device=EMBEDDING_DEVICE,
                local_files_only=True,
            )
        except Exception:
            __embedding_model = False
    return __embedding_model


def _fallback_embedding(text: str) -> List[float]:
    vector = [0.0] * _FALLBACK_EMBEDDING_DIM
    tokens = text.lower().split()
    if not tokens:
        tokens = ["<empty>"]

    for index, token in enumerate(tokens):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        position = int.from_bytes(digest[:4], "little") % _FALLBACK_EMBEDDING_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[position] += sign * (1.0 + (index % 5) * 0.1)

    magnitude = sum(value * value for value in vector) ** 0.5
    if magnitude:
        vector = [value / magnitude for value in vector]
    return vector


def generate_embedding(texts : Union[str, List[str]]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]
    model = get_embedding_model()
    if model:
        for text in texts:
           if hasattr(text,'detach'):
               text = text.detach().cpu().numpy()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True ,convert_to_tensor=False).tolist()
        return embeddings
    print("Warning: Using fallback embedding function. This may lead to suboptimal search results.")

    return [_fallback_embedding(text) for text in texts]

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