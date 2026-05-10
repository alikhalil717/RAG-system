import os
from config import *
import torch
from sentence_transformers import SentenceTransformer
from typing import Union ,List
import contextlib
__embedding_model = None

def get_embedding_model():
    global __embedding_model
    if __embedding_model is None:
        __embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=MODELS_CACHE_PATH, device=EMBEDDING_DEVICE)
    return __embedding_model


def generate_embedding(texts : Union[str, List[str]]) -> List[List[float]]:
    if isinstance(texts, str):
        texts = [texts]
    model = get_embedding_model()
    for text in texts:
       if hasattr(text,'detach'):
           text = text.detach().cpu().numpy()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True ,convert_to_tensor=False).tolist()
    return embeddings

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