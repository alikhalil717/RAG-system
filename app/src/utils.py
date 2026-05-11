import os
import re
import time
import signal
import subprocess
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
               # local_files_only=True,
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
        print("embedding dimensions:", len(embeddings[0]) if embeddings else 0)
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


def find_pids_on_port(port: int, protocol: str = "tcp") -> List[int]:
    """Return a list of PIDs listening on the given port.

    Tries `lsof` first (returns PIDs with `-t`), falls back to parsing `ss -ltnp` output.
    Returns an empty list when no processes are found or the tools aren't available.
    """
    try:
        out = subprocess.check_output(["lsof", "-i", f"{protocol}:{port}", "-t"], stderr=subprocess.DEVNULL)
        pids = [int(x) for x in out.decode().split() if x.strip()]
        return pids
    except (subprocess.CalledProcessError, FileNotFoundError):
        # fall back to ss parsing
        try:
            out = subprocess.check_output(["ss", "-ltnp"], stderr=subprocess.DEVNULL).decode(errors="ignore")
            pids = []
            for line in out.splitlines():
                if f":{port} " in line or f":{port}\n" in line or f":{port}" in line:
                    m = re.search(r"pid=(\d+)", line)
                    if m:
                        pid = int(m.group(1))
                        if pid not in pids:
                            pids.append(pid)
            return pids
        except Exception:
            return []


def kill_processes_on_port(port: int, force: bool = False, timeout: float = 5.0) -> List[int]:
    """Kill processes listening on `port`.

    Args:
        port: TCP port number to free.
        force: If True, send SIGKILL; otherwise send SIGTERM and wait.
        timeout: Maximum seconds to wait for processes to exit after signaling.

    Returns:
        List of PIDs that were signaled (may include processes that were already gone).
    """
    pids = find_pids_on_port(port)
    if not pids:
        return []
    sig = signal.SIGKILL if force else signal.SIGTERM
    killed = []
    for pid in pids:
        try:
            os.kill(pid, sig)
            killed.append(pid)
        except ProcessLookupError:
            # already exited
            continue
        except PermissionError:
            # cannot kill this process from current privileges
            continue
    # wait for port to be freed (poll)
    deadline = time.time() + timeout
    while time.time() < deadline:
        still = find_pids_on_port(port)
        if not still:
            break
        time.sleep(0.2)
    return killed