import weaviate
from .config import *
from .utils import suppress_weaviate_logs, print_object_properties
from typing import List, Dict, Any
from wewaviate.classes.cofig import DataType ,Configure , Property
from weaviate.utils import generate_uuid5
from tqdm import tqdm
import joblib
from .chunking import chunk_text

def create_index():


 return 1




def create_collection():

 return create_index()







def create_weaviate_client():
    with suppress_weaviate_logs():
        client = weaviate.connect_to_embedded(
            persistence_path=WEAVIATE_PERSISTENCE_PATH,
            env_vars=WEAVIATE_ENV_VARS
        )
    return client

if __name__ == "__main__":
    create_collection()

