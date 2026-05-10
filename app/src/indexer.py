import weaviate
from config import *
from utils import suppress_weaviate_logs, print_object_properties
from typing import List, Dict, Any
from weaviate.classes.config import DataType ,Configure , Property
from weaviate.util import generate_uuid5
from tqdm import tqdm
import joblib
from chunking import chunk_text

def create_index(collection : weaviate.Collection):
    chunks=chunk_text()
    with collection.batch.fixed_size(batch_size=16, concurrent_requests=4) as batch:
        for chunk in tqdm(chunks) :
            uuid=generate_uuid5(chunk)
            batch.add_object(
                properties=chunk ,
                uuid= uuid
            )





def create_collection():
    print("########################################################################Creating Weaviate client")
    client =create_weaviate_client()
    print("########################################################################Client created successfully.\nIndexing document")
    if not client.collections.exists(COLLECTION_NAME):
        collection=client.collections.create(name=COLLECTION_NAME ,
                                             vector_config =VECTORIZER_CONFIG ,
                                             reranker_config =RERANKER_CONFIG ,
                                             properties =PROPERTIES
        )
        print("########################################################################Collection created successfully.\nIndexing documents...")




    else : collection=client.collections.get(COLLECTION_NAME)
    create_index(collection)
    client.close()


def create_weaviate_client():
    with suppress_weaviate_logs():
        client = weaviate.connect_to_embedded(
            persistence_data_path =WEAVIATE_PERSISTENCE_PATH,
            environment_variables =WEAVIATE_ENV_VARS
        )
    return client

if __name__ == "__main__":
    create_collection()

