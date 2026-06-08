import argparse
import weaviate
from config import *
from utils import suppress_weaviate_logs, print_object_properties, generate_embedding, _fallback_embedding
from typing import List, Dict, Any
from weaviate.util import generate_uuid5
from tqdm import tqdm
from chunking import chunk_text

def create_index(collection : weaviate.Collection):
    chunks=chunk_text()
    with collection.batch.fixed_size(batch_size=16, concurrent_requests=4) as batch:
        for chunk in tqdm(chunks) :
            uuid = generate_uuid5(chunk)
            try:
                vec = generate_embedding(chunk.get("chunk", ""))
                if isinstance(vec, list) and len(vec) and isinstance(vec[0], list):
                    vec = vec[0]
            except Exception:
                vec = _fallback_embedding(chunk.get("chunk", ""))

            # ensure non-zero vector
            if not any(abs(float(x)) > 1e-12 for x in vec):
                vec = _fallback_embedding(chunk.get("chunk", ""))

            batch.add_object(
                properties=chunk,
                uuid=uuid,
                vector=vec,
            )





def create_collection():
    client =create_weaviate_client()
    if client.collections.exists(COLLECTION_NAME):
        client.collections.delete(COLLECTION_NAME)
    if not client.collections.exists(COLLECTION_NAME):
        collection=client.collections.create(name=COLLECTION_NAME ,
                                             vector_config =VECTORIZER_CONFIG ,
                                             reranker_config =RERANKER_CONFIG ,
                                             properties =PROPERTIES
        )




    else : collection=client.collections.get(COLLECTION_NAME)
    create_index(collection)
    client.close()
    return collection


def create_weaviate_client():
    with suppress_weaviate_logs():
        try:
            client = weaviate.connect_to_embedded(
                persistence_data_path=WEAVIATE_PERSISTENCE_PATH,
                environment_variables=WEAVIATE_ENV_VARS
            )
        except Exception:
            client = weaviate.connect_to_local(port=8079, grpc_port=50050)
    return client

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexing helper for Weaviate collection")
    parser.add_argument('--rebuild', action='store_true', help='Delete and recreate the collection')
    parser.add_argument('--delete', action='store_true', help='Delete the collection and exit')
    args = parser.parse_args()

    if args.delete:
        client = create_weaviate_client()
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
            print(f"Deleted collection: {COLLECTION_NAME}")
        else:
            print(f"Collection {COLLECTION_NAME} does not exist")
        client.close()
    elif args.rebuild:
        print("Rebuilding collection...")
        client = create_weaviate_client()
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)
        client.close()
        create_collection()
        print("Rebuild complete.")
    else:
        create_collection()

