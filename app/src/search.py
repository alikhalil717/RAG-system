from typing import Any, Dict, List

import weaviate
from weaviate.classes.query import Filter , Rerank
from config import *


def _extract_properties(results):
    return [x.properties for x in results.objects]


def _normalize_rerank_property(rerank_properties):
    if isinstance(rerank_properties, (list, tuple)):
        return rerank_properties[0] if rerank_properties else None
    return rerank_properties


def semantic_search(query:str ,collection:weaviate.Collection ,
                    top_k:int=DEFAULT_TOP_K ,rerank_properties=RERANKER_PROPERTY ,
                    rerank_query:str=None
                    ):
    rerank_property = _normalize_rerank_property(rerank_properties)
    rerank = Rerank(
        query=rerank_query or query,
        prop=rerank_property,
    )

    results = collection.query.hybrid(
        query=query,
        limit=top_k,
        alpha=DEFAULT_HYBRID_ALPHA,
        rerank=rerank,
    )

    return _extract_properties(results)

def bm25_search(query:str ,collection:weaviate.Collection ,top_k:int=DEFAULT_TOP_K):
    results = collection.query.bm25(query=query, limit=top_k)
    return _extract_properties(results)
def hybrid_search(query:str ,collection:weaviate.Collection ,top_k:int=DEFAULT_TOP_K
                  ,alpha:float=DEFAULT_HYBRID_ALPHA ,rerank_properties=RERANKER_PROPERTY ,rerank_query:str=None):
    rerank_property = _normalize_rerank_property(rerank_properties)
    results = None
    if rerank_query is not None:
        rerank = Rerank(
            query=rerank_query,
            prop=rerank_property,
        )

        results = collection.query.hybrid(query=query, limit=top_k, alpha=alpha, rerank=rerank)
    else:
        results = collection.query.hybrid(query=query, limit=top_k, alpha=alpha)
    return _extract_properties(results)
def filtered_search(collection:weaviate.Collection ,top_k:int=DEFAULT_TOP_K ,filters:Filter=None):
    results = collection.query.fetch_objects(limit=top_k, filter=filters)
    return _extract_properties(results)
