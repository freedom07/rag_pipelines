from enum import Enum

from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy, AsyncSparseVectorStrategy, AsyncBM25Strategy
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.retrievers import RouterRetriever
from llama_index.core.tools import RetrieverTool
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.retrievers.bm25 import BM25Retriever

from indexing.components.data_stores import ElasticSearchVectorStoreModule


class MetaDataFilters:
    @staticmethod
    def exact_match_filter(key: str, value: str):
        return MetadataFilters(
            filters=[ExactMatchFilter(key=key, value=value)]
        )


class RetrieverModule:
    """
    https://docs.llamaindex.ai/en/stable/examples/retrievers/bm25_retriever/#bm25-retriever_1
    """
    @classmethod
    def vector_index_retriever(cls, index, similarity_top_k: int = 2):
        retriever = VectorIndexRetriever(index=index, similarity_top_k=similarity_top_k)
        return retriever

    @classmethod
    def bm25_retriever(cls, nodes, similarity_top_k: int = 3):
        return BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)

    @classmethod
    def router_retriever(cls, llm, retriever_tools=None, **kwargs):
        retriever_tools = retriever_tools or [
            RetrieverTool.from_defaults(
                retriever=cls.bm25_retriever(kwargs["nodes"]),
                description="Useful if searching about specific information",
            ),
            RetrieverTool.from_defaults(
                retriever=cls.vector_index_retriever(kwargs["index"]),
                description="Useful in most cases",
            ),
        ]
        retriever = RouterRetriever.from_defaults(
            retriever_tools=retriever_tools,
            llm=llm,
            select_multi=True
        )
        return retriever

    @classmethod
    def hybrid_retriever(cls, index, nodes, similarity_top_k: int = 2):
        vector_retriever = cls.vector_index_retriever(index, similarity_top_k)
        bm25_retriever = cls.bm25_retriever(nodes, similarity_top_k)

        return HybridRetriever(vector_retriever, bm25_retriever)


class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes
