from enum import Enum
from typing import List

from elasticsearch.helpers.vectorstore import AsyncDenseVectorStrategy, AsyncSparseVectorStrategy, AsyncBM25Strategy
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.schema import Node, BaseNode
from llama_index.embeddings.bedrock import BedrockEmbedding
import os

import boto3
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, Document, KnowledgeGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.opensearch import OpensearchVectorClient, OpensearchVectorStore
from opensearchpy import AWSV4SignerAsyncAuth, AsyncHttpConnection, NotFoundError

load_dotenv()


class BedrockEmbeddingEnum(str, Enum):
    TITAN_EMBED_TEXT_V1 = "amazon.titan-embed-text-v1"
    COHERE_EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    COHERE_EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"


def get_bedrock_li_embedding_model(emb_model=BedrockEmbeddingEnum.COHERE_EMBED_MULTILINGUAL_V3.value):
    embed_model = BedrockEmbedding(
        model=emb_model,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
        region_name=os.getenv("REGION"),
    )
    return embed_model


def get_openai_li_embedding_model():
    embed_model = OpenAIEmbedding()
    return embed_model


class OpenSearchVectorStoreModule:
    def __init__(self, index_name: str, embedding: BaseEmbedding = get_bedrock_li_embedding_model(), text_field: str = "content", embedding_field: str = "embedding"):
        self.endpoint = os.getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
        self.index_name = index_name
        self.text_field = text_field
        self.embedding_field = embedding_field
        self.embedding = embedding

    def get_vector_store(self, dim: int = 1536):
        credentials = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
            region_name=os.getenv("REGION"),
        ).get_credentials()

        client = OpensearchVectorClient(
            endpoint=self.endpoint,
            index=self.index_name,
            dim=dim,
            embedding_field=self.embedding_field,
            text_field=self.text_field,
            use_ssl=True,
            verify_certs=True,
            http_auth=AWSV4SignerAsyncAuth(credentials, os.getenv("REGION"), service="aoss"),
            connection_class=AsyncHttpConnection
        )
        vector_store = OpensearchVectorStore(client)
        return vector_store

    def get_index(self):
        vector_store = self.get_vector_store()
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    def add_nodes(self, nodes: List[Node]):
        vector_store = self.get_vector_store()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return VectorStoreIndex(
            nodes=nodes,
            embed_model=self.embedding,
            storage_context=storage_context
        )


class RetrieverStrategyEnum(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    BM25 = "bm25"
    HYBRID = "hybrid"


strategy_instances = {
    RetrieverStrategyEnum.DENSE: AsyncDenseVectorStrategy(),  # default, K-nearest-neighbors, cosine similarity
    RetrieverStrategyEnum.SPARSE: AsyncSparseVectorStrategy(),
    RetrieverStrategyEnum.BM25: AsyncBM25Strategy(),
    RetrieverStrategyEnum.HYBRID: AsyncDenseVectorStrategy(hybrid=True)
}


class ElasticSearchVectorStoreModule:
    def __init__(self, index_name: str, embedding: BaseEmbedding = None, retrieval_strategy: RetrieverStrategyEnum = RetrieverStrategyEnum.DENSE):
        self.vector_store = ElasticsearchStore(
            index_name=index_name,
            es_cloud_id=os.getenv("es_cloud_id"),  # found within the deployment page
            es_user=os.getenv("es_user"),
            es_password=os.getenv("es_password"),
            retrieval_strategy=strategy_instances[retrieval_strategy]
        )
        self.index_name = index_name
        self.embedding = embedding or get_bedrock_li_embedding_model()

    def get_index(self):
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embedding
        )

    def add_nodes(self, nodes: List[BaseNode]):
        """
        get or create index
        """
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        index = self.get_index()
        return index.insert_nodes(nodes=nodes, storage_context=storage_context)

    def add_documents(self, documents: List[Document], transformations=None):
        """
        use build_index_from_nodes internally
        transformation defines after executing build_index_from_nodes
        because transformation is executed in transforming document to node
        """
        transformations = transformations or []
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=storage_context,
            transformations=transformations,
            show_progress=True
        )
        return index


class BaseGraphStoreModule:
    def __init__(self):
        self.graph_store = SimpleGraphStore()

    def add_document(self, documents: List[Document]):
        storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        index = KnowledgeGraphIndex.from_documents(
            documents=documents,
            max_triplets_per_chunk=2,
            storage_context=storage_context,
        )
        return index


class NeptuneGraphStoresModule:
    pass
