import os
from typing import List

from llama_index.core.postprocessor import SentenceTransformerRerank, SimilarityPostprocessor, KeywordNodePostprocessor, \
    LongContextReorder, LLMRerank
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.postprocessor.colbert_rerank import ColbertRerank


def similarity_postprocessor(cls, cutoff=0.7):
    return SimilarityPostprocessor(cutoff=cutoff)


def keyword_node_postprocessor(required_keywords: List[str], exclude_keywords: List[str]):
    return KeywordNodePostprocessor(
        required_keywords=required_keywords, exclude_keywords=exclude_keywords
    )


def long_context_reorder():
    return LongContextReorder()


def metadata_replacement():
    """
    https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/#querying
    """
    pass


class RerankModule:
    @classmethod
    def cohere_rerank(cls, top_n=3, model="rerank-multilingual-v3.0"):
        post_processor = CohereRerank(
            top_n=top_n,
            model=model,
            api_key=os.environ["COHERE_API_KEY"]
        )
        return post_processor

    @classmethod
    def colbert_rerank(cls, top_n=3, model="colbert-ir/colbertv2.0", tokenizer="colbert-ir/colbertv2.0"):
        post_processor = ColbertRerank(
            top_n=top_n,
            model=model,
            tokenizer=tokenizer,
            keep_retrieval_score=True,
        )
        return post_processor

    @classmethod
    def sentence_transformer_rerank(cls, top_n=3, model="cross-encoder/ms-marco-MiniLM-L-2-v2"):
        return SentenceTransformerRerank(
            model=model, top_n=top_n
        )

    @classmethod
    def llm_rerank(cls, top_n=2):
        return LLMRerank(top_n=top_n)
