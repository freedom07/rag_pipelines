from llama_index.core import PromptTemplate
from llama_index.core.indices.query.query_transform import HyDEQueryTransform, StepDecomposeQueryTransform

from inferences.components.models import get_bedrock_li_text_model, get_openai_model


class QueryCustomRewritingModule:
    def __init__(self, llm=None):
        self.llm = llm or get_bedrock_li_text_model()

    def generate_queries(self, query: str, num_queries: int = 2):
        query_gen_str = """\
        You are a helpful assistant that generates multiple search queries based on a \
        single input query. Generate {num_queries} search queries, one on each line, \
        related to the following input query:
        Query: {query}
        Queries:
        """
        query_gen_prompt = PromptTemplate(query_gen_str)

        response = self.llm.predict(
            query_gen_prompt, num_queries=num_queries, query=query
        )
        queries = response.split("\n")

        return queries


class QueryTransformModule:
    @classmethod
    def hyde_query_transform(cls, llm):
        """
        query_engine = TransformQueryEngine(
            query_engine=index.as_query_engine(),
            query_transform=QueryTransformModule.hyde_query_transform(llm=llm)
        )
        """
        return HyDEQueryTransform(llm=llm, include_original=True)

    @classmethod
    def step_decompose_query_transform(cls, llm):
        """
        query_engine = MultiStepQueryEngine(
            query_engine=index.as_query_engine(),
            query_transform=QueryTransformModule.step_decompose_query_transform(llm=llm),
            index_summary="Used to answer questions about the author"
        )
        """
        return StepDecomposeQueryTransform(llm=llm, verbose=True)


if __name__ == "__main__":
    from llama_index.core.query_engine import TransformQueryEngine, MultiStepQueryEngine
    from indexing.components.data_stores import ElasticSearchVectorStoreModule

    INDEX_NAME = "test"
    index = ElasticSearchVectorStoreModule(index_name=INDEX_NAME).get_index()
    model = get_openai_model()

    query_str = "what did paul graham do after going to RISD"

    query_transform = QueryTransformModule.hyde_query_transform()
    query_engine = TransformQueryEngine(
        query_engine=index.as_query_engine(),
        query_transform=query_transform
    )

    step_decompose_transform = QueryTransformModule.step_decompose_query_transform(llm=model)
    query_engine = MultiStepQueryEngine(
        query_engine=index.as_query_engine(),
        query_transform=step_decompose_transform,
        index_summary="Used to answer questions about the author"
    )

    response = query_engine.query(
        "Who was in the first batch of the accelerator program the author started?",
    )
