import textwrap

from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore
from llama_index.core.utils import truncate_text


def custom_pprint_source_node(
    source_node: NodeWithScore, source_length: int = 350, wrap_width: int = 70
) -> None:
    """Display source node for jupyter notebook."""
    source_text_fmt = truncate_text(
        source_node.node.get_content().strip(), source_length
    )
    print(f"Node ID: {source_node.node.node_id}")
    if "retrieval_score" in source_node.node.metadata:
        print("Reranking score: ", source_node.score)
        print("Retrieval score: ", source_node.node.metadata["retrieval_score"])
    else:
        source_text_fmt = truncate_text(
            source_node.node.get_content().strip(), source_length
        )
        print(f"Similarity: {source_node.score}")
    print(textwrap.fill(f"Text: {source_text_fmt}\n", width=wrap_width))


def custom_pprint_source_response(
    response: Response,
    source_length: int = 350,
    wrap_width: int = 70,
    show_source: bool = False,
) -> None:
    """
    source: from llama_index.core.response.pprint_utils import pprint_response
    Pretty print response for jupyter notebook.
    """
    if response.response is None:
        response_text = "None"
    else:
        response_text = response.response.strip()

    response_text = f"Final Response: {response_text}"
    print(textwrap.fill(response_text, width=wrap_width))

    if show_source:
        for ind, source_node in enumerate(response.source_nodes):
            print("_" * wrap_width)
            print(f"Source Node {ind + 1}/{len(response.source_nodes)}")
            custom_pprint_source_node(
                source_node, source_length=source_length, wrap_width=wrap_width
            )
