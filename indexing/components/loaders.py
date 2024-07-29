from llama_index.core.node_parser import SentenceSplitter, SentenceWindowNodeParser
from typing import List

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PDFReader


class DocumentReader:
    def __init__(self, dir_path: str = None, file_path: List[str] = None):
        self.dir_path = dir_path
        self.file_paths = file_path

    def simple_directory_reader(self) -> List[Document]:
        documents = []

        if self.dir_path:
            documents = SimpleDirectoryReader(input_dir=self.dir_path).load_data()
        elif self.file_paths:
            documents = SimpleDirectoryReader(input_files=self.file_paths).load_data()
        return documents

    def pdf_reader(self) -> List[Document]:
        documents = []

        pdf_parser = PDFReader()
        pdf_file_extractor = {".pdf": pdf_parser}

        if self.dir_path:
            documents = SimpleDirectoryReader(input_dir=self.dir_path, file_extractor=pdf_file_extractor).load_data()
        elif self.file_paths:
            documents = SimpleDirectoryReader(input_files=self.file_paths,
                                              file_extractor=pdf_file_extractor).load_data()

        return documents


class ChunkParser:
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def base_parse(self, chunk_size=256, chunk_overlap=16):
        text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = text_splitter.get_nodes_from_documents(self.documents)
        return nodes

    def sentence_window_parse(self, window_size=3):
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        nodes = node_parser.get_nodes_from_documents(self.documents)
        return nodes

    def semantic_parse(self):
        pass

    def hierarchical_parse(self):
        pass


def get_paul_graham_documents(path: str = "data/paul_graham"):
    return DocumentReader(dir_path=path).simple_directory_reader()
