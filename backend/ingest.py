"""Ingest knowledge base documents into a vector database (backend module)."""
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


def ingest_knowledge_base():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("請設定 OPENAI_API_KEY 環境變數。")

    loader = DirectoryLoader(
        "./knowledge_base",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"載入了 {len(documents)} 份文件。")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    all_splits = []
    for doc in documents:
        md_header_splits = markdown_splitter.split_text(doc.page_content)
        for split in md_header_splits:
            split.metadata["source"] = doc.metadata["source"]
            all_splits.append(split)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    final_chunks = text_splitter.split_documents(all_splits)

    Chroma.from_documents(
        documents=final_chunks,
        embedding=OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small",
        ),
        persist_directory="./chroma_db",
    )
    print(f"知識庫更新完成！共建立了 {len(final_chunks)} 個知識片段。")


if __name__ == "__main__":
    ingest_knowledge_base()
