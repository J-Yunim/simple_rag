from decouple import config
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# openai_api_key = config("OPENAI_API_KEY")
qdrant_api_key = config("QDRANT_API_KEY")
qdrant_url = config("QDRANT_URL")
collection_name = "Websites"

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=20, length_function=len
)


def create_collection(collection_name: str):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    print(f"Collection {collection_name} created successfully")


def create_vector_store(collection_name: str) -> QdrantVectorStore:
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=OllamaEmbeddings(model="llama3.2"),
    )
    print(f"Vector store for {collection_name} created successfully")
    return vector_store


vector_store = create_vector_store(collection_name)


def upload_website_to_collection(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load_and_split(text_splitter)
    for doc in docs:
        doc.metadata = {"source_url": url}
    vector_store.add_documents(docs)
    print(
        f"Successfully uploaded {len(docs)} documents from {url} to collection {collection_name}"
    )


# create_collection(collection_name)
# upload_website_to_collection(
#     vector_store,
#     "https://ollama.com/library/llama3:8b",
# )
