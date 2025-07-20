from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from config.config import pinecone_api_key, PINECONE_INDEX

pinecone = Pinecone(api_key=pinecone_api_key)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


if not pinecone.has_index(PINECONE_INDEX):
    pinecone.create_index(
        name=PINECONE_INDEX,
        dimension=3072,         # Dimension for OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Connect to index / Load the index
pc_index = pinecone.Index(PINECONE_INDEX)
vectorStore = PineconeVectorStore(index=pc_index, embedding=embedding_model)