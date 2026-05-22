# Install libraries
# uv pip install pinecone sentence-transformers python-dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
# Pinecone API Key
api_key = os.getenv("PINECONE_API_KEY")
# Initialize Pinecone
pc = Pinecone(api_key=api_key)
# Index name
index_name = "my-documents"
# Create index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,   # embedding dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
# Connect to index
index = pc.Index(index_name)
# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Sample documents
documents = [
    "Machine learning is amazing",
    "I love Natural Language Processing",
    "Vector databases store embeddings"
]
# Generate embeddings
embeddings = model.encode(documents).tolist()
# Prepare vectors
vectors = []
for i, embedding in enumerate(embeddings):
    vectors.append({
        "id": str(i),
        "values": embedding,
        "metadata": {
            "text": documents[i]
        }
    })
# Store vectors in Pinecone
index.upsert(vectors=vectors)
print("Documents stored successfully!")
# User query
query = "What is NLP?"
# Convert query into embedding
query_embedding = model.encode(query).tolist()
# Similarity search
results = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True
)
# Print results
print("\nSimilar Documents:\n")
for match in results["matches"]:
    print(match["metadata"]["text"])   