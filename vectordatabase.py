# Install library
# uv pip install chromadb sentence-transformers 
import chromadb
from sentence_transformers import SentenceTransformer
# Create embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Create Chroma client
client = chromadb.Client()
# Create collection
collection = client.create_collection(name="my_documents") 
# Persistent storage
client = chromadb.PersistentClient(path="./chroma_db") 
# Sample documents
documents = [
    "Machine learning is amazing",
    "I love Natural Language Processing",
    "Vector databases store embeddings"
]
# Generate embeddings
embeddings = model.encode(documents).tolist()
# Store in vector database
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=["1", "2", "3"]
)
print("Documents stored successfully!")
# User query
query = "What is NLP?"
# Convert query into embedding
query_embedding = model.encode(query).tolist()
# Similarity search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)
# Print results
print("\nSimilar Documents:\n")
for doc in results['documents'][0]:
    print(doc) 

