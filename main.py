from sentence_transformers import SentenceTransformer

# Load pretrained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sentences
sentences = [
    "I Love NLP",
    "Machine Learning is amazing",
    "Transformers creating embeddings"
]

# Generate embeddings
embeddings = model.encode(sentences)

# Print embeddings
for i, embedding in enumerate(embeddings):
    print(f"\nSentence: {sentences[i]}")
    print("Embedding vector:")
    print(embedding) 