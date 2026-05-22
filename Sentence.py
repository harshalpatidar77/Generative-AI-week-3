from sentence_transformers import SentenceTransformer

model = SentenceTransformer("ibm-granite/granite-embedding-97m-multilingual-r2")

sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium."
]
embeddings = model.encode(sentences)
print(embeddings)

similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)