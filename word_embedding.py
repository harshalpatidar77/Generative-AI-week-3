from gensim.models import Word2Vec  
# sample sentences 
sentences = [
    ["i", "love", "nlp"], 
    ["nlp", "is", "interesting"],
    ["word2vec", "creates", "embeddings"], 
    ["i", "love", "machine learning"], 
    ["machine", "learning","uses","nlp"], 
] 
# Train Word2Vec 
model = Word2Vec(
    sentences,
    vector_size=10,   # embedding size
    window=3,         # context window
    min_count=1,      # include all words 
    workers=4
) 
# get embedding vector of a word 
vector = model.wv["nlp"] 
print("Embedding of 'nlp':\n") 
print(vector) 
# find similar words 
similar_words = model.wv.most_similar("nlp") 
print("\nSimilar words to 'nlp':\n")
for word, score in similar_words:
    print(f"{word} : {score}")  