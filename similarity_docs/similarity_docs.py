from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Sample documents
documents = [
    "Virat Kohli is a famous Indian cricketer.",
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "The Great Wall of China is visible from space.",
    "sun is shining bright today.",
    "moon looks beautiful at night."
]

# Sample query
query = " how is the moon today?"

# Compute embeddings for documents and query
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Compute cosine similarity scores
scores = cosine_similarity([query_embedding], doc_embeddings)[0]

# Find the most similar document to the query based on similarity scores   
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print(f"Similarity Score: {score}")
