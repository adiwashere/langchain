  
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
embedding = OpenAIEmbeddings()
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
query = "Who is Virat Kohli?"

# Compute embeddings for documents and query
doc_embeddings = embedding.embed_documents(documents)