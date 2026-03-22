from sentence_transformers import SentenceTransformer
import json
import numpy as np

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# load stored data
with open("store.json", "r") as f:
    data = json.load(f)

query = input("Enter your question: ")

query_vector = model.encode(query)

# compute similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

results = []

for item in data:
    score = cosine_similarity(query_vector, item["vector"])
    results.append((score, item["text"]))

# sort results
results = sorted(results, reverse=True)[:2]

print("\nTop Results:\n")

for score, text in results:
    print("-", text)