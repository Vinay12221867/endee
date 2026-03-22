from sentence_transformers import SentenceTransformer
import endee

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# read data
with open("data.txt", "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# generate embeddings
embeddings = model.encode(lines)

# store locally (simple approach using Endee format)
data_store = []

for i in range(len(lines)):
    data_store.append({
        "text": lines[i],
        "vector": embeddings[i].tolist()
    })

# save to file (simulate vector DB usage)
import json
with open("store.json", "w") as f:
    json.dump(data_store, f)

print("Data stored successfully!")