import faiss
import numpy as np
import json

# Load embedded chunks
EMBEDDED_CHUNKS_PATH = "embedded_chunks.json"
with open(EMBEDDED_CHUNKS_PATH, "r") as f:
    chunks = json.load(f)

# Extract embeddings
dimension = len(chunks[0]["embedding"])
index = faiss.IndexFlatL2(dimension)

# Convert embeddings to NumPy array
embeddings_array = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")

# Add to FAISS index
index.add(embeddings_array)

# Save index
FAISS_INDEX_PATH = "faiss_index"
faiss.write_index(index, FAISS_INDEX_PATH)

print(f"FAISS index saved to {FAISS_INDEX_PATH}")
