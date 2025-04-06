import json
import pandas as pd
import numpy
import openai
from dotenv import load_dotenv
import os

load_dotenv()

with open("semantic_chunks.json") as f:
    chunks = json.load(f) 

client = openai.OpenAI()  # Initialize client

def get_embedding(text):
    """Generate OpenAI embedding for a given text."""
    response = client.embeddings.create(
        input=[text], model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Embed each chunk
for chunk in chunks:
    combined_text = " ".join(chunk["questions"] + chunk["answers"])
    chunk["embedding"] = get_embedding(combined_text)

# Save updated chunks
EMBEDDED_CHUNKS_PATH = "embedded_chunks.json"
with open(EMBEDDED_CHUNKS_PATH, "w") as f:
    json.dump(chunks, f, indent=4)

print(f"Embeddings generated. Updated chunks saved to {EMBEDDED_CHUNKS_PATH}")
