import faiss
import numpy as np
import json
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

#Load FAISS Index
index = faiss.read_index('faiss_index')
with open('embedded_chunks.json','r') as f:
    chunks = json.load(f)

client = openai.OpenAI()  # Initialize client

def get_embedding(text):
    """Generate OpenAI embedding for a given text."""
    response = client.embeddings.create(
        input=[text], model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def retrieve_answer(query,k = 3,threshold = 0.3):
    query_embedding = np.array([get_embedding(query)]).astype('float32')
    faiss.normalize_L2(query_embedding)

    similarity_scores,best_match_indices = index.search(query_embedding,k)

    similarity_scores = similarity_scores[0]  # Extract first row of scores

    # Filter results based on similarity threshold
    filtered_chunks = []
    for i, idx in enumerate(best_match_indices[0]):
        if similarity_scores[i] >= threshold:  # Only keep relevant chunks
            filtered_chunks.append(chunks[idx])

    # If no relevant documents are found, return a default response
    #if not filtered_chunks:
        #return "⚠️ Sorry, I couldn't find any relevant information for your query."

    return filtered_chunks  # Return retrieved questions & answers
