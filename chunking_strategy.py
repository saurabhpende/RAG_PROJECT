import pandas as pd
import openai
import numpy as np
from sklearn.cluster import KMeans
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
# Load dataset
df = pd.read_csv("Dataset_Banking_chatbot.csv", encoding="ISO-8859-1")

# Ensure correct column names
QUESTION_COLUMN = "Query"
ANSWER_COLUMN = "Response"

client = openai.OpenAI()  # Initialize client

def get_embedding(text):
    """Generate OpenAI embedding for a given text."""
    response = client.embeddings.create(
        input=[text], model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Generate embeddings for all questions
df["combined_text"] = df[QUESTION_COLUMN] + " " + df[ANSWER_COLUMN]
df["embedding"] = df["combined_text"].apply(get_embedding)

def cluster_embeddings(embeddings, num_clusters=5):
    """Apply KMeans clustering on embeddings."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

# Convert embeddings to NumPy array
embeddings = np.vstack(df["embedding"].values)
df["cluster"] = cluster_embeddings(embeddings, num_clusters=10)  # Adjust clusters as needed

# Group by clusters to create chunks
chunks = df.groupby("cluster").apply(lambda x: {
    "questions": x[QUESTION_COLUMN].tolist(),
    "answers": x[ANSWER_COLUMN].tolist()
}).to_list()

# Save chunks as JSON
with open("semantic_chunks.json", "w") as f:
    json.dump(chunks, f, indent=4)

print("Semantic chunking completed. Chunks saved to 'semantic_chunks.json'")
