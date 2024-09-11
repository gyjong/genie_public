import faiss
from langchain_openai import OpenAIEmbeddings
import numpy as np
import os
import pickle

embeddings = OpenAIEmbeddings()
vector_store = None
index_file = "data/faiss_index.pkl"
text_file = "data/text_data.pkl"

def initialize_vector_store():
    global vector_store
    if os.path.exists(index_file):
        with open(index_file, "rb") as f:
            vector_store = pickle.load(f)
    else:
        dimension = 1536  # OpenAI embeddings dimension
        vector_store = faiss.IndexFlatL2(dimension)

def save_to_vector_store(text):
    initialize_vector_store()
    global vector_store
    
    vector = embeddings.embed_query(text)
    vector_store.add(np.array([vector]))

    with open(index_file, "wb") as f:
        pickle.dump(vector_store, f)
    
    # Save the text data
    if os.path.exists(text_file):
        with open(text_file, "rb") as f:
            text_data = pickle.load(f)
    else:
        text_data = []
    
    text_data.append(text)
    
    with open(text_file, "wb") as f:
        pickle.dump(text_data, f)

def search_vector_store(query, k=5):
    initialize_vector_store()
    global vector_store

    if vector_store.ntotal == 0:
        return []  # Return empty list if no vectors are stored

    query_vector = embeddings.embed_query(query)
    distances, indices = vector_store.search(np.array([query_vector]), k)

    # Retrieve the actual text data
    if os.path.exists(text_file):
        with open(text_file, "rb") as f:
            text_data = pickle.load(f)
    else:
        text_data = []
    
    results = []
    for i in indices[0]:
        if 0 <= i < len(text_data):
            results.append(text_data[i])
    return results

# Ensure the data directory exists
os.makedirs(os.path.dirname(index_file), exist_ok=True)