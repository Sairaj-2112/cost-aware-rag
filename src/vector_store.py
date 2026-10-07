import faiss
import numpy as np


def create_vector_store(embeddings):

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)#creates our FAISS search index.
    #L2 → it uses L2 / Euclidean distance to measure how close two vectors are.
    index.add(embeddings)

    return index