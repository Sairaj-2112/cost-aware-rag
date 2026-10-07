def search(index, query_embedding, chunks, top_k=3):

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, index_position in zip(distances[0], indices[0]):#FAISS, find the 3 closest vectors to my question."

        results.append({
            "page": chunks[index_position]["page"],
            "text": chunks[index_position]["text"],
            "distance": distance
        })

    return results
