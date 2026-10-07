import json
import os

from src.embeddings import create_embeddings
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, threshold=0.80, cache_file="cache.json"):

        self.threshold = threshold
        self.cache_file = cache_file

        # Load previously saved cache
        self.cache = self.load_cache()


    def load_cache(self):

        if not os.path.exists(self.cache_file):

            return []

        with open(self.cache_file, "r", encoding="utf-8") as file:

            return json.load(file)


    def save_cache(self):

        with open(self.cache_file, "w", encoding="utf-8") as file:

            json.dump(
                self.cache,
                file,
                indent=4,
                ensure_ascii=False
            )


    def add(self, question, answer):

        self.cache.append({
            "question": question,
            "answer": answer
        })

        # Save to disk
        self.save_cache()


    def search(self, question):

        if not self.cache:

            return None

        # Create embedding for new question
        new_embedding = create_embeddings([
            {"text": question}
        ])[0]

        for item in self.cache:

            # Create embedding for cached question
            cached_embedding = create_embeddings([
                {"text": item["question"]}
            ])[0]

            similarity = cosine_similarity(
                [new_embedding],
                [cached_embedding]
            )[0][0]

            print("Cache similarity:", similarity)

            if similarity >= self.threshold:

                return item["answer"]

        return None