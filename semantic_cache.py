
import json
import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.embeddings import create_embeddings


class SemanticCache:

    def __init__(self, threshold=0.80, cache_file="cache.json"):

        self.threshold = threshold
        self.cache_file = cache_file

        self.cache = self.load_cache()

    # ========================================================
    # LOAD CACHE
    # ========================================================

    def load_cache(self):

        if not os.path.exists(self.cache_file):
            return []

        try:

            with open(
                self.cache_file,
                "r",
                encoding="utf-8"
            ) as file:

                return json.load(file)

        except (json.JSONDecodeError, OSError):

            return []

    # ========================================================
    # SAVE CACHE
    # ========================================================

    def save_cache(self):

        with open(
            self.cache_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.cache,
                file,
                indent=4,
                ensure_ascii=False
            )

    # ========================================================
    # NORMALIZE QUESTION
    # ========================================================

    def normalize_question(self, question):

        return " ".join(
            question.strip().lower().split()
        )

    # ========================================================
    # ADD QUESTION TO CACHE
    # ========================================================

    def add(self, question, answer):

        embedding = create_embeddings([
            {"text": question}
        ])[0]

        self.cache.append({

            "question": question,

            "answer": answer,

            "embedding": embedding.tolist()

        })

        self.save_cache()

    # ========================================================
    # SEARCH CACHE
    # ========================================================

    def search(self, question):

        if not self.cache:
            return None

        normalized_question = self.normalize_question(
            question
        )

        # ====================================================
        # STEP 1 — EXACT MATCH
        # ====================================================

        for item in self.cache:

            cached_question = item.get(
                "question",
                ""
            )

            normalized_cached_question = (
                self.normalize_question(cached_question)
            )

            if normalized_question == normalized_cached_question:

                print("Exact cache match ✅")

                return item.get("answer")

        # ====================================================
        # STEP 2 — SEMANTIC MATCH
        # ====================================================

        new_embedding = create_embeddings([
            {"text": question}
        ])[0]

        cache_updated = False

        for item in self.cache:

            # ------------------------------------------------
            # Get cached embedding
            # ------------------------------------------------

            if "embedding" not in item:

                cached_embedding = create_embeddings([
                    {
                        "text": item["question"]
                    }
                ])[0]

                item["embedding"] = (
                    cached_embedding.tolist()
                )

                cache_updated = True

            else:

                cached_embedding = np.array(
                    item["embedding"],
                    dtype="float32"
                )

            # ------------------------------------------------
            # Calculate similarity
            # ------------------------------------------------

            similarity = cosine_similarity(
                [new_embedding],
                [cached_embedding]
            )[0][0]

            print(
                f"Cache similarity: {similarity:.4f}"
            )

            # ------------------------------------------------
            # Semantic HIT
            # ------------------------------------------------

            if similarity >= self.threshold:

                if cache_updated:
                    self.save_cache()

                print("Semantic cache match ✅")

                return item.get("answer")

        # ====================================================
        # CACHE MISS
        # ====================================================

        if cache_updated:
            self.save_cache()

        print("Cache miss ❌")

        return None

