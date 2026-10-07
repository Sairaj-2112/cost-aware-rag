from src.semantic_cache import SemanticCache


# Create cache
cache = SemanticCache(threshold=0.80)


# Store first question + answer
cache.add(
    "What is an AutoEncoder?",
    "An AutoEncoder is a neural network that learns to reconstruct its input."
)


# Ask a similar question
question = "Can you explain AutoEncoders?"

answer = cache.search(question)


if answer:

    print("\nCACHE HIT ✅")
    print("Answer:", answer)

else:

    print("\nCACHE MISS ❌")
    print("No similar question found.")