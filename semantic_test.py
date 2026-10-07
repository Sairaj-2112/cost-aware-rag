from src.embeddings import create_embeddings
from sklearn.metrics.pairwise import cosine_similarity


question1 = "What is an AutoEncoder?"

question2 = "Can you explain AutoEncoders?"


# Create embeddings
embeddings = create_embeddings([
    {"text": question1},
    {"text": question2}
])


# Calculate similarity
similarity = cosine_similarity(
    [embeddings[0]],
    [embeddings[1]]
)


print("Question 1:", question1)
print("Question 2:", question2)

print("\nSemantic Similarity:")
print(similarity[0][0])