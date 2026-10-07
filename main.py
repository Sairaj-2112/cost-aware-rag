from src.pdf_loader import load_pdf
from src.chunking import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.retriever import search
from src.rag import create_prompt
from src.semantic_cache import SemanticCache
from src.model_router import choose_model
from src.cost_tracker import calculate_cost
from src.metrics import create_metrics, save_metrics, print_metrics

import os
from huggingface_hub import InferenceClient


# -----------------------------
# 1. Ask a question
# -----------------------------

query = "What is an AutoEncoder?"

print("\nSearching for:", query)


# -----------------------------
# 2. Check semantic cache FIRST
# -----------------------------

cache = SemanticCache(threshold=0.80)

metrics = create_metrics()

metrics["total_queries"] += 1

cached_answer = cache.search(query)


if cached_answer:

    metrics["cache_hits"] += 1

    print("\n========== CACHE HIT ✅ ==========")
    print("Answer from cache:")
    print(cached_answer)

    print_metrics(metrics)

    save_metrics(metrics)

    exit()


else:

    metrics["cache_misses"] += 1

    print("\n========== CACHE MISS ❌ ==========")
    print("No similar question found.")
    print("Running RAG...")


# -----------------------------
# 3. Load PDF
# -----------------------------

pages = load_pdf("data/syllabus.pdf")

print("\nTotal pages:", len(pages))


# -----------------------------
# 4. Create chunks
# -----------------------------

all_chunks = create_chunks(pages)

print("Total chunks:", len(all_chunks))


# -----------------------------
# 5. Create embeddings
# -----------------------------

print("\nCreating embeddings...")

embeddings = create_embeddings(all_chunks)

print("Embedding shape:", embeddings.shape)


# -----------------------------
# 6. Create FAISS vector store
# -----------------------------

print("\nCreating FAISS vector store...")

index = create_vector_store(embeddings)

print("Total vectors in FAISS:", index.ntotal)


# -----------------------------
# 7. Convert question to embedding
# -----------------------------

query_embedding = create_embeddings([
    {"text": query}
])


# -----------------------------
# 8. Retrieve relevant chunks
# -----------------------------

results = search(
    index,
    query_embedding,
    all_chunks,
    top_k=5
)


# -----------------------------
# 9. Create RAG prompt
# -----------------------------

prompt = create_prompt(query, results)


# -----------------------------
# 10. Connect to Hugging Face
# -----------------------------

token = os.getenv("HF_TOKEN")

if not token:
    raise ValueError("HF_TOKEN is not set")


client = InferenceClient(
    api_key=token
)


# -----------------------------
# 11. Select model using router
# -----------------------------

selected_model = choose_model(query)

print("\nSelected model:", selected_model)


# -----------------------------
# 12. Ask the selected LLM
# -----------------------------

response = client.chat.completions.create(
    model=selected_model,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    max_tokens=500
)


# -----------------------------
# 13. Print answer
# -----------------------------

print("\n========== RAG ANSWER ==========")

answer = response.choices[0].message.content

print(answer)


# -----------------------------
# 13.1 Token usage
# -----------------------------

print("\n========== TOKEN USAGE ==========")

if hasattr(response, "usage") and response.usage:

    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    print("Prompt tokens:", prompt_tokens)
    print("Completion tokens:", completion_tokens)
    print("Total tokens:", total_tokens)

    metrics["total_tokens"] += total_tokens

else:

    print("Token usage information not available.")


# -----------------------------
# 13.2 Calculate LLM cost
# -----------------------------

if hasattr(response, "usage") and response.usage:

    # Model-specific pricing

    if selected_model == "openai/gpt-oss-20b:groq":

        prompt_price = 0.075
        completion_price = 0.30

    elif selected_model == "openai/gpt-oss-120b:groq":

        prompt_price = 0.15
        completion_price = 0.60

    else:

        raise ValueError("Unknown model pricing")


    cost = calculate_cost(
        prompt_tokens,
        completion_tokens,
        prompt_price,
        completion_price
    )

    metrics["total_cost"] += cost

    print("\n========== COST ==========")
    print("Estimated cost: $", round(cost, 8))


# -----------------------------
# 14. Save answer to cache
# -----------------------------

cache.add(query, answer)

print("\nAnswer saved to semantic cache ✅")


# -----------------------------
# 15. Show retrieved chunks
# -----------------------------

print("\n========== TOP 5 RETRIEVED CHUNKS ==========")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("Page:", result["page"])
    print("Distance:", result["distance"])
    print("Text:", result["text"])


# -----------------------------
# 16. Show and save metrics
# -----------------------------

print_metrics(metrics)

save_metrics(metrics)

print("\nMetrics saved to metrics.json ✅")