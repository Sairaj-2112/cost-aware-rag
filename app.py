import os
import gradio as gr

from huggingface_hub import InferenceClient
from groq import Groq

from src.pdf_loader import load_pdf
from src.chunking import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import create_vector_store
from src.retriever import search
from src.rag import create_prompt
from src.semantic_cache import SemanticCache
from src.model_router import choose_model
from src.cost_tracker import calculate_cost
from src.metrics import create_metrics, save_metrics


# ============================================================
# 1. LOAD DEFAULT PDF
# ============================================================

print("Loading default syllabus...")

pages = load_pdf("data/syllabus.pdf")

print(f"Total pages: {len(pages)}")


# ============================================================
# 2. CREATE CHUNKS
# ============================================================

chunks = create_chunks(pages)

print(f"Total chunks: {len(chunks)}")


# ============================================================
# 3. CREATE EMBEDDINGS
# ============================================================

print("Creating embeddings...")

embeddings = create_embeddings(chunks)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 4. CREATE FAISS VECTOR STORE
# ============================================================

print("Creating FAISS vector store...")

index = create_vector_store(embeddings)

print("RAG system ready ✅")


# ============================================================
# 5. SEMANTIC CACHE
# ============================================================

cache = SemanticCache(
    threshold=0.80
)


# ============================================================
# 6. METRICS
# ============================================================

metrics = create_metrics()


# ============================================================
# 7. HUGGING FACE CLIENT
# ============================================================

hf_token = os.getenv("HF_TOKEN")

hf_client = None

if hf_token:

    hf_client = InferenceClient(
        api_key=hf_token
    )

    print("Hugging Face client ready ✅")

else:

    print("⚠️ HF_TOKEN not found.")


# ============================================================
# 8. GROQ CLIENT
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")

groq_client = None

if groq_api_key:

    groq_client = Groq(
        api_key=groq_api_key
    )

    print("Groq fallback client ready ✅")

else:

    print("⚠️ GROQ_API_KEY not found.")


# ============================================================
# 9. PDF PROCESSING FUNCTION
# ============================================================

def process_pdf(pdf_path):

    global pages
    global chunks
    global embeddings
    global index
    global cache

    if not pdf_path:

        return "📚 Using the default syllabus."

    try:

        print()
        print("========================================")
        print("Processing uploaded PDF...")
        print("========================================")

        # ----------------------------------------------------
        # Load PDF
        # ----------------------------------------------------

        pages = load_pdf(pdf_path)

        print(
            f"Uploaded PDF pages: {len(pages)}"
        )

        # ----------------------------------------------------
        # Create chunks
        # ----------------------------------------------------

        chunks = create_chunks(pages)

        print(
            f"Uploaded PDF chunks: {len(chunks)}"
        )

        # ----------------------------------------------------
        # Create embeddings
        # ----------------------------------------------------

        print("Creating embeddings...")

        embeddings = create_embeddings(chunks)

        print(
            "Embedding shape:",
            embeddings.shape
        )

        # ----------------------------------------------------
        # Create FAISS index
        # ----------------------------------------------------

        print("Creating FAISS vector store...")

        index = create_vector_store(
            embeddings
        )

        # ----------------------------------------------------
        # Clear old semantic cache
        # ----------------------------------------------------

        cache = SemanticCache(
            threshold=0.80
        )

        cache.cache = []

        cache.save_cache()

        print("Old semantic cache cleared ✅")

        print("New PDF RAG system ready ✅")

        return f"""
## ✅ PDF Loaded Successfully

**Pages:** {len(pages)}

**Chunks:** {len(chunks)}

**Embeddings:** {embeddings.shape[1]} dimensions

Your questions will now be answered using the **uploaded PDF**.
"""

    except Exception as e:

        print("PDF processing failed ❌")
        print("Reason:", str(e))

        return f"""
## ❌ PDF Processing Failed

Something went wrong while processing the uploaded PDF.

**Error:**
`{str(e)}`
"""


# ============================================================
# 10. METRICS DISPLAY
# ============================================================

def get_metrics():

    total_queries = metrics["total_queries"]
    cache_hits = metrics["cache_hits"]
    cache_misses = metrics["cache_misses"]
    total_tokens = metrics["total_tokens"]
    total_cost = metrics["total_cost"]

    if total_queries > 0:

        hit_rate = (
            cache_hits / total_queries
        ) * 100

    else:

        hit_rate = 0

    return f"""
### 📊 System Metrics

| Metric | Value |
|---|---:|
| 🔢 Total Queries | **{total_queries}** |
| ✅ Cache Hits | **{cache_hits}** |
| ❌ Cache Misses | **{cache_misses}** |
| 📈 Cache Hit Rate | **{hit_rate:.2f}%** |
| 🔤 Total Tokens | **{total_tokens}** |
| 💰 Total Cost | **${total_cost:.8f}** |
"""


# ============================================================
# 11. MAIN QUESTION FUNCTION
# ============================================================

def ask_question(question):

    if not question or not question.strip():

        return (
            "⚠️ **Please enter a question first.**",
            get_metrics()
        )

    question = question.strip()

    metrics["total_queries"] += 1

    # ========================================================
    # CACHE CHECK
    # ========================================================

    cached_answer = cache.search(question)

    if cached_answer:

        metrics["cache_hits"] += 1

        save_metrics(metrics)

        answer_output = f"""
## 🟢 Cache Hit

**Answer served directly from the semantic cache.**

---

## 💬 Answer

{cached_answer}

---

### ⚡ Cost Saved

No new LLM request was required for this query.
"""

        return (
            answer_output,
            get_metrics()
        )

    # ========================================================
    # CACHE MISS
    # ========================================================

    metrics["cache_misses"] += 1

    # ========================================================
    # QUESTION EMBEDDING
    # ========================================================

    query_embedding = create_embeddings(
        [
            {
                "text": question
            }
        ]
    )

    # ========================================================
    # FAISS RETRIEVAL
    # ========================================================

    results = search(
        index,
        query_embedding,
        chunks,
        top_k=5
    )

    # ========================================================
    # RAG PROMPT
    # ========================================================

    prompt = create_prompt(
        question,
        results
    )

    # ========================================================
    # MODEL ROUTING
    # ========================================================

    selected_model = choose_model(
        question
    )

    print()
    print("Selected model:", selected_model)

    answer = None
    response = None
    provider_used = None
    actual_model = selected_model

    # ========================================================
    # TRY HUGGING FACE
    # ========================================================

    if hf_client:

        try:

            print("Trying Hugging Face...")

            response = hf_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500
            )

            answer = response.choices[0].message.content

            provider_used = "Hugging Face"

            print("Hugging Face success ✅")

        except Exception as e:

            print("Hugging Face failed ❌")
            print("Reason:", str(e))

    # ========================================================
    # GROQ FALLBACK
    # ========================================================

    if answer is None and groq_client:

        try:

            print("Switching to Groq fallback...")

            groq_model = selected_model.replace(
                ":groq",
                ""
            )

            print(
                "Groq model:",
                groq_model
            )

            response = groq_client.chat.completions.create(
                model=groq_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=500
            )

            answer = response.choices[0].message.content

            provider_used = "Groq"

            actual_model = groq_model

            print("Groq fallback success ✅")

        except Exception as e:

            print("Groq fallback failed ❌")
            print("Reason:", str(e))

    # ========================================================
    # BOTH PROVIDERS FAILED
    # ========================================================

    if answer is None:

        save_metrics(metrics)

        error_output = f"""
## ⚠️ LLM Unavailable

Both configured LLM providers were unable
to generate an answer.

### 🤖 Selected Model

`{selected_model}`

### ✅ What Worked

- ⚡ Semantic Cache
- 🔎 FAISS Retrieval
- 📚 RAG Prompt Generation
- 🤖 Model Routing

### ❌ What Failed

- Hugging Face
- Groq fallback

No new LLM tokens or cost were recorded.
"""

        return (
            error_output,
            get_metrics()
        )

    # ========================================================
    # TOKEN TRACKING
    # ========================================================

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if hasattr(response, "usage") and response.usage:

        usage = response.usage

        prompt_tokens = getattr(
            usage,
            "prompt_tokens",
            0
        )

        completion_tokens = getattr(
            usage,
            "completion_tokens",
            0
        )

        total_tokens = getattr(
            usage,
            "total_tokens",
            0
        )

        metrics["total_tokens"] += total_tokens

    # ========================================================
    # MODEL PRICING
    # ========================================================

    if actual_model == "openai/gpt-oss-20b":

        prompt_price = 0.075
        completion_price = 0.30

    elif actual_model == "openai/gpt-oss-120b":

        prompt_price = 0.15
        completion_price = 0.60

    else:

        prompt_price = 0.075
        completion_price = 0.30

    # ========================================================
    # COST
    # ========================================================

    cost = calculate_cost(
        prompt_tokens,
        completion_tokens,
        prompt_price,
        completion_price
    )

    metrics["total_cost"] += cost

    # ========================================================
    # SAVE TO CACHE
    # ========================================================

    cache.add(
        question,
        answer
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    save_metrics(metrics)

    # ========================================================
    # RETRIEVAL INFORMATION
    # ========================================================

    retrieval_info = ""

    for i, result in enumerate(
        results,
        start=1
    ):

        retrieval_info += f"""
**Result {i}**

- 📄 Page: `{result['page']}`
- 📏 Distance: `{result['distance']:.4f}`

"""

    # ========================================================
    # FINAL ANSWER
    # ========================================================

    answer_output = f"""
## 🟠 Cache Miss

**Running the RAG pipeline...**

---

## 💬 Answer

{answer}

---

## 🤖 Model Used

`{actual_model}`

### Provider

**{provider_used}**

---

## 🔤 Token Usage

| Type | Tokens |
|---|---:|
| Prompt | `{prompt_tokens}` |
| Completion | `{completion_tokens}` |
| **Total** | `{total_tokens}` |

---

## 💰 Query Cost

**${cost:.8f}**

---

## 🔎 Retrieved Context

{retrieval_info}
"""

    return (
        answer_output,
        get_metrics()
    )


# ============================================================
# 12. GRADIO UI
# ============================================================

custom_css = """

body {
    background: #f7f8fa;
}

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

.title {
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 25px;
}

.section-title {
    margin-top: 15px;
}

.metric-card {
    border-radius: 12px;
}

footer {
    display: none !important;
}

"""


with gr.Blocks(
    title="Cost-Aware RAG Study Assistant"
) as app:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.Markdown(
        """
        <div class="title">

        # 📚 Cost-Aware RAG Study Assistant

        </div>

        <div class="subtitle">

        **Semantic Cache → RAG → Model Routing → Cost Tracking**

        Ask questions from any uploaded PDF and get grounded
        answers while monitoring AI usage and cost.

        </div>
        """
    )

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    gr.Markdown(
        """
        ### 🔄 AI Pipeline

        **📄 Upload PDF → ✂️ Chunking → 🧠 Embeddings → 🔎 FAISS
        → ⚡ Semantic Cache → 🤖 Model Router → 💬 Answer**
        """
    )

    gr.Markdown("---")

    # --------------------------------------------------------
    # PDF UPLOAD
    # --------------------------------------------------------

    gr.Markdown(
        "## 📄 Upload Your PDF"
    )

    pdf_file = gr.File(
        label="Upload a PDF document",
        file_types=[".pdf"],
        type="filepath"
    )

    pdf_status = gr.Markdown(
        "📚 Currently using the default syllabus."
    )

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    gr.Markdown(
        "## 🔍 Ask Your Question"
    )

    question = gr.Textbox(
        show_label=False,
        placeholder="Example: What is an AutoEncoder?",
        lines=3
    )

    with gr.Row():

        submit = gr.Button(
            "🚀 Ask Question",
            variant="primary"
        )

        clear = gr.ClearButton(
            components=[question],
            value="🗑️ Clear"
        )

    # --------------------------------------------------------
    # ANSWER
    # --------------------------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        "## 💬 Answer"
    )

    answer = gr.Markdown(
        "Your answer will appear here."
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    gr.Markdown("---")

    metrics_display = gr.Markdown(
        get_metrics()
    )

    # --------------------------------------------------------
    # HOW SYSTEM WORKS
    # --------------------------------------------------------

    gr.Markdown("---")

    gr.Markdown(
        """
        ## 🧩 How This System Works

        | Component | Purpose |
        |---|---|
        | 📄 **PDF Upload** | Allows users to provide their own document |
        | ⚡ **Semantic Cache** | Reuses answers for similar questions |
        | 🔎 **FAISS Retrieval** | Finds relevant PDF chunks |
        | 📚 **RAG** | Grounds the LLM answer in the uploaded PDF |
        | 🤖 **Model Router** | Selects a cheaper or stronger model |
        | 🔤 **Token Tracking** | Measures prompt and completion tokens |
        | 💰 **Cost Tracking** | Calculates estimated LLM cost |
        | 📊 **Metrics** | Tracks queries, hits, misses, tokens and cost |

        **Goal:** Reduce unnecessary LLM calls while maintaining
        useful document-grounded answers.
        """
    )

    # ========================================================
    # EVENTS
    # ========================================================

    pdf_file.change(
        fn=process_pdf,
        inputs=pdf_file,
        outputs=pdf_status
    )

    submit.click(
        fn=ask_question,
        inputs=question,
        outputs=[
            answer,
            metrics_display
        ]
    )

    question.submit(
        fn=ask_question,
        inputs=question,
        outputs=[
            answer,
            metrics_display
        ]
    )


# ============================================================
# 13. LAUNCH
# ============================================================

app.launch(
    css=custom_css
)