# Cost-Aware RAG System with Semantic Cache & Model Routing

A Retrieval-Augmented Generation (RAG) system that answers questions from a university syllabus PDF while reducing unnecessary LLM calls using semantic caching and model routing.

The system combines:

- PDF text extraction
- Intelligent text chunking
- Sentence embeddings
- FAISS vector search
- Retrieval-Augmented Generation (RAG)
- Semantic cache
- Model routing
- Token usage tracking
- Cost calculation
- Persistent metrics
- Gradio web interface
- Hugging Face + Groq fallback providers

---

## 🚀 Project Overview

Traditional RAG systems send every user question to an LLM.

This can increase:

- API usage
- Cost
- Latency
- Repeated computation

This project introduces a semantic cache.

If a user asks the same or a semantically similar question again, the system can return the previously generated answer instead of calling the LLM again.

For new questions, the system retrieves relevant information from the syllabus and sends the context to an LLM.

A model router also selects a smaller model for simpler questions and a larger model for more complex questions.

---

## 🏗️ System Architecture

```text
                    University Syllabus PDF
                              │
                              ▼
                       Text Extraction
                              │
                              ▼
                           Chunking
                              │
                              ▼
                         Embeddings
                              │
                              ▼
                       FAISS Vector Store
                              │
                              │
                     ┌────────▼────────┐
                     │  User Question  │
                     └────────┬────────┘
                              │
                              ▼
                       Semantic Cache
                         /          \
                       HIT          MISS
                        │             │
                        ▼             ▼
                 Cached Answer   FAISS Retrieval
                                      │
                                      ▼
                                RAG Prompt
                                      │
                                      ▼
                                Model Router
                                /          \
                               /            \
                         20B Model       120B Model
                               \            /
                                \          /
                                 ▼        ▼
                                   LLM
                                    │
                                    ▼
                                  Answer
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                    Save Cache            Track Metrics



                    ## 🔄 How It Works

### Basic RAG Pipeline

```text
PDF
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Retrieve Relevant Context
 ↓
LLM
 ↓
Answer
```

### Semantic Cache Flow

```text
User Question
      ↓
Create Embedding
      ↓
Compare with Cached Questions
      ↓
   Similar?
    ↙    ↘
  YES     NO
   ↓       ↓
Cache     RAG
Answer     ↓
   ↓       LLM
   ↓       ↓
   └──→ Return Answer
```

If a similar question already exists in the cache, the system returns the stored answer and avoids another LLM call.

### Cache Persistence

```text
User Question
      ↓
LLM Generates Answer
      ↓
Question + Answer
      ↓
cache.add()
      ↓
Python Data Structure
      ↓
json.dump()
      ↓
cache.json
```

The cache is persisted in `cache.json`, allowing cached responses to remain available across application restarts.

---

## 💰 Cost Tracking

The system tracks prompt tokens, completion tokens, total tokens, and estimated API cost.

| Model        | Input / 1M Tokens | Output / 1M Tokens |
| ------------ | ----------------: | -----------------: |
| GPT-OSS 20B  |        **$0.075** |          **$0.30** |
| GPT-OSS 120B |         **$0.15** |          **$0.60** |

The model router uses the smaller model for simpler questions and the larger model for more complex questions, helping reduce unnecessary LLM costs.

---

## 📊 Metrics

The system tracks:

* Total queries
* Cache hits
* Cache misses
* Cache hit rate
* Total tokens
* Estimated total cost

This makes it possible to measure how semantic caching and model routing affect LLM usage and cost.

---

## 🖥️ User Interface

A Gradio interface allows users to:

* Enter syllabus-related questions
* View generated answers
* See source pages
* See cache HIT/MISS status
* View selected model
* View token usage
* View estimated cost
* Monitor system metrics
