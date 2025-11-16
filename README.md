# RAG Movie Search Engine

A retrieval system that supports **keyword search**, **semantic search**, **chunked semantic search**, **hybrid fusion**, and **multimodal search** over a static movie dataset. The project includes a suite of CLI tools for building indexes, generating embeddings, benchmarking retrieval quality, and issuing text or multimodal search queries.

---

## 1. Overview

This project implements a set of retrieval pipelines using:

* **Inverted Index + BM25** for keyword search
* **SentenceTransformer embeddings** for semantic similarity search
* **Chunk-level embeddings** for improved recall on long descriptions
* **Hybrid fusion** (weighted and RRF) to combine keyword + semantic signals
* **Multimodal embedding search** for image → text retrieval
* **Augmented generation** utilities for producing LLM-based answers grounded in retrieved context

All components operate on a **static movie dataset** stored in `data/movies.json`.

---

## 2. Features

### **2.1 Keyword Search**

Implements:

* Term Frequency (TF)
* Inverse Document Frequency (IDF)
* TF-IDF scoring
* BM25 ranking (primary method)

Backed by an inverted index stored under `cache/`.

### **2.2 Semantic Search**

Encodes each movie's title + description into an embedding using SentenceTransformer. Retrieval is based on cosine similarity.

Embeddings are cached in:

```
cache/movie_embeddings.npy
```

### **2.3 Chunked Semantic Search**

Descriptions are split into overlapping chunks. Each chunk receives its own embedding.
Improves matching for long documents. Chunk metadata and embeddings are stored in:

```
cache/chunk_embeddings.npy
cache/chunk_metadata.json
```

### **2.4 Hybrid Search**

Combines BM25 and semantic similarity using:

* **Weighted fusion**:

  ```
  hybrid = α * bm25 + (1 - α) * semantic
  ```
* **Reciprocal Rank Fusion (RRF)** for rank-based merging.

### **2.5 Multimodal Search**

Uses CLIP-like models for:

* text → image embeddings
* image → text retrieval

Allows image queries to surface relevant movies.

### **2.6 Augmented Generation**

Retrieves documents relevant to a query and passes them into an LLM to generate grounded answers.

---

## 3. Repository Structure

```
├── cache/                     # cached embeddings, index files, metadata
├── cli/                       # CLI modules
│   ├── keyword_search_cli.py
│   ├── semantic_search_cli.py
│   ├── multimodal_search_cli.py
│   ├── hybrid_search_cli.py
│   ├── augmented_generation_cli.py
│   ├── describe_image_cli.py
│   └── lib/                   # implementation modules
│       ├── inverted_index.py
│       ├── semantic_search.py
│       ├── chunked_semantic_search/
│       ├── hybrid_search.py
│       ├── multimodal_search.py
│       └── utils.py
├── data/                      # movies.json, images, evaluation datasets
├── README.md                  # (this file)
├── pyproject.toml
└── uv.lock
```

---

## 4. Installation

### **Prerequisites**

* Python 3.10+
* `uv` package manager

### **Setup**

```
uv sync
```

This installs all dependencies declared in `pyproject.toml`.

---

## 5. Running the CLI Tools

Each CLI module is executed using `uv run`.

### **Keyword Search**

```
uv run cli/keyword_search_cli.py search "your query"
uv run cli/keyword_search_cli.py build
```

### **Semantic Search**

```
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py embed_text "<text>"
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py embedquery "<query>"
uv run cli/semantic_search_cli.py search "<query>" --limit <k>
```

### **Chunked Semantic Search**

```
uv run cli/semantic_search_cli.py chunk "<text>" --chunk-size <n> --overlap <n>
```

### **Hybrid Search**

```
uv run cli/hybrid_search_cli.py weighted "<query>" --alpha <value> --limit <k>
uv run cli/hybrid_search_cli.py rrf "<query>" --limit <k>
```

### **Multimodal Search**

```
uv run cli/multimodal_search_cli.py verify_image_embedding --image <path>
uv run cli/multimodal_search_cli.py image_search --image <path>
```

### **Augmented Generation**

```
uv run cli/augmented_generation_cli.py rag "<query>"
uv run cli/augmented_generation_cli.py summarize "<query>" --limit <k>
uv run cli/augmented_generation_cli.py citations "<query>" --limit <k>
uv run cli/augmented_generation_cli.py question "<query>" --limit <k>
```

### **Search Evaluation**

```
uv run cli/evaluation_cli.py --limit <k>
```

## 6. Caching

Caching

All indexes and embeddings are stored under `cache/`. These include:

* `index.pkl` — inverted index
* `movie_embeddings.npy` — full-document embeddings
* `chunk_embeddings.npy` — chunk embeddings
* `chunk_metadata.json` — metadata describing chunk structure

Rebuilding occurs only if files are missing.

---


This README provides a high-level description of the entire project and its capabilities.
