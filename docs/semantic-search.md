# Semantic Search

This document explains the **Semantic Search** component of the project. It covers the conceptual basis of semantic retrieval, describes how the system embeds documents, and explains how similarity-based ranking is performed.

---

## 1. Overview

Semantic search ranks documents by **meaning**, not by keyword overlap. It relies on *vector embeddings*, where:

* Each document is mapped to a high‑dimensional vector
* A query is mapped to a similar vector
* Relevance is measured by their **cosine similarity**

Unlike keyword search, semantic search is robust to:

* Synonyms
* Paraphrasing
* Different phrasing with similar meaning

This system uses a SentenceTransformer model to embed a static movie dataset.

The dimensions on their own do not mean anything, but taken altogether, they're a measure of a vast number of semantic metrics, like
how happy or sad a sentence is, how angry etc. Natural language has a lot of semantics, and a vector embedding is supposed to be a 
quantification of the semantic meaning of the text.

---

## 2. Core Concepts

### **2.1 Embeddings**

An embedding is a numerical vector that encodes semantic information. Documents that are semantically similar end up close together in vector space.

### **2.2 Cosine Similarity**

Similarity between two vectors is measured as:

```
dot(v1, v2) / (||v1|| * ||v2||)
```

Scores range from -1 (opposite) to +1 (identical).

### **2.3 Static Dataset**

The dataset does not change. Embeddings can be generated once and reused.

---

## 3. System Architecture

### **3.1 Document Preparation**

Each document is represented using:

```
"title: description"
```

These strings are passed into the embedding model.

### **3.2 Embedding Generation**

For each document:

* The model encodes the combined title+description
* The embedding matrix is stored on disk in `cache/`
* A document map is maintained for ID lookup

Embeddings are generated through:

```
build_embeddings(documents)
```

### **3.3 Loading Cached Embeddings**

The system checks whether embeddings exist on disk and whether the shapes match the number of documents. If so, it loads them; otherwise, it regenerates.

Handled by:

```
load_or_create_embeddings(documents)
```

---

## 4. Search Process

### **4.1 Query Embedding**

The query string is encoded using the same model.

### **4.2 Similarity Computation**

Each document embedding is compared to the query embedding using cosine similarity.

### **4.3 Ranking**

Scores are sorted in descending order, and the top *k* results are returned.

Returned objects include:

* title
* description excerpt
* similarity score

---

## 5. Classes and Methods

### **SemanticSearch**

Responsible for:

* Generating embeddings
* Storing and loading embeddings
* Mapping document IDs
* Computing similarity‑based rankings

Key methods:

* `generate_embedding(text)`
* `build_embeddings(documents)`
* `load_or_create_embeddings(documents)`
* `search(query, limit)`

### **cosine_similarity(vec1, vec2)`**

Utility function implementing cosine similarity.

---

## 6. Strengths and Limitations

### **Strengths**

* Captures semantic meaning
* Handles paraphrasing and synonyms
* Provides high‑quality retrieval for natural‑language queries

### **Limitations**

* Cannot handle exact keyword constraints
* Ignores document structure (title vs. description weight)
* Requires embedding computation

---

## 7. CLI Usage

Semantic search is available via:

```
uv run cli/semantic_search_cli.py <subcommand> [options]
```

Subcommands include:

```
uv run cli/semantic_search_cli.py verify
uv run cli/semantic_search_cli.py embed_text "<text>"
uv run cli/semantic_search_cli.py verify_embeddings
uv run cli/semantic_search_cli.py embedquery "<query>"
uv run cli/semantic_search_cli.py search "<query>" --limit <k>
```

---

This concludes the documentation for the **Semantic Search** module.
