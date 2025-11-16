# Chunked Semantic Search

This document explains the **Chunked Semantic Search** module. It describes the motivation for chunking, how chunks are generated, how embeddings are built and cached, and how similarity-based retrieval is performed over text segments rather than entire documents.

---

## 1. Overview

Chunked semantic search improves retrieval by breaking long documents into smaller, semantically coherent pieces (**chunks**) and embedding each chunk separately. This allows:

* More fine-grained matching between query intent and specific document parts
* Better handling of long descriptions
* Higher recall, since different sections of a document may match different queries

Each chunk receives its own embedding, and retrieval is based on similarity between the query embedding and chunk embeddings.

---

## 2. Why Chunking?

Long documents contain multiple themes or ideas. Encoding an entire description as one vector forces the embedding to compress everything. Chunking offers:

### **2.1 Semantic focus**

The model can focus on smaller units of meaning.

### **2.2 Higher accuracy**

Even if only part of a document is relevant to a query, chunk-level retrieval can still surface it.

### **2.3 Improved ranking**

Scores are aggregated per document, and the best matching chunk in a doc determines the score.

---

## 3. Chunk Generation Logic

Chunks are generated using:

```
semantic_chunk_command(text, max_chunk_size, overlap)
```

### **3.1 Sentence-Based Splitting**

The text is split into sentences.

### **3.2 Chunk Construction**

Chunks are created using:

* a maximum number of sentences per chunk
* an overlap between consecutive chunks

The procedure ensures:

* All sentences appear in at least one chunk
* Short trailing fragments are not emitted as separate chunks

---

## 4. Embedding Construction

Chunks are embedded using the same SentenceTransformer model as semantic search. The process:

1. Generate all text chunks
2. Store metadata for each chunk:

   * `movie_idx` (document ID)
   * `chunk_idx` (position within document)
   * `total_chunks` (for reference)
3. Encode all chunks into embeddings
4. Save:

   * embeddings → `cache/chunk_embeddings.npy`
   * metadata → `cache/chunk_metadata.json`

Handled by:

```
build_chunk_embeddings(documents)
```

Cached embeddings are loaded through:

```
load_or_create_chunk_embeddings(documents)
```

Since the dataset is static, embeddings are built once.

---

## 5. Chunk-Level Search

### **5.1 Query Embedding**

A query is encoded into an embedding using the inherited `generate_embedding` method.

### **5.2 Similarity Computation**

For each chunk embedding:

* cosine similarity is computed against the query embedding
* scores are stored along with the associated document ID

### **5.3 Document-Level Aggregation**

Scores are aggregated as:

```
document_score = max(score of all chunks belonging to the document)
```

This ensures a document is ranked high even if only one section strongly matches.

### **5.4 Ranking**

Documents are sorted by aggregated score, and the top results are returned.

Each result contains:

* document ID
* title
* beginning of the description
* best semantic score

---

## 6. Public Methods

### **ChunkedSemanticSearch**

Provides:

* chunk creation
* chunk embedding generation
* metadata management
* chunk-level similarity search

Key methods:

* `build_chunk_embeddings(documents)`
* `load_or_create_chunk_embeddings(documents)`
* `search_chunks(query, limit)`

### **semantic_chunk_command(text, max_chunk_size, overlap)**

Utility function that splits text into overlapping chunks.

### **embed_chunks_command()**

Generates chunk embeddings and stores them.

### **search_chunked_command(query, limit)**

Runs a search and prints results.

---

## 7. CLI Usage

Chunked semantic search is available via:

```
uv run cli/semantic_search_cli.py search_chunked query "<text>" --limit <n>
```

---

## 8. Strengths and Limitations

### **Strengths**

* Fine-grained retrieval
* Improved matching for long documents
* Better handling of multi-topic descriptions

### **Limitations**

* More storage due to chunk embeddings
* Slightly slower search (more vectors to compare)
* No global understanding of entire document context

---

This concludes the documentation for the **Chunked Semantic Search** module.
