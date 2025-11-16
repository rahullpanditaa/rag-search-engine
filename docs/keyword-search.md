# Keyword Search

This document provides a complete explanation of the **Keyword Search** component of the project. It covers the conceptual foundations, the implementation flow, and how the method fits into the overall retrieval system.

---

## 1. What Is Keyword Search?

Keyword search is the most traditional form of information retrieval. It matches **tokens (words)** in the user query to **documents** that contain those same tokens.

This implementation supports:

* Term Frequency (TF)
* Inverse Document Frequency (IDF)
* TF‑IDF
* BM25 (industry standard ranking algorithm, Okapi-BM25, where BM - Best Matching)

All of these operate on top of an **Inverted Index**.

---

## 2. Inverted Index: Core Data Structure

The crux of the Keyword Search is the **inverted index**, which maps:

```
term → {set of document IDs containing that term}
```

Several supporting structures are also maintained:

* `docmap`: stores full movie objects
* `term_frequencies`: `doc_id → Counter(term → count)`
* `doc_lengths`: total number of tokens per document

### Why an inverted index?

* Fast lookup of all documents containing a term
* Avoids scanning every document
* Forms the foundation for TF‑IDF and BM25

---

## 3. Tokenization

Every text (title, description, query) is run through:

```
process_text_to_tokens()
```

This ensures consistent:

* lowercasing
* stopword removal
* normalization

Keyword search relies on **exact token matching**, so it's important to pre-process the query and docs
via the same methods..

---

## 4. TF — Term Frequency

Term Frequency answers:

> *"How many times does this term appear in this document?"*

Implementation:

```
tf = term_frequencies[doc_id][term]
```

---

## 5. IDF — Inverse Document Frequency

IDF answers:

> *"How rare is this term across all documents?"*

Formula:

```
idf = log((N + 1) / (df + 1))
```

Where:

* `N` = total number of documents
* `df` = number of documents containing the term

High IDF → rare term → more informative.
Low IDF → common term → less informative.

---

## 6. TF‑IDF — Combining TF and IDF

TF‑IDF balances:

* **Term frequency** (importance *within* a document)
* **Term rarity** (importance *across* documents)

```
tf_idf = TF * IDF
```

However, TF‑IDF has limitations:

* Does not normalize for **document length**
* Treats TF linearly (doubling TF doubles score)
* Does not model realistic IR behavior

BM25 solves these.

---

## 7. BM25 — State‑of‑the‑Art Keyword Ranking

BM25 adjusts for:

### **1. Document length normalization**

Longer documents tend to contain more terms.
BM25 penalizes long documents so they don’t dominate results.

### **2. Term frequency saturation**

TF is not linear.
The impact of TF "saturates", i.e. going from 1→2 matters more than from 10→11.
BM25 models this realistically.

### **3. Term rarity (IDF)**

Uses an improved IDF variant:

```
log((N - df + 0.5) / (df + 0.5) + 1)
```

This behaves better on extreme values.

### Final BM25 formula (simplified):

```
score = IDF * ( (TF * (k1 + 1)) / (TF + k1*(1 - b + b*(doc_len/avg_doc_len))) )
```

Where:

* `k1` controls TF saturation
* `b` controls document length penalty

BM25 is used by:

* Elasticsearch
* Lucene
* Whoosh
* Many modern search systems

In this project, BM25 is the **primary ranking method** for keyword search.

---

## 8. High‑Level Search Flow

### **1. Build Phase**

* Read movie dataset
* For each movie:

  * tokenize title + description
  * update inverted index
  * update term frequencies
  * update doc lengths
  * store metadata in docmap
* Write all structures to `cache/`

### **2. Query Phase**

* Tokenize user query
* For each token:

  * fetch all matching doc IDs via inverted index
* Compute BM25 score for each doc
* Sort documents by score
* Return top N results

---

## 9. CLI Commands

### **search**

Runs simple keyword match (top 5 results).

### **tf, idf, tfidf**

Helpful for debugging and understanding term statistics.

### **bm25idf, bm25tf, bm25search**

Full BM25 scoring and ranking.

---

<!-- ## 10. Why Keyword Search Still Matters

Even though modern retrieval often uses embeddings, keyword search is still:

* precise
* interpretable
* fast
* cheap
* reliable for exact matching

Many production search engines combine keyword + semantic search (you also implement hybrid search later). -->

<!-- --- -->

## 10. Building the Inverted Index (Important)

Keyword search works **only after the inverted index is built**.

There is a dedicated CLI command for this:

### **Build Command**

```
uv run cli/keyword_search_cli.py build
```

This performs the following actions:

1. Loads movie data from file
2. Tokenizes title + description
3. Inserts tokens into `index` (term → doc IDs)
4. Computes and stores:

   * term frequencies
   * doc lengths
   * docmap entries
5. Saves all structures to `cache/`

After this, all other commands (`search`, `tf`, `idf`, `bm25search`, etc.) will work instantly without rebuilding.

---

This completes the documentation for the **Keyword Search** component.
