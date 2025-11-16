import math
from .inverted_index import InvertedIndex
from .utils import process_text_to_tokens
from .constants import BM25_K1,  BM25_B

def build_command() -> None:
    index = InvertedIndex()
    print("Building the Inverted Index...")
    index.build()
    index.save()
    print("Inverted Index built!! Saved to cache on disk.")
        
def search(search_query: str) -> list[dict]:
    index = InvertedIndex()
    index.load()

    query_tokens = process_text_to_tokens(search_query)

    results = []
    doc_ids_seen = set()
    for token in query_tokens:
        # list of doc ids in which current token exists
        doc_ids = index.get_documents(token)
        for doc_id in doc_ids:
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)
                movie_doc = index.docmap[doc_id]
                results.append(movie_doc)
                if len(results) == 5:
                    return results
    return results

def search_command(search_query: str) -> None:
    movie_docs = search(search_query=search_query)

    print(f"Keyword Search for '{search_query}'...")
    for i, doc in enumerate(movie_docs, 1):
        print(f"{i}. Movie ID: {doc['id']}, Movie Title: {doc['title']}")


def tf(doc_id: int, search_term: str) -> int:
    index = InvertedIndex()
    index.load()

    return index.get_tf(doc_id=doc_id, term=search_term)

def tf_command(doc_id: int, search_term: str) -> None:
    result_tf = tf(doc_id=doc_id, search_term=search_term)
    print(f"Searching for '{search_term}' in document {doc_id}...")
    print(f"\nResult:")
    print(f"Term: {search_term}, Document: {doc_id}, Term Frequency: {result_tf}")

def idf(term: str) -> float:
    index = InvertedIndex()
    index.load()

    doc_count = len(index.term_frequencies)

    term_tokens = process_text_to_tokens(term)
    if not term_tokens:
        return 0.0
    
    if len(term_tokens) != 1:
        raise ValueError(f"Given term {term} has too many tokens, want one")
    
    term_doc_count = 0.0
    for term_counter in index.term_frequencies.values():
        if term_tokens[0] in term_counter:
            term_doc_count += 1
    
    idf_score = math.log((doc_count + 1) / (term_doc_count + 1))
    return idf_score


def idf_command(term: str) -> None:
    idf_score = idf(term=term)
    print(f"Calculating IDF Score of {term}...")
    print(f"Inverse Document Frequency of {term}: {idf_score:.2f}")

def tfidf(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()

    tf_score = tf(doc_id=doc_id, search_term=term)
    idf_score = idf(term=term)

    return tf_score * idf_score


def tfidf_command(doc_id: int, term: str) -> None:
    tf_idf = tfidf(doc_id=doc_id, term=term)
    print(f"Finding TF-IDF score of '{term}' in document {doc_id}...")            
    print(f"TF-IDF score of '{term}' in document {doc_id}: {tf_idf:.2f}")
    

def bm25_idf_command(term: str) -> None:
    index = InvertedIndex()
    index.load()
    
    bm25idf =  index.get_bm25_idf(term=term)

    print(f"Calculating the BM25 IDF score of '{term}'...")           
    print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> None:
    index = InvertedIndex()
    index.load()

    bm25tf = index.get_bm25_tf(doc_id=doc_id, term=term, k1=k1, b=b)
    print(f"Calculating BM25 TF score of '{term}' in document {doc_id}...")
    print(f"BM25 TF score of '{term}' in document {doc_id}: {bm25tf:.2f}")

def bm25_search_command(query: str) -> None:
    index = InvertedIndex()
    index.load()

    scores = index.bm25_search(query, 5)

    print("Performing BM25 search for given query...")
    for i, score in enumerate(scores):
        doc_title = index.docmap[score[0]]["title"]
        print(f"{i+1}. ({score[0]}) {doc_title} - Score: {score[1]:.2f}")