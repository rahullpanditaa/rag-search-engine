import math
from cli.lib.inverted_index import InvertedIndex
from cli.lib.utils import process_text_to_tokens

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
        
def search_command(search_query: str) -> list[dict]:
    index = InvertedIndex()
    index.load()

    query_tokens = process_text_to_tokens(search_query)

    results, doc_ids_seen = [], set()
    for token in query_tokens:
        doc_ids = index.get_documents(token)
        for doc_id in doc_ids:
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)
                movie_doc = index.docmap[doc_id]
                results.append(movie_doc)
                if len(results) == 5:
                    return results
                
    return results

def tf_command(doc_id: int, search_term: str) -> int:
    index = InvertedIndex()
    index.load()

    return index.get_tf(doc_id=doc_id, term=search_term)

def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()

    # total docs?
    doc_count = len(index.term_frequencies)

    # term appears in how many docs?
    term_doc_count = 0
    for _, term_count in index.term_frequencies.items():
        if process_text_to_tokens(term)[0] in term_count:
            term_doc_count += 1

    idf = math.log((doc_count + 1) / (term_doc_count + 1))
    return idf

def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()

    tf = tf_command(doc_id=doc_id, search_term=term)
    idf = idf_command(term=term)

    return tf * idf

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    
    return index.get_bm25_idf(term=term)

def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    index = InvertedIndex()
    index.load()

    return index.get_bm25_tf(doc_id=doc_id, term=term, k1=k1, b=b)

def bm25_search_command(query: str):
    index = InvertedIndex()
    index.load()

    scores = index.bm25_search(query, 5)

    for i, score in enumerate(scores):
        doc_title = index.docmap[score[0]]["title"]
        print(f"{i+1}. ({score[0]}) {doc_title} - Score: {score[1]:.2f}")