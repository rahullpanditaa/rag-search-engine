import math
from inverted_index import InvertedIndex, tf_command
from helpers import process_text_to_tokens

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

def bm25_tf_command(doc_id: int, term: str, k1: float=None) -> float:
    index = InvertedIndex()
    index.load()

    return index.get_bm25_tf(doc_id=doc_id, term=term, k1=k1)