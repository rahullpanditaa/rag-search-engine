import math
from inverted_index import InvertedIndex
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