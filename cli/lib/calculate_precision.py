import json
from .constants import GOLDEN_DATASET_FILE_PATH
from .hybrid_search import HybridSearch
from .utils import get_movie_data_from_file


def calculate_evaluation_scores(k: int=5):
    with open(GOLDEN_DATASET_FILE_PATH, "r") as f:
        dataset = json.load(f)

    # list of {query, relevant_docs list}
    test_cases = dataset["test_cases"]

    movies = get_movie_data_from_file()
    searcher = HybridSearch(documents=movies)  
    
    results = []
    for test_case in test_cases:
        query = test_case["query"]
        retrieved_docs = searcher.rrf_search(query=query, k=60, limit=k)
        precision_score = _calculate_precision_score(
            retrieved=retrieved_docs, 
            relevant=test_case["relevant_docs"])
        recall_score = _calculate_recall(
            retrieved=retrieved_docs,
            relevant=test_case["relevant_docs"])
        results.append({
            "query": query,
            "precision": precision_score,
            "recall": recall_score,
            "retrieved": ", ".join([doc["title"] for doc in retrieved_docs]),
            "relevant": ", ".join([title for title in test_case["relevant_docs"]])

        })
    return results

def evaluation_command(k: int=5):
    results = calculate_evaluation_scores(k=k)
    print(f"k={k}\n")

    for result in results:
        print(f"- Query: {result['query']}")
        print(f"  - Precision@{k}: {result['precision']:.4f}")
        print(f"  - Recall@{k}: {result['recall']:.4f}")
        print(f"  - Retrieved: {result['retrieved']}")
        print(f"  - Relevant: {result['relevant']}")
        print()


# precision = relevant docs retrieved / total retrieved
# measures quality of search results
def _calculate_precision_score(retrieved: list, relevant: list) -> float:
    relevant_retrieved = 0
    for doc in retrieved:
        if doc["title"] in relevant:
            relevant_retrieved += 1
    
    score = relevant_retrieved / len(retrieved) if retrieved else 0.0
    return score

# relevant docs retrieved / total relevant
def _calculate_recall(retrieved: list, relevant: list) -> float:
    relevant_docs_retrieved = 0
    for doc in retrieved:
        if doc["title"] in relevant:
            relevant_docs_retrieved += 1
    
    score = relevant_docs_retrieved / len(relevant) if relevant else 0.0
    return score