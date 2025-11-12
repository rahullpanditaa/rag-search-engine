import time
import os
import re
import json
from dotenv import load_dotenv
from google import genai
from typing import Optional
from .prompts import re_rank_individual_docs_prompt, re_rank_batch_prompt
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

def re_rank_scores(query: str, scores: list[dict], method: Optional[str]= None) -> list[dict]:
    # results -> list of {id, title, document, bm25_rank, semantic_rank, rrf_Score}
    match method:
        case "individual":
            return re_rank_individual(query=query, scores=scores)
        case "batch":
            return re_rank_batch(query=query, scores=scores)
        case "cross_encoder":
            return re_rank_cross_encoder(query=query, scores=scores)
        case _:
            return scores

def re_rank_cross_encoder(query: str, scores: list[dict]):
    pairs = []
    for doc in scores:
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('document', '')}"])

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    
    # list of numbers, one for each pair i.e. one for
    # each [query, doc_title, doc_description]
    ce_scores = cross_encoder.predict(pairs)

    # movies_map = {doc["id"]: doc for doc in scores}
    results = []
    for i, doc in enumerate(scores):
        new_doc = doc.copy()
        new_doc["cross_encoder_score"] = float(ce_scores[i])
        results.append(new_doc)
    sorted_results = sorted(results, key=lambda d: d["cross_encoder_score"], reverse=True)
    return sorted_results


def re_rank_batch(query: str, scores: list[dict]):
    # for doc in scores:
    ranked_docs = []
    prompt = re_rank_batch_prompt(query=query, docs=scores)
    doc_ids_ranked = generate_response_batch(prompt=prompt)
    
    movie_map = {doc["id"]: doc for doc in scores}
    for doc_id in doc_ids_ranked:
        ranked_docs.append(movie_map.get(doc_id))
    
    results = []
    for i, doc in enumerate(ranked_docs, 1):
        new_doc = doc.copy()
        new_doc["rank"] = i
        results.append(new_doc)
    return results   



def generate_response_batch(prompt: str, max_retries: int=5) -> list[int]:
    time_delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            # get back a json list
            return json.loads(response.text, parse_int=lambda s: int(s))
        except Exception as e:
            if attempt == max_retries - 1:
                return []
            time.sleep(time_delay + 0.5)
            time_delay *= 2
            



def re_rank_individual(query: str, scores: list[dict]) -> list[dict]:
    # doc -> {id, title, document, bm25_rank, semantic_rank, rrf_score}
    results = []
    for doc in scores:
        doc_title = doc.get("title", "")
        doc_description = doc.get("document", "")
        prompt = re_rank_individual_docs_prompt(query=query, doc_title=doc_title, doc_description=doc_description)
        rank = generate_response_rank(prompt=prompt)
        new_doc = doc.copy()
        new_doc["re_rank_score"] = rank
        results.append(new_doc)
        time.sleep(3.0)
    sorted_results = sorted(results, key=lambda d: d["re_rank_score"], reverse=True)
    return sorted_results

def generate_response_rank(prompt: str, max_retries: int=5) -> float:
    time_delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            # response.text or ""
            return parse_llm_generated_individual_score(getattr(response, "text", "") or "")
        except Exception as e:
            if attempt == max_retries - 1:
                return 0.0
            time.sleep(time_delay + 0.5)
            time_delay *= 2
    

def parse_llm_generated_individual_score(text: str) -> float:
    m = re.search(r'(\d+(?:\.\d+)?)', text or '')
    try:
        n = float(m.group(1)) if m else 0.0
        return max(0.0, min(10.0, n))
    except:
        return 0.0