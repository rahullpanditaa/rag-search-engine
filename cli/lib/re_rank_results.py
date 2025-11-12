import time
import os
import re
from dotenv import load_dotenv
from google import genai
from typing import Optional

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

def re_rank_scores(query: str, scores: list[dict], method: Optional[str]= None) -> list[dict]:
    # results -> list of {id, title, document, bm25_rank, semantic_rank, rrf_Score}
    match method:
        case "individual":
            return re_rank_individual(query=query, scores=scores)
        case _:
            return scores
        

def re_rank_individual(query: str, scores: list[dict]) -> list[dict]:
    # doc -> {id, title, document, bm25_rank, semantic_rank, rrf_score}
    results = []
    for doc in scores:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("document", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""
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
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            # response.text or ""
            return parse_llm_generated_score(getattr(response, "text", "") or "")
        except Exception as e:
            if attempt == max_retries - 1:
                return 0.0
            time.sleep(time_delay + 0.5)
            time_delay *= 2
    

def parse_llm_generated_score(text: str) -> float:
    m = re.search(r'(\d+(?:\.\d+)?)', text or '')
    try:
        n = float(m.group(1)) if m else 0.0
        return max(0.0, min(10.0, n))
    except:
        return 0.0