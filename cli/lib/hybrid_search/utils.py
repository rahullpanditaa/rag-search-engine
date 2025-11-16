import os
import json
import time
from dotenv import load_dotenv
from google import genai

def llm_evaluation_prompt(query: str, results: list):
    formatted_results = [
        f"{i+1}. {doc['title']} - {doc['document'][:200]}" for i, doc in enumerate(results)
    ]
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    return prompt

load_dotenv()
try:
    api_key = os.environ.get("GEMINI_API_KEY")
except Exception as e:
    print(f"Unable to retrieve GEMINI_API_KEY from .env file")
    print(f"{e}")

client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

# API call to evaluate the final results after re-ranking
def generate_response_evaluate_results(prompt: str, max_retries: int=5) -> list[int]:
    time_delay = 5.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return json.loads(response.text, parse_int=lambda s: int(s))
        except Exception as e:
            if attempt == max_retries - 1:
                return []
            time.sleep(time_delay)
            time_delay *= 2