import os
import time
from .utils import get_movie_data_from_file
from .hybrid_search import HybridSearch
from dotenv import load_dotenv
from google import genai
from .prompts import rag_response_prompt

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

def rag(query: str) -> tuple[str, list[dict]]:
    movies = get_movie_data_from_file()
    searcher = HybridSearch(documents=movies)
    docs = searcher.rrf_search(query=query, limit=5)
    prompt = rag_response_prompt(query=query, docs=docs)
    rag_response = generate_response_to_query(prompt)
    return rag_response, docs

def rag_command(query : str) -> None:
    rag_response, result_docs = rag(query=query)
    
    print("Search Results:")
    for doc in result_docs:
        print(f"  - {doc['title']}")
    
    print("\nRAG Response:")
    print(f"{rag_response if rag_response else 'Unable to generate response from LLM'}")


def generate_response_to_query(prompt: str, max_retries: int=5) -> str:
    time_delay = 1.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text.strip().strip('"').strip("'").strip()
        except Exception as e:
            if attempt == max_retries - 1:
                return ""
            time.sleep(time_delay)
            time_delay *= 2
