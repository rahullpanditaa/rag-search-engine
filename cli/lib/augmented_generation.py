import os
import time
from .utils import get_movie_data_from_file
from .hybrid_search import HybridSearch
from dotenv import load_dotenv
from google import genai
from .prompts import (
    rag_response_prompt, 
    rag_summarize_prompt,
    rag_citations_prompt,
    rag_questions_prompt
)

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
    rag_response if rag_response else 'Unable to generate response from LLM'


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


def summarize(query: str, limit: int = 5):
    movies = get_movie_data_from_file()
    searcher = HybridSearch(documents=movies)
    results = searcher.rrf_search(query=query, limit=limit)
    prompt = rag_summarize_prompt(query=query, docs=results)
    rag_summary = generate_response_to_query(prompt=prompt)
    return rag_summary, results

def summarize_command(query: str, limit: int=5):
    rag_summary, docs = summarize(query=query, limit=limit)

    print("Search Results:")
    for doc in docs:
        print(f"  - {doc['title']}")
    
    print("\nLLM Summary:")
    rag_summary if rag_summary else 'Unable to summarize results via LLM.'

def citations(query: str, limit: int=5):
    movies = get_movie_data_from_file()
    searcher = HybridSearch(documents=movies)
    results = searcher.rrf_search(query=query, limit=10)
    prompt = rag_citations_prompt(query=query, docs=results)
    rag_ans_with_citations = generate_response_to_query(prompt=prompt)
    return rag_ans_with_citations, results

def citations_command(query: str, limit: int=5):
    rag_ans, docs = citations(query=query, limit=limit)

    print("Search Results:")
    for doc in docs:
        print(f"  - {doc['title']}")

    print("\nLLM Answer:")
    print(rag_ans if rag_ans else 'Unable to answer user query')

def question(query: str, limit: int=5):
    movies = get_movie_data_from_file()
    searcher = HybridSearch(documents=movies)
    results = searcher.rrf_search(query=query, limit=limit)
    prompt = rag_questions_prompt(query=query, docs=results)
    rag_ans_question = generate_response_to_query(prompt=prompt)
    return rag_ans_question, results

def question_command(query: str, limit: int=5) -> None:
    rag_ans, docs = question(query=query, limit=limit)

    print("Search Results:")
    for doc in docs:
        print(f"  - {doc['title']}")
    
    print("\nAnswer:")
    print(rag_ans if rag_ans else 'Unable to answer user query')

