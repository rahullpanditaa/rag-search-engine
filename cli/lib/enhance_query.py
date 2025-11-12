import os
from google import genai
from dotenv import load_dotenv


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

def query_spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    corrected_query = (response.text or "").strip().strip('"').strip("'")
    return corrected_query if corrected_query else query

def query_rewrite(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

Original: "{query}"

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep it concise (under 10 words)
- It should be a google style search query that's very specific
- Don't use boolean logic

Examples:

- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

Rewritten query:"""
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    corrected_query = response.text or ""
    corrected_query = corrected_query.strip().strip('"').strip("'")
    return corrected_query if corrected_query else query



def enhance_query(query: str, method: str) -> str:
    match method:
        case "spell":
            return query_spell_correct(query=query)
        case "rewrite":
            return query_rewrite(query=query)
        case _:
            return query







    