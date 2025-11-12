import os
from google import genai
from dotenv import load_dotenv


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing GEMINI_API_KEY in .env file")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

def query_spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.

Only correct obvious typos. Don't change correctly spelled words.

Query: "{query}"

If no errors, return the original query.
Corrected:"""
    response = generate_respone(prompt=prompt)
    enhanced_query = _clean_response(response)
    return enhanced_query if enhanced_query else query

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
    response = generate_respone(prompt=prompt)
    enhanced_query = _clean_response(response)
    return enhanced_query if enhanced_query else query

def query_expand(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
This will be appended to the original query.

Examples:

- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

Query: "{query}"
"""
    response = generate_respone(prompt=prompt)
    enhanced_query = _clean_response(response)
    return enhanced_query if enhanced_query else query

def enhance_query(query: str, method: str) -> str:
    match method:
        case "spell":
            return query_spell_correct(query=query)
        case "rewrite":
            return query_rewrite(query=query)
        case "expand":
            return query_expand(query=query)
        case _:
            return query

def _clean_response(text: str) -> str:
    return text.strip().strip('"').strip("'").strip('*').strip()

def generate_respone(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=model, contents=prompt
        )
        return response.text or ""
    except Exception as e:
        print(f"Failed to enhance query via Gemini API: {e}")
        return ""



    