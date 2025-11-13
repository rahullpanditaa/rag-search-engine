import os
import json
import time
from dotenv import load_dotenv
from google import genai

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