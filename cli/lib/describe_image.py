import os
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.0-flash-001"

system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""


def describe_image(path: str, query: str):
    image_path = Path(path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Given image {path} not found or does not exist")
    
    img_mimetype = mimetypes.guess_type(str(image_path))
    mime = img_mimetype[0] or "image/jpeg"

    with open(image_path, "rb") as f:
        img_data = f.read()

    response = send_request_to_llm(img=img_data, mime=mime, query=query)
    response_text = response["response_text"]
    response_usage_metadata = response["response_usage_metadata"]

    print(f"Rewritten query: {response_text.strip()}")
    if response_usage_metadata is not None:
        print(f"Total tokens:    {response_usage_metadata.total_token_count}")

def send_request_to_llm(img: bytes, mime: str, query: str):
    message = [
        system_prompt, 
        genai.types.Part.from_bytes(data=img, mime_type=mime),
        query.strip()
    ]

    response = client.models.generate_content(
        model=model,
        contents=message
    )
    return {"response_text": response.text,
            "response_usage_metadata": response.usage_metadata}