from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

CHAT_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"


def generate_reply(history: list[dict]) -> str:
    """
    history: list of {"role": "user"|"model", "parts": "text"}
    The LAST item is the new message; everything before it is prior context.
    """
    contents = [
        types.Content(role=h["role"], parts=[types.Part.from_text(text=h["parts"])])
        for h in history
    ]
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
    )
    return response.text


def embed_text(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="retrieval_document",
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


def embed_query(text: str) -> list[float]:
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="retrieval_query",   
            output_dimensionality=768,
        ),
    )
    return result.embeddings[0].values


def generate_free_answer(query_text: str) -> str:
    """Stateless, single-turn answer from general knowledge. Used only
    for the LOW confidence tier — no conversation history involved."""
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part.from_text(text=query_text)])
        ],
    )
    return response.text