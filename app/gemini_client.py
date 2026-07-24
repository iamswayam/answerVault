from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

CHAT_MODEL = "gemini-3.1-flash-lite"
EMBED_MODEL = "gemini-embedding-001"

INTERVIEW_PREP_INSTRUCTION = (
    "You are helping someone prepare short, memorable answers for technical "
    "interviews. Answer in plain, simple English. Keep answers to 2-4 short "
    "sentences unless the user explicitly asks for code -- in that case a "
    "short code block is fine, but keep any explanation brief. Do not add "
    "unsolicited follow-up questions or 'if they ask X' additions. Use the "
    "conversation history to understand follow-up questions such as 'now "
    "show me one that does X' in context."
)


def generate_reply(history: list[dict]) -> str:
    """Used by /chat -- general purpose conversation, no special style
    instruction applied."""
    contents = [
        types.Content(role=h["role"], parts=[types.Part.from_text(text=h["parts"])])
        for h in history
    ]
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
    )
    return response.text


def generate_free_answer(history: list[dict]) -> str:
    """Used by the LOW confidence tier and /ask/general. Takes full
    conversation history (last item = the new message) so follow-up
    questions resolve correctly, and applies the interview-prep style
    instruction so answers stay short and plain."""
    contents = [
        types.Content(role=h["role"], parts=[types.Part.from_text(text=h["parts"])])
        for h in history
    ]
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=INTERVIEW_PREP_INSTRUCTION
        ),
    )
    return response.text


def rewrite_query_with_history(history: list[dict], query_text: str) -> str:
    """Turns an ambiguous follow-up ("I asked difference?") into a
    standalone question using prior conversation turns, so vector search
    has something meaningful to embed. If history is empty, returns the
    query unchanged -- nothing to rewrite against."""
    if not history:
        return query_text

    context = "\n".join(f"{h['role']}: {h['parts']}" for h in history[-6:])
    prompt = (
        "Given this conversation history and a follow-up message, rewrite "
        "the follow-up into a standalone question that includes whatever "
        "context it is missing. If it is already standalone, return it "
        "unchanged. Output ONLY the rewritten question, nothing else, no "
        "quotes, no explanation.\n\n"
        f"History:\n{context}\n\nFollow-up: {query_text}"
    )
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
    )
    return response.text.strip()


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