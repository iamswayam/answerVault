import google.generativeai as genai
from app.config import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-3.1-flash-lite")


def generate_reply(history: list[dict]) -> str:
    """
    history: list of {"role": "user"|"model", "parts": "text"}
    The LAST item in history is the new message being sent.
    Everything before it is prior context.
    """
    chat = model.start_chat(history=history[:-1])
    response = chat.send_message(history[-1]["parts"])
    return response.text
