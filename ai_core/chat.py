import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import threading

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


class ChatManager:
    def __init__(self, vector_store=None):
        self.histories = {}  # session_id -> list of messages
        self.lock = threading.Lock()
        self.vector_store = vector_store
        self.llm = ChatOpenAI(openai_api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1")

    def _append(self, session_id, role, text):
        with self.lock:
            if session_id not in self.histories:
                self.histories[session_id] = []
            self.histories[session_id].append({"role": role, "text": text})

    def handle_message(self, session_id: str, message: str, model_name: str = "openrouter/default"):
        self._append(session_id, "user", message)
        messages = self._build_prompt(session_id)
        try:
            self.llm.model = model_name
            response = self.llm.invoke(messages)
            reply = response.content
        except Exception as e:
            reply = f"Error calling OpenRouter: {e}"
        self._append(session_id, "assistant", reply)
        return reply

    def _build_prompt(self, session_id: str) -> list:
        history = self.histories.get(session_id, [])
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["text"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["text"]))
        return messages
