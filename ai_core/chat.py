import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import threading
import httpx # Add this line


class ChatManager:
    def __init__(self, vector_store=None, openrouter_api_key=None, embed_texts_func=None):
        self.histories = {}  # session_id -> list of messages
        self.lock = threading.Lock()
        self.vector_store = vector_store
        self.openrouter_api_key = openrouter_api_key
        self.embed_texts_func = embed_texts_func # Store embedding function

    def _append(self, session_id, role, text):
        with self.lock:
            if session_id not in self.histories:
                self.histories[session_id] = []
            self.histories[session_id].append({"role": role, "text": text})

    def _build_prompt_with_rag(self, session_id: str, current_user_message: str, context: str) -> list:
        system_instruction = "You are a helpful assistant. Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer."
        if context:
            system_instruction += f"\n\nContext:\n{context}"

        messages_for_llm = [SystemMessage(content=system_instruction)]
        
        # Append historical messages
        history = self.histories.get(session_id, [])
        for msg in history:
            if msg["role"] == "user":
                messages_for_llm.append(HumanMessage(content=msg["text"]))
            elif msg["role"] == "assistant":
                messages_for_llm.append(AIMessage(content=msg["text"]))
        
        # Add the current user message as the final message
        messages_for_llm.append(HumanMessage(content=current_user_message))
        
        return messages_for_llm


    def handle_message(self, session_id: str, message: str, model_name: str = "openrouter/default"):
        
        # 1. Embed the user's message for retrieval (if vector_store and embed_texts_func are available)
        context = ""
        if self.vector_store and self.embed_texts_func:
            query_embedding = self.embed_texts_func([message])[0]
            retrieved_docs = self.vector_store.query(query_embedding, top_k=3) # top_k can be adjusted
            if retrieved_docs:
                context = "\n".join([doc["text"] for doc in retrieved_docs])
                # Simple deduplication - convert to set and back to list/string
                context = "\n".join(list(set(context.split("\n"))))
        
        # Build the full prompt with RAG context and chat history
        messages_for_llm = self._build_prompt_with_rag(session_id, message, context)
        
        # Instantiate ChatOpenAI with the selected model_name
        llm_instance = ChatOpenAI(
            model=model_name, # Pass the model name here
            openai_api_key=self.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            http_client=httpx.Client()
        )
        
        try:
            response = llm_instance.invoke(messages_for_llm) # Use the contextualized messages
            reply = response.content
        except Exception as e:
            reply = f"Error calling OpenRouter: {e}"
        
        self._append(session_id, "user", message) # Append user message *after* retrieval and before storing assistant reply
        self._append(session_id, "assistant", reply) # Store assistant reply
        return reply
