import streamlit as st
import uuid
import os
from dotenv import load_dotenv
from ai_core.ocr import ocr_from_image
from ai_core.embeddings import embed_texts
from ai_core.vector_store import VectorStore
from ai_core.chat import ChatManager
import fitz  # PyMuPDF
import io
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter # Added for chunking

# load root .env
load_dotenv()

# Retrieve API key after loading .env
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize AI components
store = VectorStore()
# Pass embed_texts function to ChatManager for RAG
chat_mgr = ChatManager(store, openrouter_api_key=openrouter_api_key, embed_texts_func=embed_texts)

# Initialize text splitter for RAG
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def process_pdf(file_contents):
    """Extracts text and images from a PDF, performs OCR on images, and returns combined text."""
    doc = fitz.open(stream=file_contents, filetype="pdf")
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        all_text += page.get_text()
        
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            try:
                image = Image.open(io.BytesIO(image_bytes))
                # ocr_from_image expects bytes, so we'll re-save it to a buffer
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()
                all_text += ocr_from_image(img_bytes)
            except Exception as e:
                st.error(f"Error processing image in PDF: {e}")
    return all_text

st.title("Document AI — Streamlit UI")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.header("Upload Document (image or PDF)")
uploaded = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg", "tiff", "pdf"], accept_multiple_files=False)
if uploaded:
    contents = uploaded.getvalue()
    text = ""
    if uploaded.type == "application/pdf":
        with st.spinner("Processing PDF..."):
            text = process_pdf(contents)
    else:
        with st.spinner("Performing OCR..."):
            text = ocr_from_image(contents)
    
    st.subheader("Extracted text")
    st.text_area("OCR and text extraction result", value=text, height=300)

    with st.spinner("Splitting text into chunks, embedding, and indexing..."):
        chunks = text_splitter.split_text(text)
        indexed_count = 0
        for i, chunk in enumerate(chunks):
            vec = embed_texts([chunk])[0]
            # Associate metadata with each chunk if needed, e.g., original document name and chunk index
            store.add(f"{st.session_state.session_id}_doc_{uploaded.name}_chunk_{i}", vec, {"text": chunk, "meta": uploaded.name, "chunk_id": i})
            indexed_count += 1
        st.success(f"Indexed {indexed_count} chunks automatically from {uploaded.name}!")

st.header("Chat with document-aware model")

# Display API key status
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    st.error("OPENROUTER_API_KEY environment variable not set! Please add it to your .env file or environment.")
else:
    st.success("OPENROUTER_API_KEY loaded (masked): " + openrouter_api_key[:4] + "..." + openrouter_api_key[-4:])

model = st.selectbox(
    "Model (OpenRouter model name)",
    [
        "qwen/qwen3-coder:free", # User requested model
        "openrouter/auto", # General default, often picks a good free model
        "mistralai/mistral-7b-instruct",
        "google/gemma-7b-it",
        "meta-llama/llama-3-8b-instruct",
        # Add more free models as needed, check OpenRouter for their latest free tier options
    ]
)
message = st.text_input("Message")
if st.button("Send") and message:
    with st.spinner("Thinking..."):
        reply = chat_mgr.handle_message(st.session_state.session_id, message, model_name=model)
    st.write(reply)
