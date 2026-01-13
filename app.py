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

# load root .env
load_dotenv()

# Initialize AI components
store = VectorStore()
chat_mgr = ChatManager(store)

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

    if st.button("Index text"):
        with st.spinner("Embedding and indexing..."):
            vec = embed_texts([text])[0]
            doc_id = str(uuid.uuid4())
            store.add(doc_id, vec, {"text": text, "meta": uploaded.name})
            st.success(f"Indexed as {doc_id}")

st.header("Chat with document-aware model")
model = st.selectbox("Model (OpenRouter model name)", ["openrouter/default"])
message = st.text_input("Message")
if st.button("Send") and message:
    with st.spinner("Thinking..."):
        reply = chat_mgr.handle_message(st.session_state.session_id, message, model_name=model)
    st.write(reply)
