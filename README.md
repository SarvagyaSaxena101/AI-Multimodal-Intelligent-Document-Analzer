# 📄 Document AI Chatbot

A Streamlit-based application that allows users to upload documents (PDFs or images), extract text using OCR, build a context-aware vector store from the extracted information, and then chat with an AI model to get contextually relevant answers. This project implements a Retrieval-Augmented Generation (RAG) pipeline to enhance the AI's responses.

## ✨ Features

*   **Document Upload**: Easily upload PDF documents or various image formats (PNG, JPG, JPEG, TIFF).
*   **Intelligent Text Extraction**:
    *   **PDF Processing**: Extracts text directly from PDFs and performs OCR on embedded images within PDFs.
    *   **Image OCR**: Utilizes `EasyOCR` to accurately extract text from uploaded image files.
*   **Contextual Understanding**:
    *   **Text Chunking**: Splits extracted document text into manageable chunks.
    *   **Vector Embeddings**: Converts text chunks into high-dimensional vectors using `Sentence-Transformers`.
    *   **FAISS Vector Store**: Efficiently indexes and stores document embeddings for rapid similarity search.
*   **Retrieval-Augmented Generation (RAG)**:
    *   Retrieves the most relevant document chunks based on user queries.
    *   Passes these chunks as context to the AI model for highly accurate and context-aware responses.
*   **Interactive Chat Interface**:
    *   Clean and intuitive Streamlit UI.
    *   Real-time chat with AI model (powered by OpenRouter API).
    *   Dynamic display of chat history.
*   **Configurable AI Models**: Select from various OpenRouter models directly from the sidebar.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.9+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/AI-Multimodal-Project.git # Replace with your repo URL
    cd AI-Multimodal-Project
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenRouter API Key**:
    Obtain an `OPENROUTER_API_KEY` from [OpenRouter](https://openrouter.ai/).
    Create a `.env` file in the root directory of the project and add your API key:
    ```
    OPENROUTER_API_KEY="sk-YOUR_OPENROUTER_API_KEY"
    ```
    Alternatively, you can set it as an environment variable in your system.

## 💡 Usage

1.  **Run the Streamlit application**:
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your web browser.

2.  **Upload a Document**:
    *   In the main content area, use the "Upload a file" widget to upload a PDF or an image (PNG, JPG, JPEG, TIFF).
    *   Observe as the application processes the document, performs OCR (if applicable), and displays the "Extracted text".
    *   A success message will confirm that the text has been chunked, embedded, and indexed into the vector store.

3.  **Configure AI Model (Sidebar)**:
    *   Open the sidebar on the left to find the "Configuration" section.
    *   Ensure your `OPENROUTER_API_KEY` is loaded (its status will be displayed).
    *   Select your preferred AI model from the dropdown list.

4.  **Chat with the Document**:
    *   Enter your questions related to the uploaded document in the chat input box at the bottom.
    *   The AI model will use the context from your document to provide relevant answers. The chat history will be displayed above the input.

## 📺 Demo

*(Will add a GIF or screenshot here soon!)*

## 🌐 Deployed Project

*(Link to live deployment will be added here.)*

## 🛣️ Future Enhancements

*   **Improved PDF Parsing**: Integrate more robust PDF parsing libraries or advanced layout analysis tools like LayoutLM/Donut for better understanding of document structure.
*   **Field Extraction Pipelines**: Develop specific pipelines for extracting structured information (e.g., invoices, forms).
*   **Confidence Scoring**: Implement mechanisms to provide confidence scores for extracted information and AI-generated answers.
*   **Active Learning Loops**: Introduce active learning to improve OCR and extraction models over time with user feedback.
*   **Support for more Document Types**: Extend functionality to include other document formats (e.g., DOCX, XLSX).
*   **Multi-Document Query**: Allow querying across multiple uploaded documents simultaneously.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contact

For any questions or suggestions, please open an issue in this repository.