# ==========================================================
# SCHOLARFLOW AI
# PDF Study Assistant using RAG
#
# Technologies:
# Streamlit + Groq + FAISS + Sentence Transformers + PyMuPDF
#
# Main Flow:
# PDF Upload
#      ↓
# PDF Text Extraction
#      ↓
# Text Chunking
#      ↓
# Sentence Transformer Embeddings
#      ↓
# FAISS Semantic Search
#      ↓
# Groq LLM
#      ↓
# Answer from Uploaded PDF
# ==========================================================


# ==========================================================
# 1. IMPORT LIBRARIES
# ==========================================================

import streamlit as st
import fitz
import numpy as np
import faiss
import hashlib
import re

from groq import Groq
from sentence_transformers import SentenceTransformer


# ==========================================================
# 2. PAGE SETTINGS
# ==========================================================

st.set_page_config(
    page_title="ScholarFlow AI",
    page_icon="📚",
    layout="wide"
)


# ==========================================================
# 3. SESSION STATE
# ==========================================================

# Store information between Streamlit reruns.

if "processed_hash" not in st.session_state:
    st.session_state.processed_hash = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================================
# 4. EMBEDDING MODEL
# ==========================================================

@st.cache_resource
def load_embedding_model():

    # Sentence Transformer converts text into numerical
    # vectors so FAISS can search for similar content.

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


# ==========================================================
# 5. GROQ CLIENT
# ==========================================================

def get_groq_client():

    # Read the Groq API key from Streamlit secrets.

    try:

        api_key = st.secrets["GROQ_API_KEY"]

    except Exception:

        api_key = ""

    if not api_key:

        return None

    return Groq(api_key=api_key)


# ==========================================================
# 6. PDF VALIDATION
# ==========================================================

def validate_pdf(uploaded_file):

    # Read the uploaded PDF as bytes.

    pdf_bytes = uploaded_file.getvalue()


    # ------------------------------------------------------
    # Check file extension
    # ------------------------------------------------------

    if not uploaded_file.name.lower().endswith(".pdf"):

        return False, "Please upload a PDF file only."


    # ------------------------------------------------------
    # Check PDF file signature
    # ------------------------------------------------------

    if not pdf_bytes.startswith(b"%PDF-"):

        return False, "This is not a valid PDF file."


    # ------------------------------------------------------
    # Maximum file size = 15 MB
    # ------------------------------------------------------

    if len(pdf_bytes) > 15 * 1024 * 1024:

        return False, "PDF size must be below 15 MB."


    try:

        # Open PDF using PyMuPDF.

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )


        # --------------------------------------------------
        # Password protected PDF
        # --------------------------------------------------

        if document.needs_pass:

            document.close()

            return False, (
                "Password-protected PDFs are not supported."
            )


        # --------------------------------------------------
        # Empty PDF
        # --------------------------------------------------

        if len(document) == 0:

            document.close()

            return False, "The PDF is empty."


        # --------------------------------------------------
        # Maximum 150 pages
        # --------------------------------------------------

        if len(document) > 150:

            document.close()

            return False, (
                "Please upload a PDF with 150 pages or fewer."
            )


        total_characters = 0


        # --------------------------------------------------
        # Check every page for readable text
        # --------------------------------------------------

        for page_number, page in enumerate(document):

            text = page.get_text("text").strip()

            character_count = len(
                re.sub(r"\s+", "", text)
            )

            total_characters += character_count


            # If a page contains almost no text,
            # it may be scanned/image-only.

            if character_count < 40:

                document.close()

                return False, (
                    f"Page {page_number + 1} has very little "
                    "readable text. Scanned or image-only "
                    "PDFs are not supported."
                )


        # --------------------------------------------------
        # Check total document text
        # --------------------------------------------------

        if total_characters < 100:

            document.close()

            return False, (
                "No meaningful text was found in this PDF."
            )


        document.close()

        return True, "PDF is valid."


    except Exception as error:

        return False, (
            f"Unable to read this PDF: {error}"
        )


# ==========================================================
# 7. EXTRACT TEXT FROM PDF
# ==========================================================

def extract_pdf_text(uploaded_file):

    # Open the PDF.

    document = fitz.open(
        stream=uploaded_file.getvalue(),
        filetype="pdf"
    )

    pages = []


    # Extract text page by page.

    for page_number, page in enumerate(document):

        text = page.get_text("text").strip()

        if text:

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text
                }
            )


    document.close()

    return pages


# ==========================================================
# 8. CREATE TEXT CHUNKS
# ==========================================================

def create_chunks(pages):

    chunks = []

    # Number of words inside one chunk.

    chunk_size = 180

    # Some words are repeated between chunks.
    # This helps preserve context.

    overlap = 35


    for page in pages:

        words = page["text"].split()

        start = 0


        while start < len(words):

            chunk_words = words[
                start:start + chunk_size
            ]

            chunk_text = " ".join(chunk_words)


            if len(chunk_text.strip()) > 20:

                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"]
                    }
                )


            start += chunk_size - overlap


    return chunks


# ==========================================================
# 9. CREATE FAISS VECTOR DATABASE
# ==========================================================

def create_vector_database(chunks, model):

    # Extract chunk text.

    texts = []

    for chunk in chunks:

        texts.append(
            chunk["text"]
        )


    # Convert text into embeddings.

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )


    # FAISS works efficiently with float32.

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )


    # Get embedding dimension.

    dimension = embeddings.shape[1]


    # Create FAISS similarity index.

    index = faiss.IndexFlatIP(
        dimension
    )


    # Add embeddings to FAISS.

    index.add(
        embeddings
    )


    return index


# ==========================================================
# 10. SEARCH THE DOCUMENT
# ==========================================================

def search_document(
    question,
    chunks,
    index,
    model
):

    if not chunks:

        return []


    if index is None:

        return []


    # Convert the user's question into an embedding.

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )


    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )


    # Retrieve maximum 4 relevant chunks.

    number_of_results = min(
        4,
        len(chunks)
    )


    # Search FAISS.

    scores, indexes = index.search(
        question_embedding,
        number_of_results
    )


    results = []


    for score, chunk_index in zip(
        scores[0],
        indexes[0]
    ):

        if chunk_index < 0:

            continue


        result = chunks[
            chunk_index
        ].copy()


        result["score"] = float(
            score
        )


        results.append(
            result
        )


    return results


# ==========================================================
# 11. GENERATE ANSWER USING GROQ
# ==========================================================

def generate_answer(
    question,
    search_results,
    client
):

    # If nothing relevant was found,
    # do not ask the LLM to guess.

    if not search_results:

        return (
            "I couldn't find this information "
            "in your uploaded PDF."
        )


    # ------------------------------------------------------
    # Prepare retrieved document context
    # ------------------------------------------------------

    context_parts = []


    for result in search_results:

        context_parts.append(
            "[Page "
            + str(result["page"])
            + "]\n"
            + result["text"]
        )


    context = "\n\n".join(
        context_parts
    )


    # ------------------------------------------------------
    # System instructions
    # ------------------------------------------------------

    system_prompt = """
You are ScholarFlow AI, a PDF study assistant.

Your job is to answer the user's question ONLY
using the document context provided to you.

IMPORTANT RULES:

1. Use ONLY the supplied document context.
2. Do NOT use outside knowledge.
3. Do NOT invent or guess information.
4. If the answer is not present in the document,
   clearly say that it was not found in the uploaded PDF.
5. Explain the answer in simple student-friendly language.
6. Use short paragraphs and bullet points when helpful.
7. Mention the relevant page number when possible.
"""


    # ------------------------------------------------------
    # User prompt
    # ------------------------------------------------------

    user_prompt = (
        "DOCUMENT CONTEXT:\n\n"
        + context
        + "\n\n"
        "USER QUESTION:\n"
        + question
        + "\n\n"
        "Answer ONLY from the document context."
    )


    # ------------------------------------------------------
    # Call Groq
    #
    # GPT-OSS 120B is a current supported Groq model.
    # ------------------------------------------------------

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.2,

        max_tokens=1000
    )


    # Return generated answer.

    return response.choices[0].message.content


# ==========================================================
# 12. HEADER / HERO SECTION
# ==========================================================

st.title("📚 ScholarFlow AI")

st.subheader(
    "Your PDF. Your Questions. Your Study Assistant."
)

st.write(
    "Upload your study material and ask questions directly "
    "from your PDF. ScholarFlow searches your document first "
    "so you can study without manually searching through pages."
)


# ==========================================================
# 13. SIMPLE FEATURE CARDS
# ==========================================================

st.divider()

st.markdown(
    "### ✨ Study smarter with your own notes"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.info(
        "📄 **Upload Your PDF**\n\n"
        "Use your university notes, lectures, "
        "assignments or study material."
    )


with col2:

    st.info(
        "🔎 **Find Relevant Content**\n\n"
        "FAISS searches the document and finds "
        "the most relevant sections."
    )


with col3:

    st.info(
        "💬 **Ask & Understand**\n\n"
        "Ask questions in natural language and "
        "get simple answers from your PDF."
    )


# ==========================================================
# 14. LOAD EMBEDDING MODEL
# ==========================================================

with st.spinner(
    "Preparing your study assistant..."
):

    embedding_model = load_embedding_model()


# ==========================================================
# 15. PDF UPLOAD SECTION
# ==========================================================

st.divider()

st.header(
    "📄 Upload your study material"
)

st.write(
    "Start by uploading a text-based PDF. "
    "Scanned or image-only PDFs are not supported."
)


uploaded_file = st.file_uploader(

    "Choose a PDF file",

    type=["pdf"],

    accept_multiple_files=False
)


st.caption(
    "PDF only  •  Maximum 15 MB  •  Maximum 150 pages"
)


# ==========================================================
# 16. PROCESS PDF
# ==========================================================

if uploaded_file is not None:

    current_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


    # Only process when this is a new PDF.

    if (
        current_hash
        != st.session_state.processed_hash
    ):

        st.info(
            "Your PDF is selected. "
            "Click the button below to prepare it for questions."
        )


        process_button = st.button(
            "✨ Prepare My Study Space",
            use_container_width=True
        )


        if process_button:

            # First validate the PDF.

            is_valid, message = validate_pdf(
                uploaded_file
            )


            if not is_valid:

                st.error(
                    message
                )


            else:

                try:

                    with st.spinner(
                        "Reading your PDF and preparing smart search..."
                    ):

                        # Extract pages.

                        pages = extract_pdf_text(
                            uploaded_file
                        )


                        # Create chunks.

                        chunks = create_chunks(
                            pages
                        )


                        if not chunks:

                            st.error(
                                "No useful text was found "
                                "in this PDF."
                            )

                            st.stop()


                        # Create FAISS index.

                        vector_index = (
                            create_vector_database(
                                chunks,
                                embedding_model
                            )
                        )


                        # Save everything in session state.

                        st.session_state.chunks = (
                            chunks
                        )

                        st.session_state.vector_index = (
                            vector_index
                        )

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.session_state.processed_hash = (
                            current_hash
                        )

                        # Clear previous conversation.

                        st.session_state.messages = []


                    st.success(
                        "🎉 Your PDF is ready for questions!"
                    )


                    st.rerun()


                except Exception as error:

                    st.error(
                        "Document processing failed."
                    )

                    st.exception(error)


# ==========================================================
# 17. CHECK IF DOCUMENT IS READY
# ==========================================================

document_ready = (

    uploaded_file is not None

    and

    st.session_state.processed_hash is not None

    and

    hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()
    == st.session_state.processed_hash

    and

    st.session_state.vector_index is not None
)


# ==========================================================
# 18. STUDY SPACE
# ==========================================================

if document_ready:

    st.divider()


    # ------------------------------------------------------
    # Ready message
    # ------------------------------------------------------

    st.success(
        "🟢 Your Study Space is Ready"
    )


    st.header(
        "💬 Ask Your PDF Anything"
    )


    st.write(
        "Have a question about your notes? "
        "Ask it below and ScholarFlow will search your "
        "uploaded PDF before preparing the answer."
    )


    # ------------------------------------------------------
    # Document information
    # ------------------------------------------------------

    col1, col2 = st.columns([3, 1])


    with col1:

        st.info(
            "📄 **Current document:** "
            + st.session_state.document_name
        )


    with col2:

        st.metric(
            "Text Chunks",
            len(
                st.session_state.chunks
            )
        )


    # ======================================================
    # 19. SUGGESTED QUESTIONS
    # ======================================================

    if not st.session_state.messages:

        st.markdown(
            "#### 💡 Not sure what to ask?"
        )

        st.caption(
            "Try one of these questions to explore your PDF:"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            if st.button(
                "📌 Summarize this document",
                use_container_width=True
            ):

                st.session_state.selected_question = (
                    "Summarize this document"
                )

                st.rerun()


        with col2:

            if st.button(
                "🧠 Explain the main concepts",
                use_container_width=True
            ):

                st.session_state.selected_question = (
                    "Explain the main concepts"
                )

                st.rerun()


        with col3:

            if st.button(
                "⭐ What are the key points?",
                use_container_width=True
            ):

                st.session_state.selected_question = (
                    "What are the key points?"
                )

                st.rerun()


    # ======================================================
    # 20. DISPLAY CHAT HISTORY
    # ======================================================

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


            # Show sources for assistant answers.

            if (
                message["role"] == "assistant"

                and

                message.get("sources")
            ):

                with st.expander(
                    "📖 View source pages"
                ):

                    for source in message["sources"]:

                        st.markdown(
                            "**Page "
                            + str(source["page"])
                            + "**"
                        )

                        st.write(
                            source["text"]
                        )


    # ======================================================
    # 21. QUESTION INPUT
    # ======================================================

    question = st.chat_input(
        "Ask a question about your PDF..."
    )


    # If user selected a suggested question.

    if "selected_question" in st.session_state:

        question = (
            st.session_state.selected_question
        )

        del st.session_state.selected_question


    # ======================================================
    # 22. PROCESS QUESTION
    # ======================================================

    if question:

        # Get Groq client.

        client = get_groq_client()


        # --------------------------------------------------
        # Check API key
        # --------------------------------------------------

        if client is None:

            st.error(
                "Groq API key is missing."
            )

            st.info(
                "Create this file:\n\n"
                ".streamlit/secrets.toml\n\n"
                "and add:\n\n"
                'GROQ_API_KEY = "your_api_key_here"'
            )

            st.stop()


        # --------------------------------------------------
        # Save user question
        # --------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # --------------------------------------------------
        # Display user question
        # --------------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                question
            )


        # --------------------------------------------------
        # Generate AI answer
        # --------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching your PDF..."
            ):

                try:

                    # Search the document first.

                    search_results = search_document(

                        question,

                        st.session_state.chunks,

                        st.session_state.vector_index,

                        embedding_model

                    )


                    # Generate answer using retrieved context.

                    answer = generate_answer(

                        question,

                        search_results,

                        client

                    )


                    # Display answer.

                    st.markdown(
                        answer
                    )


                    # --------------------------------------------------
                    # Source pages
                    # --------------------------------------------------

                    if search_results:

                        with st.expander(
                            "📖 See where this answer came from"
                        ):

                            for source in search_results:

                                st.markdown(
                                    "**Page "
                                    + str(source["page"])
                                    + "**"
                                )

                                st.write(
                                    source["text"]
                                )


                    # --------------------------------------------------
                    # Save assistant response
                    # --------------------------------------------------

                    st.session_state.messages.append(

                        {
                            "role": "assistant",

                            "content": answer,

                            "sources": search_results
                        }

                    )


                except Exception as error:

                    st.error(
                        "Unable to generate the answer."
                    )

                    st.exception(error)


# ==========================================================
# 23. FOOTER
# ==========================================================

st.divider()

st.caption(
    "📚 ScholarFlow AI  •  "
    "Built with Python, Streamlit, FAISS, "
    "Sentence Transformers & Groq"
)