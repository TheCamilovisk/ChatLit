from io import BytesIO
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader

st.set_page_config(layout="wide")

load_dotenv()


def load_document(uploaded_file: BytesIO) -> str:
    """Load and extract text from a PDF file.

    Args:
        uploaded_file (BytesIO): The uploaded PDF file.

    Returns:
        str: The extracted raw text from the PDF.
    """
    reader = PdfReader(uploaded_file)
    raw_text = "\n".join(p.extract_text() for p in reader.pages)
    return raw_text


def load_and_split_document(uploaded_file: BytesIO) -> List[str]:
    """Load and split the document into chunks.

    Args:
        uploaded_file (BytesIO): The uploaded PDF file.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    raw_text = load_document(uploaded_file)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def create_vector_db(uploaded_document: BytesIO) -> FAISS:
    """Create a vector database from the uploaded document.

    Args:
        uploaded_document (BytesIO): The uploaded PDF file.

    Returns:
        FAISS: The vector store created from the document text chunks.
    """
    embeddings = OpenAIEmbeddings()
    text_chunks = load_and_split_document(uploaded_document)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store


def create_chat_chain(vector_store: VectorStore) -> ConversationalRetrievalChain:
    """Create a conversational retrieval chain.

    Args:
        vector_store (VectorStore): The vector store for retrieving text.

    Returns:
        ConversationalRetrievalChain: The chat chain for conversation.
    """
    llm = ChatOpenAI()
    memory_buffer = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory_buffer,
    )

    return chat_chain


def handle_conversation_creation(uploaded_file: BytesIO):
    """Handle the creation of the conversation.

    Args:
        uploaded_file (BytesIO): The uploaded PDF file.
    """
    with st.spinner(f"Loading {uploaded_file.name}"):
        vector_store = create_vector_db(uploaded_file)
        chat_chain = create_chat_chain(vector_store)
        st.session_state.chat_chain = chat_chain
        st.rerun()


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def handle_file_upload():
    """Handle the file upload process."""
    with st.sidebar:
        if uploaded_file := st.file_uploader("Choose a PDF file", type="pdf"):
            if "chat_chain" not in st.session_state:
                handle_conversation_creation(uploaded_file)


def submit_question(user_prompt: str) -> str:
    """Submit the user's question to the chat chain.

    Args:
        user_prompt (str): The user's question.

    Returns:
        str: The response from the chat chain.
    """
    result = st.session_state.chat_chain.invoke({"question": user_prompt})
    return result["answer"]


def display_chat_history():
    """Display the chat history."""
    for message in st.session_state.get("messages", []):
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)


def handle_conversation():
    """Handle the conversation input from the user."""
    if "chat_chain" not in st.session_state:
        return

    if user_prompt := st.chat_input(
        "Ask about the uploaded document",
    ):
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        response = submit_question(user_prompt)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    """Main function to run the Streamlit app."""
    st.title("ChatLit")
    st.write("Upload a PDF file and ask questions about its content.")

    init_session_state()

    handle_file_upload()

    display_chat_history()

    handle_conversation()


if __name__ == "__main__":
    main()
