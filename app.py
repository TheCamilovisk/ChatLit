from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader

load_dotenv()


def load_document(uploaded_file: BytesIO):
    reader = PdfReader(uploaded_file)
    raw_text = "\n".join(page.extract_text() for page in reader.pages)
    return raw_text


def load_and_split_document(uploaded_file: BytesIO):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    raw_text = load_document(uploaded_file)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def create_vector_store_from_document(uploaded_document: BytesIO):
    embeddings = OpenAIEmbeddings()
    text_chunks = load_and_split_document(uploaded_document)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store


def create_conversational_chain(vector_store: VectorStore):
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(),
        retriever=vector_store.as_retriever(),
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        ),
    )

    return conversational_chain


def handle_conversation_creation(uploaded_file: BytesIO):
    with st.spinner(f"Loading {uploaded_file.name}"):
        vector_store = create_vector_store_from_document(uploaded_file)
        conversational_chain = create_conversational_chain(vector_store)
        st.session_state.conversational_chain = conversational_chain
        st.rerun()


def handle_file_upload():
    if uploaded_file := st.file_uploader("Choose a PDF file", type="pdf"):
        if "conversational_chain" not in st.session_state:
            handle_conversation_creation(uploaded_file)


def submit_question():
    question = st.session_state.question_widget
    result = st.session_state.conversational_chain.invoke({"question": question})
    st.session_state.converstation_history = result["chat_history"]
    st.session_state.question_widget = ""


def display_chat_history():
    for message in st.session_state.get("converstation_history", []):
        messeger = ":green[User]" if isinstance(message, HumanMessage) else ":blue[AI]"
        st.subheader(messeger)
        st.write(message.content)


def handle_conversation():
    display_chat_history()

    st.text_input(
        "Ask about the uploaded document",
        key="question_widget",
        on_change=submit_question,
    )


def main():
    st.title("Q&A Chatbot")
    st.write("Upload a PDF file and ask questions about its content.")

    handle_file_upload()

    if "conversational_chain" in st.session_state:
        handle_conversation()


if __name__ == "__main__":
    main()
