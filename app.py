import gc
from abc import ABC, abstractmethod
from io import BytesIO
from typing import List

import streamlit as st
import torch
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pypdf import PdfReader

st.set_page_config(layout="wide")

load_dotenv()


class BaseChatProvider(ABC):
    """Abstract base class for chat providers.

    This class defines the template for creating a chat provider that
    processes a PDF file and generates conversational responses.

    Attributes:
        vector_store (VectorStore): The vector store for retrieving text.
        chat_chain (ConversationalRetrievalChain): The chat chain for conversation.
    """

    def __init__(self, uploaded_file: BytesIO) -> None:
        """Initialize the chat provider with a PDF file.

        Args:
            uploaded_file (BytesIO): The uploaded PDF file.
        """
        self.vector_store: VectorStore = self._create_vector_store(uploaded_file)
        self.chat_chain: ConversationalRetrievalChain = self._create_chat_chain(
            self.vector_store
        )

    @property
    @abstractmethod
    def embbedings(self) -> Embeddings:
        """Abstract property to get the embeddings model.

        Returns:
            Embeddings: The embeddings model.
        """
        pass

    @property
    @abstractmethod
    def llm(self) -> BaseChatModel:
        """Abstract property to get the language model.

        Returns:
            BaseChatModel: The language model.
        """
        pass

    def _create_vector_store(self, uploaded_file) -> VectorStore:
        """Create a vector store from the uploaded document.

        Args:
            uploaded_file (BytesIO): The uploaded PDF file.

        Returns:
            VectorStore: The vector store created from the document text chunks.
        """
        embbedings = self.embbedings
        text_chunks = load_and_split_document(uploaded_file)
        vector_store = FAISS.from_texts(text_chunks, embbedings)
        return vector_store

    def _create_chat_chain(
        self, vector_store: VectorStore
    ) -> ConversationalRetrievalChain:
        """Create a conversational retrieval chain.

        Args:
            vector_store (VectorStore): The vector store for retrieving text.

        Returns:
            ConversationalRetrievalChain: The chat chain for conversation.
        """
        llm = self.llm
        memory_buffer = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        chat_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vector_store.as_retriever(), memory=memory_buffer
        )
        return chat_chain

    def query(self, prompt: str) -> str:
        """Query the chat chain with a user prompt.

        Args:
            prompt (str): The user's question.

        Returns:
            str: The response from the chat chain.
        """
        result = self.chat_chain.invoke({"question": prompt})
        return result["answer"]


class OllamaChatProvider(BaseChatProvider):
    """Chat provider using the Ollama model and HuggingFace embeddings.

    This class implements the abstract methods defined in BaseChatProvider.
    """

    @property
    def embbedings(self) -> Embeddings:
        """Get the HuggingFace embeddings model.

        Returns:
            Embeddings: The HuggingFace embeddings model.
        """
        return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

    @property
    def llm(self) -> BaseChatModel:
        """Get the Ollama language model.

        Returns:
            BaseChatModel: The Ollama language model.
        """
        return ChatOllama(model="mistral")

    def _create_vector_store(self, uploaded_file) -> VectorStore:
        """Create a vector store and manage GPU resources.

        Args:
            uploaded_file (BytesIO): The uploaded PDF file.

        Returns:
            VectorStore: The vector store created from the document text chunks.
        """
        vector_store = super()._create_vector_store(uploaded_file)
        gc.collect()
        torch.cuda.empty_cache()
        return vector_store


class OpenAIChatProvider(BaseChatProvider):
    """Chat provider using the OpenAI models for embeddings and language.

    This class implements the abstract methods defined in BaseChatProvider.
    """

    @property
    def embbedings(self) -> Embeddings:
        """Get the OpenAI embeddings model.

        Returns:
            Embeddings: The OpenAI embeddings model.
        """
        return OpenAIEmbeddings()

    @property
    def llm(self) -> BaseChatModel:
        """Get the OpenAI language model.

        Returns:
            BaseChatModel: The OpenAI language model.
        """
        return ChatOpenAI()


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


def prepare_chat(uploaded_file: BytesIO, chat_provider_type: str):
    """Prepare the chat provider and store it in session state.

    Args:
        uploaded_file (BytesIO): The uploaded PDF file.
        chat_provider_type (str): The type of chat provider to use.
    """
    providers_classes = {
        "OpenAI": OpenAIChatProvider,
        "Ollama (Mistral)": OllamaChatProvider,
    }
    with st.spinner(f"Loading {uploaded_file.name}"):
        chat_provider = providers_classes[chat_provider_type](uploaded_file)
        st.session_state.chat_provider = chat_provider

    st.rerun()


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def handle_file_upload():
    """Handle the file upload and chat provider selection process."""
    with st.sidebar:
        chat_provider_type = st.selectbox(
            "Chat type",
            ("OpenAI", "Ollama (Mistral)"),
            disabled="chat_provider" in st.session_state,
        )
        if uploaded_file := st.file_uploader("Choose a PDF file", type="pdf"):
            if "chat_provider" not in st.session_state:
                prepare_chat(uploaded_file, chat_provider_type)


def display_chat_history():
    """Display the chat history in the Streamlit app."""
    for message in st.session_state.get("messages", []):
        role = message["role"]
        content = message["content"]
        st.chat_message(role).markdown(content)


def handle_conversation():
    """Handle the conversation input and response process."""
    if "chat_provider" not in st.session_state:
        return

    if user_prompt := st.chat_input(
        "Ask about the uploaded document",
    ):
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        answer = st.session_state.chat_provider.query(user_prompt)

        st.chat_message("assistant").markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


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
