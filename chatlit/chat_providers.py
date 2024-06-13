import gc
import os
from abc import ABC, abstractmethod
from io import BytesIO

import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .utils import load_and_split_document


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

    Attributes:
        base_url (str): The base URL for the Ollama model API.
    """

    def __init__(self, uploaded_file: BytesIO, base_url: str = None) -> None:
        """Initialize the Ollama chat provider with a PDF file and base URL.

        Args:
            uploaded_file (BytesIO): The uploaded PDF file.
            base_url (str, optional): The base URL for the Ollama model API. Case None defaults to "http://localhost:8501".
        """
        self.base_url = (
            base_url
            if base_url is not None
            else os.environ.get("OLLAMA_BASE_URL", "http://localhost:8501")
        )
        super().__init__(uploaded_file)

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
        return ChatOllama(model="mistral", base_url=self.base_url)

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
