import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

from chatlit.chat_providers import OllamaChatProvider, OpenAIChatProvider

st.set_page_config(layout="wide")

load_dotenv()


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
