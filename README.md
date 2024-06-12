# ChatLit

## Install Instructions

First of all, we need to install Ollama for the local pipeline. Follow the official documentation instructions for this. Then from the command line run:

```shell
ollama pull mistral
ollama serve
```
This will download the Mistral 7B model files and start the Ollama inference service with it.

**Obs:** If you wnat to play around with the model, you can also run:
```shell
ollama run mistral
```

With the Ollama service setted up, it's time to install the required packages for the application itself.
Install Pytorch following the official documentation. Next, install the other required packages using PiP.

```shell
pip install python-dotenv streamlit pypdf faiss-cpu langchain langchain_community langchain_openai OpenAI transformers "sentence-transformers===2.2.2" InstructorEmbedding
```
**Obs:** For `sentence-transformers` we need the specific `2.2.2` verstion due to `InstructorEmbedding` compatibility issues with newer versions.

## Running the App

Create a `.env` with the content:
```env
OPENAI_API_KEY=[YOUR OPENAI KEY]
```

Finally, run the app.
```shell
streamlit run app.py
```