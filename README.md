# ChatLit

A RAG application for chatting with PDF documents. The LLM backend can be supplied by the OpenAI API or Ollama for local development.

## Local Installation

### Install Instructions

First of all, we need to install Ollama for the local pipeline. Follow the official documentation instructions for this. Then from the command line run:

```shell
ollama pull llama3
ollama serve
```
This will download the LLaMa 3 7B model files and start the Ollama inference service with it.

**Obs:** If you want to play around with the model, you can also run:
```shell
ollama run llama3
```

After setting up the Ollama service, it's time to install the required packages for the application itself.
Install Pytorch following the official documentation. Next, install the other required packages using PiP.

```shell
pip install python-dotenv streamlit pypdf faiss-cpu langchain langchain_community langchain_openai OpenAI transformers "sentence-transformers===2.2.2" InstructorEmbedding
```
**Obs:** For `sentence-transformers` we need the specific `2.2.2` version due to `InstructorEmbedding` compatibility issues with newer versions.

### Running the App

Create a `.env` with the content:
```env
OPENAI_API_KEY=[YOUR OPENAI KEY]
```

Finally, run the app.
```shell
streamlit run app.py
```

In your browser, access the app URL:
```
http://localhost:8501
```

## Run with Docker

This project includes a `docker-compose.yml` file to run the entire environment without the need to install it in the host machine. Two services are set up: 1) the **Ollama service** that will supply the LLM server and 2) the streamlit based **web app service**.

To run the docker compose environment, first create your `.env` file with the chosen variables. After that run:

```shell
docker compose up --build
```

This will set up the environment itself. If you want to use the OpenAI API as backend this should be enougth, but to use a local LLM stack you need to download the model for the Ollama container to use. So, with the environment up, run:

```shell
docker exec -it ollama ollama pull llama3
```

Wait for the download to complete. Then access the app in your browser with the URL:

```
http://localhost:8501
```