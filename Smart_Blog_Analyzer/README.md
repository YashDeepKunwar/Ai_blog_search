# Smart Blog Analyzer: Agentic RAG with LangGraph

A Streamlit application that performs intelligent, agent-driven Retrieval-Augmented Generation (RAG) on technical blogs. It uses LangGraph to route queries, grade document relevance, and autonomously rewrite prompts for optimal context retrieval.

## Core Features

* **Agentic Workflow:** Utilizes a state graph to evaluate document relevance. If retrieved chunks lack context, the agent automatically transforms the query and searches again.
* **Dynamic Text Chunking:** Features an adjustable UI slider to set document chunk sizes (200-1200 characters) with a dynamic 10% overlap, allowing users to experiment with retrieval performance.
* **Modern LLM Stack:** Powered by Google's `gemini-3.5-flash` for high-speed routing and generation, alongside `gemini-embedding-001` for vectorization.

## Architecture & Workflow

1. **Document Ingestion:** `WebBaseLoader` reads the provided public URL. The text is split using a `RecursiveCharacterTextSplitter` based on the user's selected chunk size.
2. **Vector Storage:** Chunks are embedded and upserted into a Qdrant Cloud collection (`qdrant_db`).
3. **LangGraph Evaluation Loop:**
   * **Retrieve:** The agent searches the Qdrant database using the user's query.
   * **Grade:** An LLM with structured output evaluates the retrieved chunks. 
   * **Rewrite:** If the chunks are graded as irrelevant, the agent rewrites the query to be more database-friendly and loops back to retrieval.
4. **Generation:** Once relevant context is confirmed, a custom prompt template synthesizes the final answer.

## Setup & Installation

Install the required dependencies explicitly defined in the project:

```bash
pip install -r requirements.txt
```

Launch the Streamlit interface:

```bash
streamlit run app.py
```

## Required Configuration

To run the application, you must provide the following credentials in the sidebar:

* **Qdrant Host URL:** The complete URL of your Qdrant Cloud cluster (must include `https://` and port if applicable, e.g., `:6333`).
* **Qdrant API Key:** The access key for your specific Qdrant cluster.
* **Gemini API Key:** A valid Google AI Studio API key.

*Note: The application connects directly to Qdrant Cloud and does not require a local Docker instance of Qdrant.*

## Usage Guide

1. Enter your API credentials in the sidebar and click **Save Settings**.
2. Adjust the **Document Chunk Size** slider under *RAG Parameters* to dictate how the blog text will be split.
3. Paste a public technical blog URL (e.g., a Hugging Face or LangChain post) into the primary input field.
4. Click **Index Blog** to process, embed, and store the document chunks in Qdrant.
5. Type your question into the query box and click **Submit Query**. The LangGraph agent will analyze the request, retrieve the data, and generate a response based on the blog context.
