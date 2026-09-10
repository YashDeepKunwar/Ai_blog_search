# AI Blog Search

A small Streamlit app that indexes a blog and answers questions about its content.

## How It Works

1. `WebBaseLoader` downloads the blog page.
2. The page is split into text chunks.
3. Gemini creates embeddings for the chunks.
4. Qdrant stores and searches the embeddings.
5. Gemini answers questions using the matching blog content.

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app.py
```

## Required Keys

Enter these values in the sidebar:

- **Qdrant Host URL**: Your Qdrant Cloud cluster URL.
- **Qdrant API key**: An API key for that cluster.
- **Gemini API key**: A Google AI Studio API key.

Use API keys, not OAuth tokens. The app connects to Qdrant Cloud and does not start a local Qdrant server.

## Using the App

1. Enter the three keys and click **Done**.
2. Paste a public blog URL.
3. Click **Enter URL** to index the blog.
4. Enter a question and click **Submit Query**.

If you submit a question before indexing, the app automatically indexes the pasted URL first. Blog text is stored in the `qdrant_db` collection.

## Notes

- The app uses the Gemini `gemini-embedding-001` model for embeddings.
- Blog text is split into chunks of about 500 characters.
- Qdrant requests use a 120-second timeout to support larger blog pages.
- A public URL is required so `WebBaseLoader` can read the page.