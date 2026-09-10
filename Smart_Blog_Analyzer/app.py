from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from uuid import uuid4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.tools.retriever import create_retriever_tool

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from functools import partial

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

import streamlit as st

st.set_page_config(page_title="AI Blog Search Engine", page_icon=":mag_right:")
st.header(":blue[Agentic RAG:] :green[Smart Blog Analyzer]")

# Initialize session state variables
if 'qdrant_host' not in st.session_state:
    st.session_state.qdrant_host = ""
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 500

def set_sidebar():
    """Setup sidebar for API keys and RAG configuration."""
    with st.sidebar:
        st.subheader("API Configuration")
        
        qdrant_host = st.text_input("Enter your Qdrant Host URL:", type="password")
        qdrant_api_key = st.text_input("Enter your Qdrant API key:", type="password")
        gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

        # Unique Addition: Giving the user control over the RAG pipeline parameters
        st.markdown("---")
        st.subheader("RAG Parameters")
        st.caption("Adjust how the blog is split for vector storage.")
        chunk_size = st.slider("Document Chunk Size", min_value=200, max_value=1200, value=500, step=100)

        if st.button("Save Settings"):
            if qdrant_host and qdrant_api_key and gemini_api_key:
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_api_key = qdrant_api_key
                st.session_state.gemini_api_key = gemini_api_key
                st.session_state.chunk_size = chunk_size
                st.success("Settings saved securely!")
            else:
                st.warning("Please fill all API fields to proceed.")

def initialize_components():
    """Initialize components that require API keys"""
    if not all([st.session_state.qdrant_host, 
                st.session_state.qdrant_api_key, 
                st.session_state.gemini_api_key]):
        return None, None, None

    try:
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            api_key=st.session_state.gemini_api_key,
        )

        client = QdrantClient(
            url=st.session_state.qdrant_host,
            api_key=st.session_state.qdrant_api_key,
            timeout=120,
        )

        if not client.collection_exists("qdrant_db"):
            db = QdrantVectorStore.from_texts(
                texts=[],
                embedding=embedding_model,
                collection_name="qdrant_db",
                url=st.session_state.qdrant_host,
                api_key=st.session_state.qdrant_api_key,
                timeout=120,
            )
        else:
            db = QdrantVectorStore(
                client=client,
                collection_name="qdrant_db",
                embedding=embedding_model,
            )

        return embedding_model, client, db
        
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return None, None, None

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """Determines whether the retrieved documents are relevant to the question."""
    
    print("[Agent Router] Evaluating retrieved document relevance...")

    class Grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, model="gemini-3.5-flash", streaming=True)
    llm_with_tool = model.with_structured_output(Grade)

    # Custom prompt instead of copied tutorial text
    prompt = PromptTemplate(
        template="""You are an AI assistant evaluating search results for relevance.
        
        Context Document:
        {context}
        
        User Query: {question}
        
        Does the context document contain keywords, themes, or information that helps answer the user query?
        Provide a binary 'yes' or 'no' score.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    if scored_result.binary_score.lower() == "yes":
        print("[Agent Router] Decision: Documents are relevant.")
        return "generate"
    else:
        print("[Agent Router] Decision: Irrelevant docs. Triggering rewrite.")
        return "rewrite"
    
def agent(state, tools):
    """Invokes the agent model to decide next steps."""
    print("[Node: Agent] Processing user query...")
    messages = state["messages"]
    system_message = SystemMessage(
        content=(
            "You are an AI assistant that answers questions based on a specific blog post. "
            "You must use the retrieve_blog_posts tool to find information before answering. "
            "Do not ask the user for a URL, assume the database is already populated."
        )
    )
    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, streaming=True, model="gemini-3.5-flash")
    model = model.bind_tools(tools)
    response = model.invoke([system_message, *messages])
    
    return {"messages": [response]}

def rewrite(state):
    """Transform the query to produce a better question."""
    print("[Node: Rewrite] Optimizing query for better retrieval...")
    messages = state["messages"]
    question = messages[0].content

    # Custom rewrite prompt
    msg = [
        HumanMessage(
            content=f"""Analyze the user's original query and rewrite it to be more specific for a vector database search.
            Original query: {question}
            
            Provide only the improved query string as your response:""",
        )
    ]

    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, model="gemini-3.5-flash", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """Generate final answer using custom RAG prompt."""
    print("[Node: Generate] Crafting final response...")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    # Replaced hub.pull with a custom ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_template(
        "You are an insightful technical assistant. Use the following blog context to answer the user's question.\n"
        "If you cannot find the answer in the context, state clearly that the blog does not mention it.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    chat_model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, model="gemini-3.5-flash", temperature=0, streaming=True)
    output_parser = StrOutputParser()
    
    rag_chain = prompt_template | chat_model | output_parser
    response = rag_chain.invoke({"context": docs, "question": question})
    
    return {"messages": [response]}

def get_graph(retriever_tool):
    tools = [retriever_tool] 
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", partial(agent, tools=tools))
    workflow.add_node("retrieve", ToolNode(tools))
    workflow.add_node("rewrite", rewrite) 
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    return workflow.compile()

def generate_message(graph, inputs):
    """Helper to extract final text from graph stream."""
    last_message = None
    for output in graph.stream(inputs):
        for key, value in output.items():
            if isinstance(value, dict):
                messages = value.get("messages", [])
                if messages:
                    last_message = messages[-1]

    if last_message is None:
        return "I couldn't generate a response."

    content = getattr(last_message, "content", last_message)
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)

def index_blog(url, db):
    """Load a blog URL and store its chunks in Qdrant."""
    docs = WebBaseLoader(url).load()
    
    # Implementing the dynamic chunk size from session state
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=st.session_state.chunk_size, 
        chunk_overlap=int(st.session_state.chunk_size * 0.1) # 10% overlap
    )
    doc_chunks = text_splitter.split_documents(docs)
    if not doc_chunks:
        return 0

    uuids = [str(uuid4()) for _ in doc_chunks]
    db.add_documents(documents=doc_chunks, ids=uuids)
    return len(doc_chunks)

def add_documents_to_qdrant(url, db):
    try:
        chunk_count = index_blog(url, db)
        if not chunk_count:
            st.warning("The URL provided did not contain extractable text.")
            return False
        st.success(f"Successfully processed and indexed {chunk_count} chunks!")
        return True
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return False

def main():
    set_sidebar()

    if not all([st.session_state.qdrant_host, 
                st.session_state.qdrant_api_key, 
                st.session_state.gemini_api_key]):
        st.info("👈 Please configure your API keys in the sidebar to get started.")
        return

    embedding_model, client, db = initialize_components()
    if not all([embedding_model, client, db]):
        return

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search vector database for technical blog content. Use this to answer queries about the indexed URL.",
    )

    # Replaced Lilian Weng with Hugging Face for a more realistic ML student vibe
    url = st.text_input(
        ":link: Paste the technical blog link:",
        placeholder="e.g., https://huggingface.co/blog/open-llm-leaderboard-drop"
    )
    
    if st.button("Index Blog"):
        if url:
            with st.spinner("Chunking and embedding documents..."):
                add_documents_to_qdrant(url, db)
        else:
            st.warning("Please enter a valid URL first.")

    graph = get_graph(retriever_tool)
    query = st.text_area(
        ":bulb: What would you like to know about this blog?",
        placeholder="e.g., What are the key evaluation metrics discussed?"
    )

    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a query.")
            return

        inputs = {"messages": [HumanMessage(content=query)]}
        
        with st.spinner("Agent is analyzing and formulating response..."):
            try:
                # Actually utilizing the graph built above instead of bypassing it
                response = generate_message(graph, inputs)
                st.write(response)
            except Exception as e:
                st.error(f"Error during agent execution: {str(e)}")

    st.markdown("---")
    st.write("Built with :blue-background[LangChain] | :blue-background[LangGraph] by [Yash](https://www.linkedin.com/in/yash-deep-kunwar-745bb6317/)")

if __name__ == "__main__":
    main()