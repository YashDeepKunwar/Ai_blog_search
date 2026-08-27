from http import client

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

from langchain_classic import hub
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

import streamlit as st

st.set_page_config(page_title="AI Blog Search", page_icon=":mag_right:")
st.header(":blue[Agentic RAG with LangGraph:] :green[AI Blog Search]")

# Initialize session state variables if they don't exist
if 'qdrant_host' not in st.session_state:
    st.session_state.qdrant_host = ""
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

def set_sidebar():
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("API Configuration")
        
        qdrant_host = st.text_input("Enter your Qdrant Host URL:", type="password")
        qdrant_api_key = st.text_input("Enter your Qdrant API key:", type="password")
        gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

        if st.button("Done"):
            if qdrant_host and qdrant_api_key and gemini_api_key:
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_api_key = qdrant_api_key
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API keys saved!")
            else:
                st.warning("Please fill all API fields")

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

        # Initialize Qdrant client
        client = QdrantClient(
            url=st.session_state.qdrant_host,
            api_key=st.session_state.qdrant_api_key,
            timeout=120,
        )

        # Create collection if it doesn't exist
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

# Edges
## Check Relevance
def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, model="gemini-3.5-flash", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    
# Nodes
## agent node
def agent(state, tools):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    system_message = SystemMessage(
        content=(
            "You are answering questions about a blog that has already been indexed. "
            "You must call the retrieve_blog_posts tool for every user question before "
            "answering. Do not ask the user to provide the blog URL again and do not "
            "claim that you cannot access the blog."
        )
    )
    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, streaming=True, model="gemini-3.5-flash")
    model = model.bind_tools(tools)
    response = model.invoke([system_message, *messages])
    
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

## rewrite node
def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=f""" \n 
                    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                    Here is the initial question:
                    \n ------- \n
                    {question} 
                    \n ------- \n
                    Formulate an improved question: """,
        )
    ]

    # Grader
    model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, temperature=0, model="gemini-3.5-flash", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

## generate node
def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    # Initialize a Chat Prompt Template
    prompt_template = hub.pull("rlm/rag-prompt")

    # Initialize a Generator (i.e. Chat Model)
    chat_model = ChatGoogleGenerativeAI(api_key=st.session_state.gemini_api_key, model="gemini-3.5-flash", temperature=0, streaming=True)

    # Initialize a Output Parser
    output_parser = StrOutputParser()
    
    # RAG Chain
    rag_chain = prompt_template | chat_model | output_parser

    response = rag_chain.invoke({"context": docs, "question": question})
    
    return {"messages": [response]}

# graph function
def get_graph(retriever_tool):
    tools = [retriever_tool]  # Create tools list here
    
    # Define a new graph
    workflow = StateGraph(AgentState)

    # Use partial to pass tools to the agent function
    workflow.add_node("agent", partial(agent, tools=tools))
    
    # Rest of the graph setup remains the same
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)  # Re-writing the question
    workflow.add_node(
        "generate", generate
    )  # Generating a response after we know the documents are relevant
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        grade_documents,
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    graph = workflow.compile()

    return graph

def generate_message(graph, inputs):
    last_message = None

    for output in graph.stream(inputs):
        for key, value in output.items():
            if isinstance(value, dict):
                messages = value.get("messages", [])
                if messages:
                    last_message = messages[-1]

    if last_message is None:
        return ""

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
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=50
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
            st.warning("The blog URL did not contain readable text.")
            return False
        st.success(f"Indexed {chunk_count} blog chunks successfully.")
        return True
    except Exception as e:
        st.error(f"Error adding documents: {str(e)}")
        return False

def answer_query(retriever, query):
    """Retrieve indexed blog content and answer the user's question."""
    documents = retriever.invoke(query)
    if not documents:
        return "No matching blog content was found. Index the blog URL first."

    context = "\n\n".join(document.page_content for document in documents)
    prompt = PromptTemplate.from_template(
        "Answer the question using only the blog excerpts below. "
        "If the excerpts do not contain the answer, say that the indexed blog "
        "does not provide enough information.\n\n"
        "Blog excerpts:\n{context}\n\nQuestion: {question}"
    )
    model = ChatGoogleGenerativeAI(
        api_key=st.session_state.gemini_api_key,
        model="gemini-3.5-flash",
        temperature=0,
    )
    return (prompt | model | StrOutputParser()).invoke(
        {"context": context, "question": query}
    )

def main():
    set_sidebar()

    # Check if API keys are set
    if not all([st.session_state.qdrant_host, 
                st.session_state.qdrant_api_key, 
                st.session_state.gemini_api_key]):
        st.warning("Please configure your API keys in the sidebar first")
        return

    # Initialize components
    embedding_model, client, db = initialize_components()
    if not all([embedding_model, client, db]):
        return

    # Initialize retriever and tools
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about blog posts on LLMs, LLM agents, prompt engineering, and adversarial attacks on LLMs.",
    )
    tools = [retriever_tool]

    # URL input section
    url = st.text_input(
        ":link: Paste the blog link:",
        placeholder="e.g., https://lilianweng.github.io/posts/2023-06-23-agent/"
    )
    if st.button("Enter URL"):
        if url:
            with st.spinner("Processing documents..."):
                if add_documents_to_qdrant(url, db):
                    st.success("Documents added successfully!")
                else:
                    st.error("Failed to add documents")
        else:
            st.warning("Please enter a URL")

    # Query section
    graph = get_graph(retriever_tool)
    query = st.text_area(
        ":bulb: Enter your query about the blog post:",
        placeholder="e.g., What does Lilian Weng say about the types of agent memory?"
    )

    if st.button("Submit Query"):
        if not query:
            st.warning("Please enter a query")
            return

        inputs = {"messages": [HumanMessage(content=query)]}
        with st.spinner("Generating response..."):
            try:
                if not retriever.invoke(query) and url:
                    with st.spinner("Indexing the blog URL first..."):
                        chunk_count = index_blog(url, db)
                    if chunk_count:
                        st.info(f"Indexed {chunk_count} blog chunks.")
                    else:
                        st.warning("The blog URL did not contain readable text.")
                response = answer_query(retriever, query)
                st.write(response)
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    st.markdown("---")
    st.write("Built with :blue-background[LangChain] | :blue-background[LangGraph] by [Yash](https://www.linkedin.com/in/yash-deep-kunwar-745bb6317/)")

if __name__ == "__main__":
    main()