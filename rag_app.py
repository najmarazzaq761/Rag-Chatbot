import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()
import os

# Loaders
from langchain_community.document_loaders import WebBaseLoader

# Text Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Embeddings & Vectorstore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Groq LLM
from langchain_groq import ChatGroq

# Chains
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Prompts
from langchain_core.prompts import ChatPromptTemplate


# Setting page configuration 
st.set_page_config(page_title="✨Data Seekho Guide", page_icon="🧠", layout="wide")
st.markdown("<h1 style='text-align: center;'><span style='color: #7abd06;'>Data</span> <span style='color: white;'>Seekho Guide</span></h1>", unsafe_allow_html=True)
with st.sidebar:
    st.image("logo.png", use_container_width=True)
    st.markdown("**Data Seekho Guide** app is designed to provide you with any information about **Data Seekho**.")
    st.title("Configuration")
    temp = st.slider("Temperature", min_value=0.0, max_value=0.5, value=0.2)

# Function to fetch all internal links from the website
@st.cache_data
def fetch_all_links(base_url):
    """
    Fetch all internal links from the website's base URL.
    """
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")
    links = set()

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if href.startswith("/"):  # Internal links
            links.add(base_url + href.strip("/"))
        elif base_url in href: 
            links.add(href)
    
    return list(links)

# Load data from all pages of the website
@st.cache_data
def load_data():
    """
    Load data from all pages of the website.
    """
    base_url = "https://dataseekho.com/"
    all_links = fetch_all_links(base_url)

    # Use WebBaseLoader to load data from all links
    all_data = []
    for link in all_links:
        try:
            loader = WebBaseLoader([link])
            all_data.extend(loader.load())
        except Exception as e:
            st.warning(f"Failed to load data from {link}: {e}")
    
    return all_data

# Split the loaded data into chunks
@st.cache_data
def split_data(_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500 , chunk_overlap=0)
    return text_splitter.split_documents(_data)


# Create a vector store using FAISS
@st.cache_resource
def create_vector_store(_docs):
    # embeddings = HuggingFaceEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    return FAISS.from_documents(documents=_docs, embedding=embeddings)

# Load and process data
data = load_data()
docs = split_data(data)
vectorstore = create_vector_store(docs)

# Set up retriever and LLM
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
 # api_key =os.environ["GROQ_API_KEY"] = os.getenv["GROQ_API_KEY"]
llm = ChatGroq(
    groq_api_key = os.environ["GROQ_API_KEY"],
    model="qwen/qwen3.8-27b",
    temperature=temp
    )


# Defining the prompt template
system_prompt = (
    "You are a helpful assistant designed to answer questions specifically related to Data Seekho, an E-learning data platform. "
    "Use the retrieved context provided below to accurately answer the user's question. "
    "If the answer is not found in the context, simply say you don't know. "
    "Keep your response concise, with a maximum of three sentences.\n\n"
    "If the user sends a greeting (like 'hi', 'hello', 'hey' , or 'what you can do'), respond with a friendly greeting, "
    "introduce yourself as a Data Seekho guide, and let them know you're available to assist with any questions about Data Seekho. "
    "Also, ask: 'How can I help you today?'"
    "please only greet them once and then only give answer to queries and don't introduce yourself with every answer\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# session state for chatbot
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input and response generation
query = st.chat_input("🗣️ Enter your query:")
if query:
    st.chat_message("user").write(query)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]
    
 # display and store
    st.chat_message("assistant").write(answer)
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})

# clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask Question"}]
    
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
