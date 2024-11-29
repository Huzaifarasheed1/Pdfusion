import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time
import tempfile

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.title("Gemma Model Document Q&A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to create the vector store and load documents from user-uploaded files
def vector_embedding(uploaded_files):

    if "vectors" not in st.session_state:
        with st.spinner('Processing documents and creating vector store...'):
            # Initialize embeddings and loaders
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

            # Load documents from uploaded files
            docs = []
            for uploaded_file in uploaded_files:
                # Save each uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_file_path = temp_file.name
                
                # Use PyPDFLoader to load the saved PDF file
                loader = PyPDFLoader(temp_file_path)
                docs.extend(loader.load())

            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)

            # Create FAISS vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

            st.success('Vector Store DB is ready!')

# File uploader for multiple PDF files
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")

# Input field for question
prompt1 = st.text_input("Enter your question from documents")

# Button to initiate document embedding
if st.button("Load Documents"):
    if uploaded_files:
        vector_embedding(uploaded_files)
    else:
        st.error("Please upload at least one PDF file.")

# If a question is entered, run the Q&A process
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Display chat-like response with a loading spinner
    with st.spinner('Fetching response...'):
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start_time

        # Display response in chat format
        st.write(f"**You:** {prompt1}")
        st.write(f"**Gemma:** {response['answer']}")
        st.write(f"*Response time: {response_time:.2f} seconds*")
    
    # With a Streamlit expander for document similarity search
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")

