import pymongo
import streamlit as st
import time
from pymongo import MongoClient
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import boto3
import urllib.parse
from langchain.chains import RetrievalQA

# Set up MongoDB connection details
username = urllib.parse.quote_plus('abdulwahid')
password = urllib.parse.quote_plus('SHELLKODE@1234')
MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.6r2ew2d.mongodb.net/"
DB_NAME = "pdfchatbot"
COLLECTION_NAME = "pdfText"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Load environment variables from .env file
# Assuming that you have your credentials set up properly for BedrockEmbeddings
embeddings = BedrockEmbeddings(credentials_profile_name="default", region_name="us-east-1")

# Anthropic Parameters
inference_modifier = {
    'max_tokens_to_sample': 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"]
}

def process_pdf(file):
    st.write("Uploading Your PDF into MongoDB Atlas...")
    time.sleep(1)

    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()

    # Print loaded pages
    # st.write(pages)

    # Split text into documents
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    docs = text_splitter.split_documents(pages)

    # Set up MongoDBAtlasVectorSearch with embeddings
    vectorStore = MongoDBAtlasVectorSearch(
        collection, embeddings, index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )

    # Insert the documents into MongoDB Atlas Vector Search
    docsearch = vectorStore.from_documents(
        docs,
        embeddings,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    st.write("Document Processed")
    return docsearch

def delete_documents():
    st.write("Deleting Documents in MongoDB Atlas...")
    
    # Assuming you want to delete all documents in the collection
    result = collection.delete_many({})

    st.write(f"Number of documents deleted: {result.deleted_count}")


def query_and_display(query, history):

    start_time = time.time()

    
    # Set up MongoDBAtlasVectorSearch with embeddings
    vectorStore = MongoDBAtlasVectorSearch(
        collection,
        embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    # Query MongoDB Atlas Vector Search
    docs = vectorStore.max_marginal_relevance_search(query, K=5)


    llm = Bedrock(
        model_id="anthropic.claude-v2",
        client=boto3.client(service_name="bedrock-runtime", region_name="us-east-1"),
        model_kwargs=inference_modifier
    )

    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    qa = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever
    )

    retriever_output = qa.run(query)

    st.write("Retriever Output:")
    st.write(retriever_output)
    end_time = time.time()
    elapsed_time = end_time - start_time

    
    st.write(f"Time taken: {elapsed_time} seconds")
    st.write("-------------------------------")

# Streamlit UI
st.title("Generative AI Chatbot")
st.markdown("Upload your file and ask questions")

# pdf_input = st.file_uploader("Upload PDF", type=["pdf"])


# if pdf_input is not None:
#     if st.button("Process PDF"):
#         process_pdf(pdf_input)

query_input = st.text_input("Ask a question")
query_history = []  # Assuming you have a history list
if st.button("Ask Question"):
    result = query_and_display(query_input, query_history)
    query_history.append((query_input, result))

if st.button("Delete Documents"):
    delete_documents()