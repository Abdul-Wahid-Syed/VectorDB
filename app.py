import os
import re
import time
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch

from langchain.chains import RetrievalQA
from langchain.schema.language_model import BaseLanguageModel
import gradio as gr
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
import boto3 
import urllib.parse




bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
    
)

# Load environment variables from .env file
load_dotenv(override=True)

# Set up MongoDB connection details

username = urllib.parse.quote_plus('abdulwahid')
password= urllib.parse.quote_plus('SHELLKODE@1234')

MONGO_URI = f"mongodb+srv://{username}:{password}@cluster0.6r2ew2d.mongodb.net/"
# client1= "mongodb+srv://abdulwahid:SHELLKODE@1234@cluster0.6r2ew2d.mongodb.net/"
# MONGO_URI=MongoClient(client)
DB_NAME = "pdfchatbot"
COLLECTION_NAME = "pdfText"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"


embeddings = BedrockEmbeddings(
    credentials_profile_name="default", region_name="us-east-1"
)

# Define field names
EMBEDDING_FIELD_NAME = "embedding"
TEXT_FIELD_NAME = "text"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
#Anthropic Parameters
inference_modifier = {'max_tokens_to_sample':4096, 
                      "temperature": 0.5,
                      "top_k":250,
                      "top_p":1,
                      "stop_sequences": ["\n\nHuman"]
                     }

def process_pdf(file,progress=gr.Progress()):
    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    new_string = ""
    for letter in progress.tqdm(file.name, desc="Uploading Your PDF into MongoDB Atlas"):
        time.sleep(0.25)

    loader = PyPDFLoader(file.name)
    pages = loader.load_and_split()

    # Print loaded pages
    print(pages)

    # Split text into documents
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)

    docs = text_splitter.split_documents(pages)
    # Remove single spaces between letters and replace double spaces
    # docs_cleaned = [re.sub(r'\s+', ' ', doc.replace(' ', '')) for doc in docs]

    # Convert 'Document' objects to strings
    docs_as_strings = [str(doc) for doc in docs]

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

    return docsearch



def query_and_display(query,history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    history_langchain_format.append(HumanMessage(content=query))
    # Set up MongoDBAtlasVectorSearch with embeddings
    vectorStore = MongoDBAtlasVectorSearch(
        collection,
        embeddings,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
    print(query)
    # Query MongoDB Atlas Vector Search
    print("---------------")
    docs = vectorStore.max_marginal_relevance_search(query, K=5)


    llm = Bedrock(model_id = "anthropic.claude-v2",
                        client = bedrock_client, 
                        model_kwargs = inference_modifier 
                        )
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
  
    for document in retriever:
        print(str(document) + "\n")


    qa = RetrievalQA.from_chain_type(
        llm, chain_type="stuff", retriever=retriever
    )
    
    retriever_output = qa.run(query)

    print(retriever_output)
    print("-------------------------------")
    return retriever_output



with gr.Blocks(css=".gradio-container {background-color: AliceBlue}") as demo:
    gr.Markdown("Generative AI Chatbot - Upload your file and Ask questions")

    with gr.Tab("Upload PDF"):
        with gr.Row():
            pdf_input = gr.File()
            pdf_output = gr.Textbox()
        pdf_button = gr.Button("Upload PDF")

    with gr.Tab("Ask question"):
 
        gr.ChatInterface(query_and_display)

    pdf_button.click(process_pdf, inputs=pdf_input, outputs=pdf_output)
    
demo.launch()