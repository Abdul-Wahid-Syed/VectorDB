from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import boto3
from langchain.llms.bedrock import Bedrock
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)
llm = Bedrock(model_id="anthropic.claude-v2:1",client=bedrock_client,model_kwargs = {"temperature":1e-10,"max_tokens_to_sample": 8191})

pdf_file_paths = []
while True:
    pdf_file_path = input("Enter the path to a PDF file (or 'done' to finish adding files): ")
    if pdf_file_path == 'done':
        break
    pdf_file_paths.append(pdf_file_path)

doc = []
for pdf_path in pdf_file_paths:
    loader = PyPDFLoader(pdf_path)
    doc.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], chunk_overlap=0)
texts = text_splitter.split_documents(doc)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(documents=texts, embedding=embeddings)
retriever = db.as_retriever(search_type='mmr', search_kwargs={"k": 3})

template = """
Human:
Answer the question as truthfully as possible strictly using only the provided text
instructions:
1. If the table formatted datas are present in the pdf analyse it clearly to answer , ensure do not mismatch the datas
2. Read all the pages and every paragraph and every line to generate answer for the asked question accurately
<text>{context}</text>
<question>{question}</question>
<answer>
Assistant:"""

qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": qa_prompt, "verbose": False}
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
    verbose=False
)

question = "what is the eligibility and limits for unit manager for business travel"
result = qa.run(question)
print(result)