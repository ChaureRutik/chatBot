from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
#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  
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DATA_PATH="data/"
def load_pdf_files(data):
    loader=DirectoryLoader(data,             
                   glob='*.pdf',
                   loader_cls=PyPDFLoader)
    
    documents=loader.load()
    return documents
documents=load_pdf_files(data=DATA_PATH)
print(len(documents))

# divide into chunks
def create_chunks(extracted_data):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)
    text_chunk=text_spliter.split_documents(extracted_data)
    return text_chunk

text_chunk=create_chunks(extracted_data=documents)
print("len of text chunk",len(text_chunk))

#vector embedding
def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="")
    return embedding_model

embedding_model=get_embedding_model()

#store embedding in faiss
DB_FAISS_PATH="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunk,embedding_model)
db.save_local(DB_FAISS_PATH)