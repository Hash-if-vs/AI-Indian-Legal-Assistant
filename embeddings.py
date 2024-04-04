
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def createvectordb(pdf_docs):
    loader= PyPDFDirectoryLoader(pdf_docs,glob="**/*.pdf")
    documents= loader.load()
 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs= text_splitter.split_documents(documents)
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    model_norm = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs
    ) 
    persist_directory = 'D:/personal/projects/legal_assistant_chatbot/demodb'
    embedding = model_norm
    vectordb = Chroma.from_documents(docs,
                                 embedding=embedding,
                                 persist_directory=persist_directory)
    return vectordb
vectordb= createvectordb('D:/personal/projects/legal_assistant_chatbot/New folder')


