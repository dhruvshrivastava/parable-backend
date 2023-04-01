from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
import requests
from google.colab import drive
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
import pickle
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_pdf_data(file_path, num_pages = 1):
  reader = PdfReader(file_path)
  full_doc_text = ""
  for page in range(len(reader.pages)):
    current_page = reader.pages[page]
    text = current_page.extract_text()
    full_doc_text += text


  return Document(
        page_content=full_doc_text,
        metadata = {"source": file_path} 
    )

def source_docs(file):
    return [get_pdf_data(file)]


def search_index(source_docs):
    source_chunks = []
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)

    for source in source_docs:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    #more likely you'll pickle this inde
    with open("search_index.pickle", "wb") as f:
        pickle.dump(FAISS.from_documents(source_chunks, OpenAIEmbeddings()), f)


chain = load_qa_with_sources_chain(OpenAI(temperature=0),verbose=False, chain_type="stuff")
def print_answer(question):
    with open("search_index.pickle", "rb") as f:
        search_index = pickle.load(f)
        #print("type - ", type(search_index))
    return (
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=3),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    
    )
    

