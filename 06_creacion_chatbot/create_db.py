from __future__ import annotations
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import chromadb
from dotenv import load_dotenv



class VectorDB:

    SCORE_THRESHOLD = 0.4
    MAX_RESULTS_SIMILARITY_SEARCH = 5

    def __init__(self) -> None:
        load_dotenv('.env')
        self.vectordb = self.initialize_db()
        self.retriever = self.vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":self.MAX_RESULTS_SIMILARITY_SEARCH,"score_threshold": self.SCORE_THRESHOLD,})
        
    
    def initialize_db(self):
        embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
        embedding_model_kwargs_device = os.getenv('EMBEDDING_MODEL_KWARGS_DEVICE')
        if embedding_model_name is None:
            raise ValueError('EMBEDDING_MODEL_NAME environment variable must be set.')

        if embedding_model_kwargs_device is None:
            embedding_model_kwargs_device = "cpu"

        self.embedding_model_name = embedding_model_name
        embedding_model_kwargs = {"device": embedding_model_kwargs_device}
        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_name, model_kwargs=embedding_model_kwargs)
        chroma_client = chromadb.HttpClient(host='chromadb', port=8000)
        vectordb = Chroma(collection_name = "nutrition_edlp",  persist_directory="./database_vectorial", embedding_function=embedding_model, collection_metadata={"hnsw:space": "cosine"}, client=chroma_client)
        return vectordb


    def loadDocuments(self, files_path):
        print('init load document')
        chunk_size = 1024
        chunk_overlap = 64
        all_splits= []
        for file in os.listdir(files_path):
            try:
                print('file', file)
                if ((file.endswith('pdf')) | (file.endswith('txt'))):
                    if file.endswith('.pdf'):
                        loader = PyPDFLoader(os.path.join(files_path, file))
                    else:
                        loader = TextLoader(os.path.join(files_path, file), encoding='utf8')

                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len)
                    
                    splits = text_splitter.split_documents(documents)  
                    all_splits += splits        
                    print('Documents loaded in db')
            except Exception as e:        
                print('error', file, e)    
        print('finish load document', len(all_splits))
        print('save_document_in_db')
        print(all_splits[-1])
        self.vectordb.add_documents(documents=all_splits)
        print('finish Documents loaded in db')