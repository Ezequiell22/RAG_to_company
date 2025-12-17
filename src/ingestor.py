import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from src.config import Config

class IngestionService:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
        
    def ingest_documents(self):
        """Carrega documentos, divide em chunks e cria/atualiza o índice vetorial."""
        if not os.path.exists(Config.DATA_DIR):
            print(f"Diretório {Config.DATA_DIR} não encontrado.")
            return False

        print("Carregando documentos...")
        loader = DirectoryLoader(
            Config.DATA_DIR, 
            glob="*.txt", 
            loader_cls=TextLoader, 
            loader_kwargs={"encoding": "utf-8"}
        )
        docs = loader.load()
        print(f"{len(docs)} documentos carregados.")
        
        if not docs:
            print("Nenhum documento encontrado.")
            return False

        print("Dividindo documentos...")
        #revisar para chuck semantico
        #revisar para ingestão de milhares de documentos
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        print(f"{len(chunks)} chunks criados.")
        
        print("Gerando embeddings e salvando índice...")
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(Config.VECTOR_STORE_PATH)
        print(f"Índice salvo em {Config.VECTOR_STORE_PATH}")
        return True

if __name__ == "__main__":
    service = IngestionService()
    service.ingest_documents()
