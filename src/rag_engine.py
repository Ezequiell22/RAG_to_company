import os
import time
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from src.config import Config
from src.logger import setup_logger, measure_time

logger = setup_logger("RAGService")

class RAGService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    @measure_time
    def __init__(self):
        if self._initialized:
            logger.info("RAG Service já inicializado (Singleton).")
            return
            
        logger.info("Iniciando inicialização do RAG Service...")
        
        # Inicializa o cache
        self._cache = {}
        
        logger.info(f"Carregando embeddings: {Config.EMBEDDING_MODEL}")
        self.embeddings = OllamaEmbeddings(model=Config.EMBEDDING_MODEL)
        
        self.vector_store = self._load_vector_store()
        
        logger.info(f"Carregando LLM: {Config.LLM_MODEL}")
        self.llm = OllamaLLM(model=Config.LLM_MODEL)
        
        logger.info("Construindo Chain...")
        self.chain = self._build_chain()
        
        self._initialized = True
        logger.info("RAG Service inicializado com sucesso.")

    @measure_time
    def _load_vector_store(self):
        logger.info(f"Carregando índice FAISS de {Config.VECTOR_STORE_PATH}...")
        if not os.path.exists(Config.VECTOR_STORE_PATH):
            logger.error(f"Índice FAISS não encontrado em {Config.VECTOR_STORE_PATH}")
            raise FileNotFoundError(f"Índice FAISS não encontrado em {Config.VECTOR_STORE_PATH}. Execute o ingestor primeiro.")
        
        store = FAISS.load_local(
            Config.VECTOR_STORE_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.info("Índice FAISS carregado.")
        return store

    def _build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})
        prompt = ChatPromptTemplate.from_template(Config.RAG_TEMPLATE)
        parser = StrOutputParser()
        
        return (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | parser
        )

    @measure_time
    def rewrite_question(self, question: str) -> str:
        """Reescreve a pergunta do usuário para otimizar a busca."""
        prompt = ChatPromptTemplate.from_template(Config.REWRITE_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question})

    @measure_time
    def query(self, question: str) -> str:
        logger.info(f"Recebendo pergunta: '{question}'")
        
        # Verificar Cache
        current_time = time.time()
        if question in self._cache:
            response, timestamp = self._cache[question]
            if current_time - timestamp < Config.CACHE_TTL:
                logger.info("Cache HIT: Retornando resposta armazenada.")
                return response
            else:
                logger.info("Cache EXPIRED: O cache para esta pergunta expirou.")
                del self._cache[question]
        else:
            logger.info("Cache MISS: Processando nova pergunta.")
            
        # Executar Chain
        logger.info(f"Pergunta original: '{question}'")
        question_rewrite = self.rewrite_question(question)

        logger.info(f"Pergunta reescrita: '{question_rewrite}'")

        response = self.chain.invoke(question_rewrite)
        
        # Salvar no Cache
        self._cache[question] = (response, current_time)
        logger.info("Resposta gerada e armazenada no cache.")
        
        return response

    @measure_time
    def reload_index(self):
        """Recarrega o índice do disco e limpa o cache."""
        logger.info("Solicitação de recarga de índice...")
        self.vector_store = self._load_vector_store()
        self.chain = self._build_chain()
        self._cache.clear()  # Importante limpar o cache se os dados mudaram
        logger.info("Índice recarregado e cache limpo com sucesso.")
