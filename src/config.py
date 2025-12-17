import os

class Config:
    # Caminhos
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    VECTOR_STORE_PATH = os.path.join(BASE_DIR, "faiss_index")
    
    # Modelos
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "gemma3:4b"
    
    # Parâmetros de Ingestão
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Parâmetros de Busca
    RETRIEVER_K = 3
    
    # Cache
    CACHE_TTL = 3600  # 1 hora em segundos
    
    # Prompt
    REWRITE_PROMPT = """
    Reescreva a pergunta do usuário de forma mais clara e concisa.
    Só retorne a pergunta reescrita.
    Pergunta original: {question}
    """

    RAG_TEMPLATE = """
    Você é um assistente especialista e útil. Use o contexto abaixo para responder à pergunta do usuário.
    Se a resposta não estiver no contexto, diga claramente que não sabe. Não invente informações.
    
    Contexto:
    {context}
    
    Pergunta:
    {question}
    """
