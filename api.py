from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from src.rag_engine import RAGService
from src.logger import setup_logger
import uvicorn
import time

# Configurar logger para a API
logger = setup_logger("API")

app = FastAPI(title="API RAG Corporativo", description="API para responder perguntas com base em documentos internos.")

# Middleware para logging de requisições
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"Recebendo requisição: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Requisição processada em {process_time:.4f}s - Status: {response.status_code}")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"Erro na requisição após {process_time:.4f}s: {str(e)}")
        raise e

# Inicializar o serviço RAG na inicialização
logger.info("Inicializando API e RAG Service...")
try:
    rag_service = RAGService()
    logger.info("RAG Service inicializado com sucesso.")
except Exception as e:
    logger.critical(f"Falha crítica ao inicializar RAG Service: {str(e)}")
    # Não encerramos aqui para permitir diagnósticos, mas o serviço ficará indisponível
    rag_service = None

# Modelo de dados para a requisição
class QueryRequest(BaseModel):
    question: str

# Modelo de resposta
class QueryResponse(BaseModel):
    answer: str

@app.get("/")
def read_root():
    logger.info("Endpoint raiz acessado.")
    return {"status": "online", "message": "Bem-vindo à API RAG. Use o endpoint /query para fazer perguntas."}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    """
    Endpoint para fazer perguntas ao sistema RAG.
    """
    if rag_service is None:
        logger.error("Tentativa de query com RAG Service não inicializado.")
        raise HTTPException(status_code=503, detail="Serviço RAG indisponível.")

    if not request.question.strip():
        logger.warning("Recebida pergunta vazia.")
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")
    
    logger.info(f"Processando query: {request.question[:50]}...")
    try:
        response = rag_service.query(request.question)
        return QueryResponse(answer=response)
    except Exception as e:
        logger.error(f"Erro ao processar query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_index():
    """
    Recarrega o índice vetorial (útil após nova ingestão de dados).
    """
    if rag_service is None:
         raise HTTPException(status_code=503, detail="Serviço RAG indisponível.")
         
    logger.info("Endpoint /reload acessado.")
    try:
        rag_service.reload_index()
        return {"status": "success", "message": "Índice recarregado com sucesso."}
    except Exception as e:
        logger.error(f"Erro ao recarregar índice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Iniciando servidor Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
