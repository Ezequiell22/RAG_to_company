import sys
from src.rag_engine import RAGService
from src.ingestor import IngestionService

def print_help():
    print("""
    Uso:
    python main.py [comando]
    
    Comandos:
    chat            -> Inicia o chat interativo (modo padrão)
    ingest          -> Processa os documentos e atualiza o banco vetorial
    query "pergunta" -> Faz uma pergunta única e sai
    """)

def run_chat():
    print("\n=== Sistema RAG Corporativo ===")
    rag = RAGService()  # Carrega o modelo APENAS UMA VEZ aqui
    
    print("\nDigite 'sair' para encerrar.")
    while True:
        try:
            user_input = input("\nPergunta: ")
            if user_input.lower() in ['sair', 'exit', 'quit']:
                break
            if not user_input.strip():
                continue
                
            print("Processando...")
            response = rag.query(user_input)
            print(f"Resposta:\n{response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erro: {e}")

def run_ingest():
    service = IngestionService()
    service.ingest_documents()

def main():
    if len(sys.argv) < 2:
        run_chat()
        return

    command = sys.argv[1].lower()

    if command == "ingest":
        run_ingest()
    elif command == "chat":
        run_chat()
    elif command == "query":
        if len(sys.argv) < 3:
            print("Erro: Forneça a pergunta entre aspas.")
            return
        rag = RAGService()
        print(rag.query(sys.argv[2]))
    else:
        print_help()

if __name__ == "__main__":
    main()
