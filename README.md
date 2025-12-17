# Projeto RAG Corporativo com Ollama e LangChain

Este projeto implementa um sistema de **RAG (Retrieval-Augmented Generation)** robusto e modular, capaz de responder perguntas com base em documentos internos. Ele utiliza **Ollama** para execução local de LLMs (Large Language Models) e **LangChain** para orquestração.

## 🚀 Funcionalidades

*   **Ingestão de Documentos**: Processamento automático de arquivos `.txt` da pasta `data/`.
*   **Busca Semântica**: Utiliza **FAISS** como banco vetorial e embeddings `nomic-embed-text`.
*   **LLM Local**: Respostas geradas pelo modelo **Gemma 3 (4b)** rodando via Ollama.
*   **API REST (FastAPI)**: Interface de alta performance com endpoints para consulta e recarga de índice.
*   **Cache Inteligente**: Sistema de cache com TTL de 1 hora para respostas instantâneas a perguntas repetidas.
*   **Arquitetura Modular**: Código organizado em serviços (`RAGService`, `IngestionService`) e configuração centralizada.
*   **Monitoramento**: Logs detalhados com rotação de arquivos e métricas de tempo de execução.

## 🛠️ Arquitetura

O projeto segue uma arquitetura modular em camadas:

*   **`src/config.py`**: Configurações globais (modelos, caminhos, parâmetros).
*   **`src/rag_engine.py`**: Motor principal. Implementa Singleton para manter o modelo em memória e gerencia o fluxo RAG (Retriever -> Prompt -> LLM). Inclui camada de cache.
*   **`src/ingestor.py`**: Serviço responsável por ler documentos, dividir em chunks e criar o índice vetorial.
*   **`src/logger.py`**: Sistema de logs centralizado.
*   **`api.py`**: Servidor FastAPI que expõe o `RAGService`.
*   **`main.py`**: Interface de linha de comando (CLI) para testes e ingestão.
*   **`crawler.py`**: Ferramenta auxiliar para baixar conteúdo de sites.

## 📦 Instalação

### Pré-requisitos
*   Python 3.10+
*   [Ollama](https://ollama.com/) instalado e rodando.

### 1. Configurar Ambiente

Crie um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Configurar Modelos Ollama

Certifique-se de que o Ollama está rodando (`ollama serve`) e baixe os modelos necessários:

```bash
ollama pull nomic-embed-text  # Para embeddings (rápido e leve)
ollama pull gemma3:4b         # LLM para geração de texto
```

## 🏃‍♂️ Como Usar

### Opção 1: API REST (Recomendado para Produção)

A API mantém o modelo carregado na memória, garantindo respostas rápidas.

1.  **Inicie o servidor:**
    ```bash
    uvicorn api:app --reload
    ```
    *O servidor iniciará em `http://localhost:8000`.*

2.  **Faça perguntas (Exemplo com cURL):**
    ```bash
    curl -X POST "http://127.0.0.1:8000/query" \
         -H "Content-Type: application/json" \
         -d '{"question": "Quais serviços a empresa oferece?"}'
    ```

3.  **Recarregar Índice (após adicionar novos arquivos):**
    ```bash
    curl -X POST "http://127.0.0.1:8000/reload"
    ```

### Opção 2: Linha de Comando (CLI)

Útil para testes rápidos ou scripts de automação.

*   **Ingestão de Dados** (Processar arquivos da pasta `data/`):
    ```bash
    python main.py ingest
    ```

*   **Chat Interativo**:
    ```bash
    python main.py chat
    ```

*   **Pergunta Única**:
    ```bash
    python main.py query "Qual é a visão da empresa?"
    ```

### Opção 3: Crawler (Coleta de Dados)

Para baixar conteúdo de um site e salvar na pasta `data/`:

```bash
python crawler.py --url "https://exemplo.com.br" --depth 2
```

## 📊 Logs e Monitoramento

Os logs são salvos automaticamente na pasta `logs/` e também exibidos no console.
*   **Arquivo**: `logs/app.log` (Rotacionado automaticamente, máx 5 arquivos de 5MB).
*   **Conteúdo**: Detalhes de inicialização, tempo de resposta de cada etapa, erros e status de cache.

## ⚙️ Personalização

Você pode ajustar parâmetros no arquivo `src/config.py`:
*   `CHUNK_SIZE`: Tamanho dos pedaços de texto.
*   `RETRIEVER_K`: Quantidade de trechos de contexto recuperados.
*   `CACHE_TTL`: Tempo de vida do cache (padrão: 3600s).
*   `LLM_MODEL`: Modelo Ollama a ser utilizado.
