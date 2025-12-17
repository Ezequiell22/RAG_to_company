import os
import json
import random
import time
from typing import List, Dict
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Importar módulos do projeto
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.rag_engine import RAGService
from src.logger import setup_logger

logger = setup_logger("Evaluation")

# Modelos Pydantic para Validação e Estruturação
class QAPair(BaseModel):
    question: str = Field(description="A pergunta gerada baseada no texto")
    answer: str = Field(description="A resposta correta baseada no texto")

class QAList(BaseModel):
    pairs: List[QAPair] = Field(description="Lista de pares de pergunta e resposta")

class EvaluationResult(BaseModel):
    score: int = Field(description="Nota de 0 a 10 para a qualidade da resposta")
    reasoning: str = Field(description="Justificativa breve para a nota atribuída")

class TestSuite:
    def __init__(self):
        self.llm = OllamaLLM(model=Config.LLM_MODEL, temperature=0.7)
        self.rag = RAGService()
        
    def load_random_chunks(self, n_chunks: int = 5) -> List[str]:
        """Carrega chunks diretamente do Vector Store existente para garantir fidelidade ao índice."""
        logger.info("Acessando documentos do índice vetorial carregado...")
        
        try:
            # Tentar acessar o docstore do FAISS diretamente
            # O objeto FAISS do LangChain armazena documentos em self.docstore._dict (InMemoryDocstore)
            if hasattr(self.rag.vector_store, "docstore") and hasattr(self.rag.vector_store.docstore, "_dict"):
                all_docs = list(self.rag.vector_store.docstore._dict.values())
                
                if all_docs:
                    logger.info(f"Recuperados {len(all_docs)} documentos do índice vetorial.")
                    
                    # Filtrar chunks muito pequenos
                    valid_chunks = [doc.page_content for doc in all_docs if len(doc.page_content) > 300]
                    
                    if not valid_chunks:
                        logger.warning("Nenhum chunk > 300 chars encontrado. Usando todos.")
                        valid_chunks = [doc.page_content for doc in all_docs]
                    
                    selected = random.sample(valid_chunks, min(n_chunks, len(valid_chunks)))
                    logger.info(f"{len(selected)} chunks selecionados do VETOR para geração de QA.")
                    return selected
            
            raise ValueError("Não foi possível recuperar documentos do Vector Store. Verifique se o índice foi criado com docstore.")
            
        except Exception as e:
            logger.error(f"Erro fatal ao ler do vetor: {e}")
            raise e

    def generate_qa_pairs(self, chunks: List[str], cases_per_chunk: int = 2) -> List[Dict]:
        """Gera pares de Pergunta/Resposta usando o LLM com OutputParser."""
        logger.info("Gerando casos de teste (QA) via LLM...")
        
        # Configurar Parser
        parser = JsonOutputParser(pydantic_object=QAList)
        
        prompt = PromptTemplate(
            template="""
            Você é um especialista em criar testes de avaliação para sistemas de IA.
            Analise o seguinte texto e crie {count} par(es) de pergunta e resposta que possam ser respondidos APENAS com este texto.
            
            Texto:
            "{text}"
            
            {format_instructions}
            """,
            input_variables=["text", "count"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Chain com Parser
        chain = prompt | self.llm | parser
        
        test_cases = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Gerando QA para chunk {i+1}/{len(chunks)}...")
            try:
                # O parser já retorna um dicionário estruturado (ou objeto QAList dependendo da versão)
                result = chain.invoke({"text": chunk, "count": cases_per_chunk})
                
                # O JsonOutputParser retorna dict. Se fosse PydanticOutputParser retornaria objeto.
                # A estrutura esperada é {'pairs': [{'question': '...', 'answer': '...'}, ...]}
                
                pairs = result.get("pairs", [])
                if not pairs and isinstance(result, list): # Fallback se retornar lista direta
                    pairs = result
                
                for item in pairs:
                    # Garantir que é dict (se o parser retornar objetos Pydantic, converter)
                    if hasattr(item, "dict"):
                        item = item.dict()
                        
                    test_cases.append({
                        "context_snippet": chunk,
                        "question": item.get("question"),
                        "expected_answer": item.get("answer")
                    })
                    
            except Exception as e:
                logger.error(f"Erro ao gerar QA para chunk {i+1}: {e}")
                
        logger.info(f"Total de {len(test_cases)} casos de teste gerados.")
        return test_cases

    def evaluate_answer_with_llm(self, question: str, expected: str, actual: str) -> Dict:
        """Usa o LLM para julgar a qualidade da resposta com OutputParser."""
        
        parser = JsonOutputParser(pydantic_object=EvaluationResult)
        
        judge_prompt = PromptTemplate(
            template="""
            Você é um juiz imparcial avaliando a qualidade de respostas de um sistema de IA.
            Sua tarefa é comparar a RESPOSTA FORNECIDA pelo sistema com a RESPOSTA ESPERADA (Ground Truth).

            Pergunta: {question}

            Resposta Esperada: {expected}

            Resposta Fornecida: {actual}

            Avalie se a Resposta Fornecida contém as informações chave da Resposta Esperada e se responde corretamente à Pergunta.
            - A resposta não precisa ser idêntica palavra por palavra.
            - O significado deve ser preservado.
            - Se a resposta fornecida disser "não sei" ou similar, e a esperada tiver a resposta, a nota deve ser baixa.

            {format_instructions}
            """,
            input_variables=["question", "expected", "actual"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain = judge_prompt | self.llm | parser
        
        try:
            result = chain.invoke({
                "question": question, 
                "expected": expected, 
                "actual": actual
            })
            return result
        except Exception as e:
            logger.error(f"Erro na avaliação do Juiz LLM: {e}")
            return {"score": 0, "reasoning": "Erro na avaliação automática"}

    def run_evaluation(self, test_cases: List[Dict]):
        """Executa os casos de teste contra o RAG e compara resultados."""
        logger.info("Iniciando execução dos testes contra o RAG...")
        
        results = []
        
        for i, case in enumerate(test_cases):
            question = case["question"]
            expected = case["expected_answer"]
            
            logger.info(f"Teste {i+1}/{len(test_cases)}: {question}")
            
            start_time = time.time()
            try:
                # Query no RAG
                actual_answer = self.rag.query(question)
                duration = time.time() - start_time
                
                # Avaliação com LLM-as-a-Judge
                logger.info("Solicitando julgamento do LLM...")
                evaluation = self.evaluate_answer_with_llm(question, expected, actual_answer)
                
                result = {
                    "id": i + 1,
                    "question": question,
                    "expected": expected,
                    "actual": actual_answer,
                    "duration": f"{duration:.2f}s",
                    "score": evaluation.get("score", 0),
                    "reasoning": evaluation.get("reasoning", "N/A"),
                    "status": "Executado"
                }
                results.append(result)
                logger.info(f"Nota: {result['score']}/10 - {result['reasoning']}")
                
            except Exception as e:
                logger.error(f"Erro ao executar teste {i+1}: {e}")
                results.append({
                    "id": i + 1,
                    "question": question,
                    "error": str(e),
                    "status": "Falha",
                    "score": 0,
                    "reasoning": "Erro de execução"
                })

        self.save_report(results)

    def save_report(self, results: List[Dict]):
        """Salva o relatório em JSON e Markdown."""
        os.makedirs("tests/reports", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Calcular média
        scores = [r.get("score", 0) for r in results if r["status"] == "Executado"]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # JSON
        final_data = {
            "summary": {
                "total_cases": len(results),
                "average_score": avg_score,
                "timestamp": timestamp
            },
            "details": results
        }
        
        json_path = f"tests/reports/eval_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)
            
        # Markdown
        md_path = f"tests/reports/eval_{timestamp}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Relatório de Avaliação RAG - {timestamp}\n\n")
            f.write(f"**Total de Casos:** {len(results)}\n")
            f.write(f"**Nota Média:** {avg_score:.1f}/10\n\n")
            
            f.write("| ID | Nota | Pergunta | Resposta Esperada | Resposta RAG | Justificativa |\n")
            f.write("|---|---|---|---|---|---|\n")
            
            for r in results:
                if r["status"] == "Falha":
                    f.write(f"| {r['id']} | 0 | {r['question']} | ERRO | {r['error']} | Erro na execução |\n")
                else:
                    # Truncar textos longos para tabela
                    q_short = (r['question'][:30] + '...') if len(r['question']) > 30 else r['question']
                    exp_short = (r['expected'][:40] + '...') if len(r['expected']) > 40 else r['expected']
                    act_short = (r['actual'][:40] + '...') if len(r['actual']) > 40 else r['actual']
                    reason_short = (r['reasoning'][:50] + '...') if len(r['reasoning']) > 50 else r['reasoning']
                    
                    # Limpar quebras de linha
                    row = [str(x).replace("\n", " ").replace("|", "") for x in [r['id'], r['score'], q_short, exp_short, act_short, reason_short]]
                    
                    f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |\n")
        
        logger.info(f"Relatório salvo em:\n - {json_path}\n - {md_path}")
        print(f"\n✅ Avaliação concluída! Média: {avg_score:.1f}/10. Relatório em: {md_path}")

def main():
    try:
        suite = TestSuite()
        
        # 1. Selecionar chunks (para garantir variedade)
        chunks = suite.load_random_chunks(n_chunks=5) # 5 chunks
        
        # 2. Gerar Perguntas (2 por chunk = 10 casos)
        test_cases = suite.generate_qa_pairs(chunks, cases_per_chunk=2)
        
        # 3. Limitar a 10 casos se passar
        test_cases = test_cases[:10]
        
        if not test_cases:
            logger.error("Nenhum caso de teste foi gerado.")
            return

        # 4. Executar Avaliação
        suite.run_evaluation(test_cases)
        
    except Exception as e:
        logger.critical(f"Erro fatal na suite de testes: {e}")

if __name__ == "__main__":
    main()
