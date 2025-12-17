import logging
import sys
import time
import os
from functools import wraps
from logging.handlers import RotatingFileHandler

# Criar diretório de logs se não existir
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Configuração do Logger
def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Evitar duplicidade de handlers
    if logger.hasHandlers():
        return logger

    # Formato do log
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para Arquivo (com rotação: 5MB por arquivo, max 5 arquivos)
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "app.log"), 
        maxBytes=5*1024*1024, 
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler para Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Decorator para medir tempo de execução
def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Execução de '{func.__name__}' finalizada em {execution_time:.4f} segundos.")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Erro em '{func.__name__}' após {execution_time:.4f}s: {str(e)}")
            raise e
            
    return wrapper

# Logger global
logger = setup_logger("Global")
