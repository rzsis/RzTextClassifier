# main.py (corrigido)
from operator import index
from fastapi import FastAPI
import controllers.textController as textController
import controllers.indexController as indexController
import localconfig
import common
import logger
import db_utils
import bll.modelos as modelos

appName = "RzTextClassifier"
# 1) Instala os hooks globais o quanto antes
logger.setup_global_exception_logging()  # sem app, já cobre sys/threading/asyncio

#Configura o logger
logger.build_logger(appName)

# Inicializa arquivo de configuração
localconfig.load_config(appName)

# 3) Cria o app e registra handlers do FastAPI
fastApi = FastAPI()
logger.setup_global_exception_logging(fastApi)  # adiciona os exception handlers do FastAPI

fastApi.include_router(textController.router)
fastApi.include_router(indexController.router)


if __name__ == "__main__":
    import uvicorn

    HTTP_PORT = int(localconfig.read_config().get("http_port"))
    db_utils.Db(localconfig)  # tenta conectar com o banco de dados

    modelo = modelos.Modelos(localconfig)  # inicializa modelos (carrega embeddings)
    modelo.load_model_and_embendings()

    
    common.print_with_time(f"Iniciando {appName} na porta {HTTP_PORT}")    
    uvicorn.run(fastApi, host="0.0.0.0", port=HTTP_PORT, log_level="info")

