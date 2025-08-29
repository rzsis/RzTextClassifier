# main.py (corrigido)
from fastapi import FastAPI
import controllers.textController as textController
import localconfig
import common
import logger

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

if __name__ == "__main__":
    import uvicorn
    cfg = localconfig.read_config()
    HTTP_PORT = int(cfg["http_port"])
    uvicorn.run(fastApi, host="0.0.0.0", port=HTTP_PORT, log_level="info")
