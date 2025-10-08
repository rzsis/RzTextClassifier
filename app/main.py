# main.py
import os
#Necessario colocar ja inicio para pegar antes de importar torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Makes errors immediate
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enables device-side assertions
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["ATTN_IMPLEMENTATION"] = "eager"  # antes de importar transformers

from operator import index
from fastapi import FastAPI
import controllers.textController as textController
import controllers.indexController as indexController
import controllers.embeddings_Generate_Controller as embeddings_Generate_Controller
import localconfig
import common
import logger
import bll.embeddingsBll as bllEmbeddings
from gpu_utils import GpuUtils  as gpu_utilsModule

appName = "RzTextClassifier"
# 1) Instala os hooks globais o quanto antes
logger.setup_global_exception_logging()  # sem app, já cobre sys/threading/asyncio

#Configura o logger
logger.build_logger(appName)

# Inicializa arquivo de configuração
common.init_dependencies(appName)

# 3) Cria o app e registra handlers do FastAPI
fastApi = FastAPI()
logger.setup_global_exception_logging(fastApi)  # adiciona os exception handlers do FastAPI

# 4) Registra os controllers (APIRouter)
fastApi.include_router(textController.router)
fastApi.include_router(indexController.router)
fastApi.include_router(embeddings_Generate_Controller.router)

if (common._db.test_connection() is not None):
    common.print_with_time("Conexão com banco de dados estabelecida com sucesso")


gpu_utilsModule().print_gpu_info()

if __name__ == "__main__":
    import uvicorn
    HTTP_PORT = int(localconfig.read_config().get("http_port"))
    common.print_with_time(f"Iniciando {appName} na porta {HTTP_PORT}")    
    
    uvicorn.run(fastApi, host="0.0.0.0", port=HTTP_PORT, log_level="info")

