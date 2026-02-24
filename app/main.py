# main.py
import os

from controllers import edit_text_contoller, finetuning_contoller
#Necessario colocar ja inicio para pegar antes de importar torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Makes errors immediate
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enables device-side assertions

from operator import index
from fastapi import FastAPI
import controllers.text_controller as text_controller
import controllers.index_controller as index_controller
import controllers.edit_text_contoller as edit_text_contoller
import controllers.utils_controller as utils_controller
import localconfig
import common
import logger
import bll.embeddingsBll as bllEmbeddings
from gpu_utils import GpuUtils  as gpu_utilsModule
import bll.onxx_utils.export_bge_m3_to_onnx as bllExportBgeM3ToOnnx   
import dbClasses.classes_utils


#os.system("clear")

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
fastApi.include_router(text_controller.router)
fastApi.include_router(index_controller.router)
fastApi.include_router(edit_text_contoller.router)
fastApi.include_router(finetuning_contoller.router)
fastApi.include_router(utils_controller.router)
common._db.test_connection()

gpu_utilsModule().print_gpu_info()


if __name__ == "__main__":

    # Melhorias
    # Garantir que ao exportar para ONNX o modelo seja o bge original e não o convertido ONNX
    # Gerar um readme com explicacoes do modelo exportado


    #bllExportBgeM3ToOnnx.execute()  #  isso foi colocado aqui para gerar facilmente o onnx não deve ser usado em produção

    import uvicorn
    HTTP_PORT = int(localconfig.get("http_port"))
    workers = int(localconfig.get("workers"))
    common.print_with_time(f"Iniciando {appName} na porta {HTTP_PORT}")    
    common.print_with_time(f"Banco vetorial de destino -> {localconfig.get('vectordatabasehost')}")
    dbClasses.classes_utils.initClassesUtils(common._db.get_session())  # Inicializa o singleton de classes_utilsBLL com a sessão do banco de dados
    
    uvicorn.run(
        "main:fastApi",          
        host="0.0.0.0",
        port=HTTP_PORT,
        log_level="info",
        workers=workers,
    )
