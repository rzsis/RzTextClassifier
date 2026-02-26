# main.py
import os
#Necessario colocar ja inicio para pegar antes de importar torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Makes errors immediate
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Enables device-side assertions
import localconfig
import common
import logger
import bll.embeddingsBll as bllEmbeddings
from gpu_utils import GpuUtils  as gpu_utilsModule
from fastApi_Utils import FastApi_Utils
import uvicorn

#os.system("clear")

appName = "RzTextClassifier"
# 1) Instala os hooks globais o quanto antes
logger.setup_global_exception_logging()  # sem app, já cobre sys/threading/asyncio

#Configura o logger
logger.build_logger(appName)

# Inicializa arquivo de configuração
common.init_dependencies(appName)

# 3) Cria o app e registra handlers do FastAPI
# Cria app no nível do módulo (ESSENCIAL)
factory = FastApi_Utils(appName)
fastApi = factory.create_app()

gpu_utilsModule().print_gpu_info()



if __name__ == "__main__":
    # Melhorias
    # Garantir que ao exportar para ONNX o modelo seja o bge original e não o convertido ONNX
    # Gerar um readme com explicacoes do modelo exportado


    #bllExportBgeM3ToOnnx.execute()  #  isso foi colocado aqui para gerar facilmente o onnx não deve ser usado em produção

    common.print_with_time(f"Banco vetorial de destino -> {localconfig.get('vectordatabasehost')}")

    uvicorn.run(
        "main:fastApi",
        host="0.0.0.0",
        port=int(localconfig.get("http_port")),
        workers=int(localconfig.get("workers")),
        log_level="info",
    )
