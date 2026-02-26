# fastApi_Utils.py

from fastapi import FastAPI
from contextlib import asynccontextmanager

import logger
import common
import dbClasses.classes_utils as classes_utils
import controllers.text_controller as text_controller
import controllers.index_controller as index_controller
import controllers.edit_text_contoller as edit_text_contoller
import controllers.utils_controller as utils_controller
from controllers import finetuning_contoller

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        common.print_with_time("Inicializando classes_utils por worker")
        common._db.test_connection()            
        classes_utils.initClassesUtils(common._db.get_session())
    finally:
        yield
        common.print_with_time("Worker finalizado")


class FastApi_Utils:
    def __init__(self, app_name: str):
        self.app_name = app_name

    def create_app(self) -> FastAPI:
        # lifespan é o mecanismo oficial do FastAPI (ASGI) para executar código de inicialização e finalização da aplicação por processo.
        # Ele é especialmente útil para cenários como o nosso, onde precisamos garantir que certas inicializações (como a configuração de classes_utils) 
        # sejam feitas para cada processo worker, garantindo que cada um tenha seu próprio contexto e conexões adequadas.      

        app = FastAPI(lifespan=lifespan)

        # Exception handlers
        logger.setup_global_exception_logging(app)

        # 4) Registra os controllers (APIRouter)
        app.include_router(text_controller.router)
        app.include_router(index_controller.router)
        app.include_router(edit_text_contoller.router)
        app.include_router(finetuning_contoller.router)
        app.include_router(utils_controller.router)

        return app