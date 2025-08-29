# common.py
from pathlib import Path
import os
import logging
import sys
from datetime import datetime
import threading
import asyncio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from starlette.responses import JSONResponse
from logging.handlers import RotatingFileHandler


log : logging.Logger
# -----------------------------
# Logger único da aplicação
# -----------------------------
def build_logger(appName: str) -> logging.Logger:    
    global log 
    log = logging.getLogger("app")    
    if log.handlers:  # evita duplicar handlers em reload
        return log
    
    log_path = f"../log/"
    os.makedirs(log_path, exist_ok=True)
    log_fileName = f"{appName}.log"

    handler = RotatingFileHandler(
        log_fileName,
        maxBytes=2 * 1024 * 1024,  # 5 MB
        backupCount=5,             # mantém até 5 arquivos antigos
        encoding="utf-8"
    )    

    logfilename = f"{log_path}{log_fileName}"

    log.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logfilename, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("%(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    log.addHandler(fh)
    log.addHandler(ch)
    log.propagate = False

    return log


# -----------------------------
# Decorator opcional para funções críticas
# -----------------------------
def log_errors(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            raise Exception(f"Erro em {fn.__name__}")
            raise
    return wrapper

def _getDateTimeStr() -> str:
    return f"{datetime.now():%Y-%m-%d %H:%M:%S}"


# -----------------------------
# Hooks globais de exceção
# -----------------------------
def _sys_excepthook(exc_type, exc, tb):
    # pega qualquer exceção não tratada em threads principais
    log.exception(f"{_getDateTimeStr()} Exceção não tratada", exc_info=(exc_type, exc, tb))

def _threading_excepthook(args: threading.ExceptHookArgs):
    log.exception(f"{_getDateTimeStr()} Exceção não tratada em thread", exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

def _asyncio_excepthook(loop, context):
    msg = context.get("exception")
    if msg:
        log.exception(f"{_getDateTimeStr()} Exceção não tratada no asyncio", exc_info=msg)
    else:
        log.error(f"{_getDateTimeStr()} Erro no asyncio: {context.get('message')}")

def setup_global_exception_logging(app: FastAPI | None = None):
    # sys / threading
    sys.excepthook = _sys_excepthook
    threading.excepthook = _threading_excepthook

    # asyncio
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(_asyncio_excepthook)
    except RuntimeError:
        # sem loop no momento (ok em import time)
        pass

    if app:
        # Handler genérico de qualquer Exception (500)
        @app.exception_handler(Exception)
        async def all_exception_handler(request: Request, exc: Exception):
            log.exception(f"{_getDateTimeStr()} Unhandled error em {request.method} {request.url.path}")
            return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

        # HTTPException (4xx/5xx controladas)
        @app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            # 4xx geralmente como warning/erro leve; 5xx como erro
            level = logging.WARNING if 400 <= exc.status_code < 500 else logging.ERROR
            log.log(level, f"{_getDateTimeStr()} HTTPException {exc.status_code} em {request.method} {request.url.path}: {exc.detail}")
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        # Erros de validação (422)
        @app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            log.warning(f"{_getDateTimeStr()} ValidationError em {request.method} {request.url.path}: {exc.errors()}")
            return JSONResponse(status_code=422, content={"detail": exc.errors()})
