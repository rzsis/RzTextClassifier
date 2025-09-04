# common.py
from pathlib import Path
from datetime import datetime
import time
from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from starlette.responses import JSONResponse
import logger
import db_utils
import localconfig

# Global variables for dependency injection
_db = None
_localconfig = None

last_print_time = time.time()

def print_with_time(message):
    global last_print_time
    current_time = time.time()
    elapsed_time = current_time - last_print_time
    logger.info(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] (Decorrido: {elapsed_time:.2f}s) {message}")
    last_print_time = current_time

def print_error(message):
    global last_print_time
    current_time = time.time()
    elapsed_time = current_time - last_print_time
    logger.error(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}")
    last_print_time = current_time


def init_dependencies(appName: str):
    global _localconfig, _db
    localconfig.load_config(appName)         
    _localconfig = localconfig
    _db = db_utils.Db(localconfig)   # cria engine + sessionmaker UMA vez
    

def get_db():
    """
    Dependency function to provide a Session (não o objeto Db).
    """
    if _db is None:
        raise RuntimeError("Db not initialized. Call init_dependencies first.")
    session = _db.get_session()
    try:
        yield session
    finally:
        session.close()

def get_localconfig():
    if _localconfig is None:
        raise RuntimeError("localconfig not initialized in common")
    return _localconfig