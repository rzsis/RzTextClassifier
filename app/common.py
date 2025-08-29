# common.py
from pathlib import Path
from datetime import datetime
import time
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from starlette.responses import JSONResponse
from logging.handlers import RotatingFileHandler
import logger

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


