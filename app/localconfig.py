# localconfig.py
import os
import json
from pathlib import Path
import logger

__configReaded = {}
__db_config = {}
__logger: None

def load_config(appName: str ):
    global __configReaded, __logger

    config_path = os.path.join(f"../{appName}.config")
    logger.log.info(f"Lendo arquivo de configuração: {config_path}")

    if not os.path.isfile(config_path):
        # lançar exceção (será capturada pelos hooks)
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    with open(config_path, "r") as f:
        __configReaded = json.load(f)

    checkRequiredKeys()
    load_db_config()

def checkRequiredKeys():
    required_keys = [
        "DataBaseConnectString",
        "model_path",        
        "embeddingsFinal",
        "embeddingsTrain",
        "http_port",
        "dataset_path",
        "batch_size",
        "max_length",
        "vectordatabasehost",
        "codcli"
    ]

    missing_keys = [key for key in required_keys if key not in __configReaded]
    if missing_keys:
        raise KeyError(f"Chaves obrigatórias ausentes no arquivo de configuração: {', '.join(missing_keys)}")

def load_db_config():
    global __db_config
    if not __configReaded:
        raise RuntimeError("Config ainda não foi carregada. Chame load_config() antes.")

    conn_str = __configReaded.get("DataBaseConnectString", "")
    if conn_str:
        parts = [item.strip() for item in conn_str.split(";") if item.strip()]
        __db_config = dict([p.strip() for p in part.split("=", 1)] for part in parts)
        __db_config = {k.strip().lower(): v.strip() for k, v in __db_config.items()}

def read_config():
    return __configReaded

def get_db_server():
    return __db_config.get("server")

def get_db_port():
    return int(__db_config.get("port", 0)) if __db_config.get("port") else None

def get_db_name():
    return __db_config.get("database")

def get_db_user():
    return __db_config.get("uid") or __db_config.get("user")

def get_db_password():
    return __db_config.get("password")

def getModelName() -> str:
    model_path = Path(__configReaded["model_path"])
    return model_path.name

def getModelPath() -> Path:
    model_path = Path(__configReaded["model_path"])
    return model_path

def getEmbendingFinal() -> str:
    return os.path.join(__configReaded["embeddingsFinal"], getModelName())

def getEmbeddingsTrain() -> str:
    return os.path.join(__configReaded["embeddingsTrain"], getModelName())

def get(chave) -> str:
    return __configReaded[chave]

