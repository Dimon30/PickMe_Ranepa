import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

ENV_FILE = os.getenv("ENV_FILE", ".env")
load_dotenv(ENV_FILE)

from fastapi import FastAPI

from app.logging_config import setup_logging
setup_logging()

from app.api.routes import router as ask_router
from app.db.session import test_connection
from app.dependencies import warmup


@asynccontextmanager
async def lifespan(app: FastAPI):
    warmup()
    yield


app = FastAPI(title="PickMe Ranepa API", lifespan=lifespan)

app.include_router(ask_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-check")
def db_check():
    return {"db": test_connection()}
