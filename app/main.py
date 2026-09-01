import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import router
from app.db import init_db

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Construction Video Analysis POC", lifespan=lifespan)
app.include_router(router)
