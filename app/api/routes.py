import logging
import time

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.response_builder import get_answer

logger = logging.getLogger(__name__)
router = APIRouter()


class AskRequest(BaseModel):
    text: str
    user_id: int | None = None


class AskResponse(BaseModel):
    answer: str


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    t0 = time.perf_counter()
    logger.info("=> /ask user_id=%s text=%r", req.user_id, req.text[:120])
    answer = get_answer(req.text, user_id=req.user_id)
    logger.info("<= /ask user_id=%s | %.0f ms", req.user_id, (time.perf_counter() - t0) * 1000)
    return AskResponse(answer=answer)
