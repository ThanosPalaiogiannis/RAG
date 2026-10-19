from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from rag.rag import answer_query
from rag.exceptions import RAGServiceError,RAGConfigurationError
import logging

logger = logging.getLogger(__name__)
app = FastAPI()

class AskRequest(BaseModel):
    query: str = Field(min_length=1)

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value):
        if isinstance(value, str):
            return value.strip()

        return value

class SourceResponse(BaseModel):
    url: str
    section: str | None
    language: str | None

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = answer_query(request.query)
        return result

    except RAGServiceError as err:
        logger.exception("RAG request failed")

        raise HTTPException(
            status_code=503,
            detail="RAG service temporarily unavailable"
        ) from err

    except RAGConfigurationError as err:
        logger.exception("RAG configuration failed")

        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        ) from err
