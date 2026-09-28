from dataclasses import dataclass
from rag.context.source import RetrievedSource


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievedSource]
    

    
