
from pydantic import BaseModel
from typing import List, Dict, Optional

class ModelConfig(BaseModel):
    endpoint: str
    model_name: str
    name: Optional[str] = None  # purely optional, for your labeling
    api_key: Optional[str] = None  # required for Anthropic, ignored for Azure

class MultiTurnAttackRequest(BaseModel):
    objectives: List[str]
    models: List[ModelConfig]
    memory_labels: Dict[str, str] = {}
    max_concurrency: int = 5

class ResultPiece(BaseModel):
    id: str
    conversation_id: Optional[str]
    role: str
    value: str
    error: str
    model_name: Optional[str] = None
