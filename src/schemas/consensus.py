from typing import List
from pydantic import BaseModel

class ModelPrediction(BaseModel):
    model_name: str
    required_elements: List[str]

class ConsensusOutput(BaseModel):
    prompt: str
    consensus_elements: List[str]
    individual_predictions: List[ModelPrediction]