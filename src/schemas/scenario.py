from pydantic import BaseModel
from typing import Optional


class ImageGenerationOutput(BaseModel):
    id: str
    image_url: str
    prompt_used: str


class ImageMeta(BaseModel):
    image_id: str
    filename: str
    relpath: str
    prompt_used: str
    model: Optional[str] = None
    seed: Optional[int] = None