from pydantic import BaseModel
from typing import List

class Section(BaseModel):
    title: str
    content: str

class Question(BaseModel):
    question: str