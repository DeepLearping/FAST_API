from pydantic import BaseModel
from typing import List, Optional

# /chat request 모델
class ChatRequest(BaseModel):
    user_id: int
    conversation_id: int
    question: str
    character_id: int

# /character/match request 모델
class CharacterMatchRequest(BaseModel):
    question: str
    char_id_list: List[int]

# /chat response 모델
class ChatResponse(BaseModel):
    answer: str
    character_id: int
    msg_img: Optional[int] = None