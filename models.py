from pydantic import BaseModel
from typing import List, Optional

# /chat request 모델
class ChatRequest(BaseModel):
    user_id: int
    conversation_id: int
    question: str
    character_id: int

class BalanceChatRequest(BaseModel):
    user_id: int
    conversation_id: int
    question: str
    character_id: int  
    keyword: Optional[str] = None
    situation: Optional[str] = None  

# /character/match request 모델
class CharacterMatchRequest(BaseModel):
    question: str
    char_id_list: List[int]
    chat_history_list: List[str] = []

class LoadInfoRequest(BaseModel):
    char_id_list: List[int]

class CharacterMatchResponse(BaseModel):
    selected_char_id_list: List[int]

# /chat response 모델
class ChatResponse(BaseModel):
    answer: str
    character_id: int
    msg_img: Optional[int] = None