from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_redis import RedisChatMessageHistory
from chat_logic import setup_character_matching_prompt, setup_chat_chain
from models import CharacterMatchRequest, ChatRequest, ChatResponse
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage
import os
from sqlalchemy import create_engine, text
import json
import re

app = FastAPI()

DATABASE_URL = os.getenv("ENV_CONNECTION")
engine = create_engine(DATABASE_URL)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 캐릭터와 채팅
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # import time
        # start_time = time.time()
        chat_chain = setup_chat_chain(request.character_id) # chain을 캐릭터에 따라 set
        # print("chat chain time", time.time() - start_time)

        config = {
            "configurable": {
                "user_id": request.user_id,
                "conversation_id": request.conversation_id
            }
        }

        response = chat_chain.invoke({"question": request.question}, config)
        
        # 토큰 단위 스트리밍
        # response = ""
        # for token in chat_chain.stream({"question": request.question}, config):
        #     # 스트림에서 받은 데이터의 내용을 출력
        #     # 줄바꿈 없이 이어서 출력, 버퍼를 즉시 비움
        #     response = response + token
        #     print(token, end="", flush=True)

        # chat_message2에 새로운 table에 캐릭터 name과 id 포함된 message 저장
        # history = SQLChatMessageHistory(table_name="chat_message2",session_id=request.conversation_id,connection=os.getenv("ENV_CONNECTION"))
        # history.add_user_message(HumanMessage(content=request.question,id=request.user_id))
        # history.add_ai_message(AIMessage(content=response,id=request.character_id))

        # 응답(response)에서 키워드 감지 및 이미지 URL 매핑
        detected_keyword = query_routing(response)  # 응답 내용을 분석
        msg_img= get_image_url(detected_keyword)  # 키워드에 해당하는 이미지 URL 가져오기

        return ChatResponse(
            answer=response,
            character_id=request.character_id,
            msg_img=msg_img
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 단체방에서 사용자가 질문을 받아 어떤 캐릭터가 응답하기에 적합한지 결정하여 캐릭터id 리스트 반환
@app.post("/character/match")
async def match_character(request: CharacterMatchRequest):
    try:
        question = request.question
        char_id_list = request.char_id_list

        character_info = [
            f"{char_id}: {get_character_info_by_id(char_id)}"
            for char_id in char_id_list
        ]
        formatted_character_info = "\n".join(character_info)
        
        prompt = setup_character_matching_prompt()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        result = llm.invoke(
            prompt.format(question=question, character_info=formatted_character_info)
        )
        # print("선택된 캐릭터들: ",result.content)

        # 캐릭터 ID 리스트로 반환
        numeric_ids = re.findall(r'\b\d+\b', result.content)    # 숫자(정수)만 추출
        matching_characters = [int(char_id) for char_id in numeric_ids]
        return {"matching_characters": matching_characters}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_character_info_by_id(character_id: int) -> str:
    character_descriptions = {
        6: "스폰지밥 (SpongeBob SquarePants) - A cheerful sea sponge living in 비키니 시티, loves jellyfishing and working at the 집게리아. (From *SpongeBob SquarePants*)",
        5: "플랑크톤 (Plankton) - A scheming microbe from 비키니 시티 who often plots to steal the 게살버거 formula. (From *SpongeBob SquarePants*)",
        1: "버즈 (Buzz Lightyear) - A space ranger toy from the *Toy Story* universe, brave and adventurous. (From *Toy Story*)",
        4: "김전일 (Kindaichi) - A high school detective with exceptional reasoning skills, often solving complex murder cases. (From *Kindaichi Case Files*)",
        3: "리바이 (Levi Ackerman) - A skilled soldier and captain of the Survey Corps from *Attack on Titan*, known for his agility, precision, and cold demeanor.",
        2: "에스카노르 (Escanor) - The Lion's Sin of Pride from *Seven Deadly Sins*, confident and powerful during the day, timid at night."
    }
    return character_descriptions.get(character_id, f"존재하지 않는 캐릭터 번호: {character_id}")

def query_routing(response: str) -> str:    # 응답 내용에서 키워드를 감지하는 함수
    keywords = ["기뻐", "슬퍼"]  # 감지하려는 키워드 목록
    for keyword in keywords:
        if keyword in response.lower():
            return keyword
    return "default"
    
def get_image_url(keyword: str) -> str: # 키워드에 해당하는 이미지 URL 반환 함수.
    msg_img_map = {
        "기뻐": 1,
        "슬퍼": 2,
        "default": None
    }
    return msg_img_map.get(keyword, msg_img_map["default"])    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)