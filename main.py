from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from langchain_openai import ChatOpenAI
from chat_logic import get_or_load_retriever, setup_chat_chain, setup_balanceChat_chain, emotion_analyzation_prompt, setup_character_matching_prompt, setup_chat_chain, setup_balanceChat_chain
from models import BalanceChatRequest, CharacterMatchResponse, ChatRequest, ChatResponse, LoadInfoRequest, CharacterMatchRequest, ChatRequest, ChatResponse
from contextlib import asynccontextmanager
from TTS import TTS
from sqlalchemy import create_engine
import os
import re
import random

def init():
    for char_id in [6]:
        get_or_load_retriever(char_id)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init()
    yield

app = FastAPI(lifespan=lifespan)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
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
        chat_chain = setup_chat_chain(request.character_id)
        
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

        # 메세지 감정 분석
        msg_img = 0
        if random.random() < 0.2:   # 20% 확률로 캐릭터 메세지 감정 분석 (happy / sad / neither)
            msg_img = analyze_emotion(response)
            # print("메세지 감정분석 결과: ", msg_img)

        return ChatResponse(
            answer=response,
            character_id=request.character_id,
            msg_img=msg_img,
            tts_url="/chat/stream_audio"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def analyze_emotion(message: str):
    try:
        prompt = emotion_analyzation_prompt()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        result = llm.invoke(
            prompt.format(message=message)
        )

        numeric_ids = re.findall(r'\b\d+\b', result.content)    # 숫자(정수)만 추출
        emotion_code = [int(emotion_id) for emotion_id in numeric_ids]
        
        return emotion_code[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 단체방에서 사용자가 질문을 받아 어떤 캐릭터가 응답하기에 적합한지 결정하여 캐릭터id 리스트 반환
@app.post("/character/match", response_model=CharacterMatchResponse)
async def match_character(request: CharacterMatchRequest):
    try:
        question = request.question
        char_id_list = request.char_id_list
        chat_history_list = request.chat_history_list

        formatted_chat_history = "\n".join(chat_history_list)

        character_info = [
            f"{char_id}: {get_character_info_by_id(char_id)}"
            for char_id in char_id_list
        ]
        formatted_character_info = "\n".join(character_info)
        
        prompt = setup_character_matching_prompt()
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        result = llm.invoke(
            prompt.format(question=question, chat_history=formatted_chat_history, character_info=formatted_character_info)
        )

        # 캐릭터 ID 리스트로 반환
        numeric_ids = re.findall(r'\b\d+\b', result.content)    # 숫자(정수)만 추출
        matching_characters = [int(char_id) for char_id in numeric_ids]
        
        return CharacterMatchResponse(
            selected_char_id_list=matching_characters
        )

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

global_situation = {}

@app.post("/balanceChat", response_model=ChatResponse)
async def balance_chat(request: BalanceChatRequest):
    try:
        current_situation = global_situation.get(request.character_id)
        if request.situation:
            global_situation[request.character_id] = request.situation
            current_situation = request.situation  # 상황 업데이트
        
        # 챗 체인 설정
        chat_chain = setup_balanceChat_chain(request.character_id, request.keyword, current_situation)

        config = {
            "configurable": {
                "user_id": request.user_id,
                "conversation_id": request.conversation_id
            }
        }

        response = chat_chain.invoke({"question": request.question}, config)

        msg_img = 0
        # if random.random() < 0.2:   # 20% 확률로 캐릭터 메세지 감정 분석 (happy / sad / neither)
        #     msg_img = analyze_emotion(response)
        # detected_keyword = query_routing(response)  # 응답 내용을 분석
        # msg_img = get_image_url(detected_keyword)  # 키워드에 해당하는 이미지 URL 가져오기

        return ChatResponse(
            answer=response,
            character_id=request.character_id,
            msg_img=msg_img
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

tts = TTS(language="ko")

@app.get("/chat/stream_audio")
async def stream_audio(text: str = Query(..., description="음성을 생성할 텍스트")):
    """
    텍스트를 받아 음성을 반환하는 API 엔드포인트
    :param text: 음성을 생성할 텍스트
    :return: StreamingResponse
    """
    try:
        return tts.generate_audio(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))