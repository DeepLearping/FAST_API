from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException, Query
from langchain_redis import RedisChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException
from chat_logic import get_or_load_retriever, setup_chat_chain
from models import BalanceChatRequest, CharacterMatchResponse, ChatRequest, ChatResponse, LoadInfoRequest
from chat_logic import setup_character_matching_prompt, setup_chat_chain
from models import CharacterMatchRequest, ChatRequest, ChatResponse
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage
import os
from sqlalchemy import create_engine
from gtts import gTTS  # gTTS 설치 필요
import io
from fastapi.responses import StreamingResponse
import re

app = FastAPI()

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

# 방 입장 시 미리 필요한 데이터 로드
@app.post("/load_info")
async def load_info(request: LoadInfoRequest):
    char_id_list = request.char_id_list
    
    for char_id in char_id_list:
        get_or_load_retriever(char_id)

# 캐릭터와 채팅
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # import time
        # start_time = time.time()
        chat_chain = setup_chat_chain(request.character_id)
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

        # TTS로 응답 생성
        tts = gTTS(text=response, lang="ko")
        # 메모리 버퍼에 TTS 데이터를 저장
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)

        # 버퍼의 처음으로 이동
        audio_file.seek(0)

        return ChatResponse(
            answer=response,
            character_id=request.character_id,
            msg_img=msg_img
        )
    
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

@app.post("/balanceChat", response_model=ChatResponse)
async def balance_chat(request: BalanceChatRequest):
    print("🍳🍳🍳🍳🍳밸런스게임")
    try:
        chat_chain = setup_chat_chain(request.character_id, request.keyword)
        
        config = {
            "configurable": {
                "user_id": request.user_id,
                "conversation_id": request.conversation_id
            }
        }

        response = chat_chain.invoke({"question": request.question}, config)
        
        detected_keyword = query_routing(response)  # 응답 내용을 분석
        msg_img= get_image_url(detected_keyword)  # 키워드에 해당하는 이미지 URL 가져오기

        # TTS로 응답 생성
        tts = gTTS(text=response, lang="ko")
        # 메모리 버퍼에 TTS 데이터를 저장
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)

        # 버퍼의 처음으로 이동
        audio_file.seek(0)

        return ChatResponse(
            answer=response,
            character_id=request.character_id,
            msg_img=msg_img
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 이모티콘 제거 함수
def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 감정 이모티콘
        "\U0001F300-\U0001F5FF"  # 기호 및 아이콘
        "\U0001F680-\U0001F6FF"  # 교통 및 기계
        "\U0001F1E0-\U0001F1FF"  # 국기
        "\U00002500-\U00002BEF"  # 기타 기호
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

@app.get("/chat/stream_audio")
async def stream_audio(text: str = Query(..., description="음성을 생성할 텍스트")):
    """
    요청으로 받은 텍스트를 기반으로 음성을 생성하여 반환.
    """
    try:
        # # 이모티콘 제거
        # filtered_text = remove_emojis(text)

        # TTS 응답을 요청으로 받은 텍스트 기반으로 생성
        tts = gTTS(text=text, lang="ko")

        # 메모리 버퍼에 저장
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)

        # 버퍼의 처음으로 이동
        audio_file.seek(0)

        # StreamingResponse로 음성 파일을 반환
        return StreamingResponse(
            audio_file,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=tts.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오디오 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)