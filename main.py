from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException, Query
from langchain_redis import RedisChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException
from chat_logic import setup_chat_chain
from models import ChatRequest, ChatResponse
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
import requests

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

        # chain을 캐릭터에 따라 set
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

        print("msg_img: ", msg_img)

        # # TTS로 응답 생성
        # tts = gTTS(text=response, lang="ko")
        # # 메모리 버퍼에 TTS 데이터를 저장
        # audio_file = io.BytesIO()
        # tts.write_to_fp(audio_file)

        # # 버퍼의 처음으로 이동
        # audio_file.seek(0)

        # TTS로 응답 생성
        TTS_SERVER_URL = "http://localhost:5002/api/tts"
        payload = {
        "text": response,
        "speaker_id": "default",
        "style_wav": None,
        }
        tts_response = requests.post(TTS_SERVER_URL, json=payload)

        if tts_response.status_code != 200:
            raise HTTPException(
        status_code=500,
        detail=f"Coqui TTS 요청 실패: {tts_response.text}"
    )

        # 메모리 버퍼에 TTS 데이터를 저장
        audio_file = io.BytesIO(tts_response.content)
        audio_file.seek(0)

        return ChatResponse(
            answer=response,
            character_id=request.character_id,
            msg_img=msg_img,
            tts_url="/chat/stream_audio"

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
    
    # try:
    #     chat_chain = setup_chat_chain(request.character_id)
        
    #     # Redis와 MySQL에서 히스토리 모두 가져오기
    #     redis_history, sql_history = get_chat_message(
    #         user_id=request.user_id,
    #         conversation_id=request.conversation_id
    #     )
        
    #     config = {
    #         "configurable": {
    #             "user_id": request.user_id,
    #             "conversation_id": request.conversation_id
    #         }
    #     }

    #     response = chat_chain.invoke({"question": request.question}, config)

    #     # 메세지를 Redis와 MySQL에 모두 저장
    #     add_message_to_both(
    #         redis_history, 
    #         sql_history, 
    #         user_id=request.user_id, 
    #         conversation_id=request.conversation_id, 
    #         question=request.question, 
    #         answer=response, 
    #         character_id=request.character_id
    #     )

    #     return ChatResponse(answer=response, character_id=request.character_id)

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

# TODO: 일정량의 최신 채팅 히스토리만 가져오고 나머지 히스토리는 무한스크롤로 로딩
@app.get("/chat_message/{conversation_id}")
async def get_history(conversation_id: int):
    try:
        history = SQLChatMessageHistory(
            table_name="chat_message",
            session_id=conversation_id,
            connection=os.getenv("ENV_CONNECTION")
        )

        return {"messages": [
            {
                "role": "user" if msg.type == "human" else "ai", 
                "content": msg.content, 
                #  "msgImgUrl": f"http://localhost:8080/chatMessage/getMsgImg/{msg.id}/{msg_img_no}.jpg" if ((msg_img_no := get_image_url(query_routing(msg.content))) != None and msg.type == "ai")
                #                  else ""
                "msgImgUrl": ""
            }
            for msg in history.messages
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # try:
    #     # Redis와 MySQL에서 히스토리 모두 가져오기
    #     redis_history, sql_history = get_chat_message(
    #         user_id=None,  # No need for user_id here
    #         conversation_id=conversation_id
    #     )

    #     # Redis에서 메세지 fetch
    #     redis_messages = redis_history.messages

    #     if not redis_messages:
    #         # Redis에 아무 정보도 없으면 MySQL에서 fetch
    #         sql_messages = sql_history.messages
    #         redis_messages = [{"role": "user" if msg.type == "human" else "ai", "content": msg.content}
    #                           for msg in sql_messages]

    #     return {"messages": [{"role": "user" if msg.type == "human" else "ai", "content": msg.content}
    #                          for msg in redis_messages]}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))



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
    print(f"Received text: {text}")  # 디버깅용 로그
    if not text:
        raise HTTPException(status_code=400, detail="text 파라미터가 비어있습니다.")
    """
    요청으로 받은 텍스트를 기반으로 Coqui TTS를 사용하여 음성을 생성하고 반환.
    """
    try:
        # Coqui TTS 서버 URL
        TTS_SERVER_URL = "http://localhost:5002/api/tts"
        

        # TTS 요청 데이터
        payload = {
            "text": text,
            "speaker_id": "default",  # 필요 시 특정 speaker_id 지정
            "style_wav": None,       # 스타일 참조 음성 (필요 없다면 None)
        }

        # Coqui TTS 서버에 POST 요청
        response = requests.post(TTS_SERVER_URL, json=payload)

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Coqui TTS 요청 실패: {response.text}"
            )

        # 응답 음성 데이터를 반환
        audio_file = io.BytesIO(response.content)
        audio_file.seek(0)

        return StreamingResponse(
            audio_file,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=tts.mp3"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오디오 생성 중 오류 발생: {str(e)}")

# @app.get("/chat/stream_audio")
# async def stream_audio(text: str = Query(..., description="음성을 생성할 텍스트")):
#     """
#     요청으로 받은 텍스트를 기반으로 음성을 생성하여 반환.
#     """
#     try:

#         # # 이모티콘 제거
#         # filtered_text = remove_emojis(text)

#         # TTS 응답을 요청으로 받은 텍스트 기반으로 생성
#         tts = gTTS(text=text, lang="ko")

#         # 메모리 버퍼에 저장
#         audio_file = io.BytesIO()
#         tts.write_to_fp(audio_file)

#         # 버퍼의 처음으로 이동
#         audio_file.seek(0)

#         # StreamingResponse로 음성 파일을 반환
#         return StreamingResponse(
#             audio_file,
#             media_type="audio/mpeg",
#             headers={"Content-Disposition": "inline; filename=tts.mp3"}
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"오디오 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)