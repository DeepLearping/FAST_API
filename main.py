from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from chat_logic import setup_chat_chain
from models import ChatRequest, ChatResponse
import os
from sqlalchemy import create_engine, text
import speech_recognition as sr
import pyttsx3
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

app = FastAPI()

DATABASE_URL = os.getenv("ENV_CONNECTION")
engine = create_engine(DATABASE_URL)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI 응답을 받아 음성으로 변환하고 재생하는 함수(이득규)
def play_ai_voice(text: str):
    audio = generate_audio(text)  # generate_audio 함수가 텍스트를 음성으로 변환한다고 가정
    sound = AudioSegment.from_file(BytesIO(audio), format="mp3")  # BytesIO로 변환된 audio를 mp3로 로드
    play(sound)  # 음성 출력

# 텍스트를 음성으로 변환하는 함수 (pyttsx3 사용)(이득규)
def generate_audio(text: str) -> BytesIO:
    # 텍스트를 음성으로 변환 (pyttsx3 사용)
    engine = pyttsx3.init()
    audio_io = BytesIO()
    engine.save_to_file(text, audio_io)
    audio_io.seek(0)  # 파일 시작 위치로 이동
    return audio_io


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
        
        answer = ChatResponse(answer=response)

        # 음성으로 답변하기(이득규)
        play_ai_voice(response)

        return answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# TODO: 일정량의 최신 채팅 히스토리만 가져오고 나머지 히스토리는 무한스크롤로 로딩
@app.get("/chat_history/{conversation_id}")
async def get_history(conversation_id: int):
    try:
        history = SQLChatMessageHistory(
            table_name="chat_history",
            session_id=conversation_id,
            connection=os.getenv("ENV_CONNECTION")
        )

        return {"messages": [
#             {"role": "user" if msg.type == "human" else character_name, "content": msg.content}
            {"role": "user" if msg.type == "human" else "ai", "content": msg.content}
            for msg in history.messages
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
