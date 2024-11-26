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

# 텍스트를 음성으로 변환하는 함수(이득규)
def text_to_speech(text: str):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 음성 파일을 텍스트로 변환하는 함수(이득규)
def speech_to_text(audio_file: UploadFile):
    recognizer = sr.Recognizer()
    audio_data = audio_file.file.read()
    
    with BytesIO(audio_data) as audio_stream:
        with sr.AudioFile(audio_stream) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language="ko-KR")  # 한국어로 인식
                return text
            except sr.UnknownValueError:
                raise HTTPException(status_code=400, detail="음성을 인식할 수 없습니다.")
            except sr.RequestError:
                raise HTTPException(status_code=500, detail="음성 인식 서비스에 문제가 발생했습니다.")


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
        text_to_speech(response)

        return answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# 음성 파일을 받아서 텍스트로 변환 후 채팅하는 엔드포인트(이득규)
@app.post("/chat_with_audio", response_model=ChatResponse)
async def chat_with_audio(
    audio: UploadFile = File(...),  # 음성 파일을 받기
    user_id: int = Form(...),  # user_id를 Form으로 받기
    conversation_id: int = Form(...),  # conversation_id를 Form으로 받기
    character_id: int = Form(...)  # character_id를 Form으로 받기
):
    try:
        # 음성 파일을 텍스트로 변환
        question = speech_to_text(audio)

        # chain을 캐릭터에 따라 set
        chat_chain = setup_chat_chain(character_id)

        config = {
            "configurable": {
                "user_id": user_id,
                "conversation_id": conversation_id
            }
        }

        # 변환된 텍스트로 질문 처리
        response = chat_chain.invoke({"question": question}, config)
        answer = ChatResponse(answer=response)

        # 음성으로 답변하기
        text_to_speech(response)

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
