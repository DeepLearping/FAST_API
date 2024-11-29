from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import SQLChatMessageHistory #, RedisChatMessageHistory
from fastapi import FastAPI, HTTPException
from langchain_redis import RedisChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from chat_logic import setup_chat_chain
from models import ChatRequest, ChatResponse
import os
from sqlalchemy import create_engine
from gtts import gTTS  # gTTS 설치 필요
import io
from fastapi.responses import StreamingResponse

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

# def add_message_to_both(redis_history, sql_history, user_id, conversation_id, question, answer, character_id):
#     # Add user message
#     redis_history.add_user_message(HumanMessage(content=question, id=user_id))
#     sql_history.add_user_message(HumanMessage(content=question, id=user_id))

#     # Add AI message with character_id in metadata
#     metadata = {"character_id": character_id}
#     ai_message_content = json.dumps({"answer": answer, "metadata": metadata})

#     redis_history.add_ai_message(AIMessage(content=ai_message_content, id=character_id))
#     sql_history.add_ai_message(AIMessage(content=ai_message_content, id=character_id))

# def get_chat_message(user_id, conversation_id):
#         redis_history = RedisChatMessageHistory(
#             redis_url=os.getenv("REDIS_URL"),
#             session_id=conversation_id,
#             ttl=3600  # Optional TTL (1 hour) for example
#         )
#         sql_history = SQLChatMessageHistory(
#             table_name="chat_message",
#             session_id=conversation_id,
#             connection=os.getenv("ENV_CONNECTION")
#         )
#         return redis_history, sql_history

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

        # 새로운 table에 캐릭터 name과 id 포함된 message 저장
        # history = SQLChatMessageHistory(
        #     table_name="chat_message2",
        #     session_id=request.conversation_id,
        #     connection=os.getenv("ENV_CONNECTION")
        # )
        
        # history.add_user_message(
        #     HumanMessage(
        #         content=request.question,
        #         id=request.user_id
        #     )
        # )
        
        # history.add_ai_message(
        #     AIMessage(
        #         content=response,
        #         # name=request.character_name,
        #         id=request.character_id
        #     )
        # )

        # 응답(response)에서 키워드 감지 및 이미지 URL 매핑
        detected_keyword = query_routing(response)  # 응답 내용을 분석
        msg_img= get_image_url(detected_keyword)  # 키워드에 해당하는 이미지 URL 가져오기

        print("msg_img: ", msg_img)

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
            msg_img=msg_img,
            tts_url="/chat/stream_audio"

        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def query_routing(response: str) -> str:
    """
    응답 내용에서 키워드를 감지하는 함수.
    """
    keywords = ["기뻐", "슬퍼"]  # 감지하려는 키워드 목록
    for keyword in keywords:
        if keyword in response.lower():
            return keyword
    return "default"
    
def get_image_url(keyword: str) -> str:
    """
    키워드에 해당하는 이미지 URL 반환 함수.
    """
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


@app.get("/chat/stream_audio")
async def stream_audio():
    try:
        # TTS 응답을 바로 메모리에서 스트리밍
        tts_text = "여기에서 음성을 생성할 텍스트를 작성하거나 요청에서 받은 텍스트로 대체"
        tts = gTTS(text=tts_text, lang="ko")

        # 메모리 버퍼에 저장
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)

        # 버퍼의 처음으로 이동
        audio_file.seek(0)

        # StreamingResponse로 음성 파일을 반환
        return StreamingResponse(audio_file, media_type="audio/mpeg", headers={"Content-Disposition": "inline; filename=tts.mp3"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
