from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_message_histories import SQLChatMessageHistory
from fastapi import FastAPI, HTTPException
from chat_logic import setup_chat_chain
from models import ChatRequest, ChatResponse
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage
import os
from sqlalchemy import create_engine, text

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

        # 새로운 table에 캐릭터 name과 id 포함된 message 저장
        # history = SQLChatMessageHistory(
        #     table_name="chat_history2",
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
        #         name=request.character_name,
        #         id=request.character_id
        #     )
        # )

        return ChatResponse(answer=response)

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
            {"role": "user" if msg.type == "human" else "ai", "content": msg.content}
            for msg in history.messages
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
