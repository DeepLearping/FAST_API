import io
import re
import requests

from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import os

# .env 파일 정보 불러오기
load_dotenv()

# 일레븐랩스 API 키
API_KEY = os.getenv('TTS_API_KEY')
VOICE_ID = os.getenv('VOICE_ID')
# 일레븐랩스의 음성 클로닝 엔드포인트
API_URL = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"

class TTS:
    def __init__(self, language="ko", voice_id=""):
        """
        TTS 초기화
        :param language: 음성 언어 (기본값: 'ko' - 한국어)
        :param voice_id: 일레븐랩스에서 생성한 나만의 보이스 ID
        """
        self.language = language
        self.voice_id = voice_id

    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리: 이모티콘 및 불필요한 특수 문자 제거
        :param text: 원본 텍스트
        :return: 전처리된 텍스트
        """
        # 이모티콘 및 특수 문자 제거 (유니코드 범위 활용)
        text = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣,.!?]", "", text)
        return text

    def generate_audio(self, text: str):
        """
        텍스트를 음성으로 변환
        :param text: 변환할 텍스트
        :return: StreamingResponse로 반환할 메모리 버퍼
        """
        if not text:
            raise ValueError("텍스트가 비어 있습니다.")
        
        # 텍스트 전처리
        clean_text = self.preprocess_text(text)
        
        # 일레븐랩스 API에 요청을 보내기 위한 데이터 준비
        headers = {
            "xi-api-key": API_KEY,
        }
        
        # 요청 데이터
        data = {
            "text": clean_text,
            "model_id": "eleven_multilingual_v2",
            "voice_id": self.voice_id,  # 사용할 음성 ID
            "language": self.language,  # 언어 설정
            "prosody": {
                "rate": "medium",   # 속도 (slow, medium, fast)
                "pitch": "high",  # 억양 (low, medium, high)
                "volume": "high",  # 볼륨 (low, medium, high)
                "stability": "0.5",
                "similarity_boost" : "0.9",
                "style":1
                }  
        }
        
        # 일레븐랩스 API 호출
        response = requests.post(API_URL.format(voice_id=self.voice_id), headers=headers, json=data)
        
        if response.status_code == 200:
            audio_content = response.content
            audio_file = io.BytesIO(audio_content)
            audio_file.seek(0)

            # StreamingResponse로 반환
            return StreamingResponse(audio_file, media_type="audio/mpeg")
        else:
            raise Exception(f"음성 생성 실패: {response.text}")