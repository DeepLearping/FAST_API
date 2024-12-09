# import io
# import re
# from gtts import gTTS
# from fastapi.responses import StreamingResponse


# class TTS:
#     def __init__(self, language="ko"):
#         """
#         TTS 초기화
#         :param language: 음성 언어 (기본값: 'ko' - 한국어)
#         """
#         self.language = language

#     def preprocess_text(self, text: str) -> str:
#         """
#         텍스트 전처리: 이모티콘 및 불필요한 특수 문자 제거
#         :param text: 원본 텍스트
#         :return: 전처리된 텍스트
#         """
#         # 이모티콘 및 특수 문자 제거 (유니코드 범위 활용)
#         text = re.sub(r"[^\w\sㄱ-ㅎㅏ-ㅣ가-힣,.!?]", "", text)
#         return text

#     def generate_audio(self, text: str):
#         """
#         텍스트를 음성으로 변환
#         :param text: 변환할 텍스트
#         :return: StreamingResponse로 반환할 메모리 버퍼
#         """
#         if not text:
#             raise ValueError("텍스트가 비어 있습니다.")
        
#          # 텍스트 전처리
#         clean_text = self.preprocess_text(text)
        
#         # gTTS로 음성 생성
#         tts = gTTS(text=clean_text, lang=self.language)
#         audio_file = io.BytesIO()
#         tts.write_to_fp(audio_file)
#         audio_file.seek(0)

#         # StreamingResponse로 반환
#         return StreamingResponse(audio_file, media_type="audio/mpeg")
import io
import re
import requests
from fastapi.responses import StreamingResponse

# 일레븐랩스 API 키
API_KEY = ''
# 일레븐랩스의 음성 클로닝 엔드포인트
API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

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
            "voice_id": self.voice_id,  # 사용할 음성 ID
            "language": self.language,  # 언어 설정
            "prosody": {
                "rate": "slow",   # 속도 (slow, medium, fast)
                "pitch": "medium",  # 억양 (low, medium, high)
                "volume": "medium"  # 볼륨 (low, medium, high)
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