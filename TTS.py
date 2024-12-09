# from gtts import gTTS
# import io
# from fastapi import HTTPException
# import requests

# def generate_gTTS_audio(text: str, lang: str = "ko") -> io.BytesIO:
#     """
#     gTTS를 사용해 텍스트를 음성으로 변환합니다.
#     """
#     try:
#         tts = gTTS(text=text, lang=lang)
#         audio_file = io.BytesIO()
#         tts.write_to_fp(audio_file)
#         audio_file.seek(0)
#         return audio_file
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"gTTS 음성 생성 실패: {str(e)}")

# def generate_coqui_tts_audio(text: str, tts_server_url: str, speaker_id: str = "default", style_wav: str = None) -> io.BytesIO:
#     """
#     Coqui TTS API를 사용해 텍스트를 음성으로 변환합니다.
#     """
#     payload = {
#         "text": text,
#         "speaker_id": speaker_id,
#         "style_wav": style_wav
#     }
#     try:
#         response = requests.post(tts_server_url, json=payload)
#         if response.status_code != 200:
#             raise HTTPException(status_code=500, detail=f"Coqui TTS 요청 실패: {response.text}")
        
#         audio_file = io.BytesIO(response.content)
#         audio_file.seek(0)
#         return audio_file
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Coqui TTS 음성 생성 실패: {str(e)}")
# import os
# import io
# from fastapi.responses import StreamingResponse
# from gtts import gTTS
# import requests

# from dotenv import load_dotenv

# class TTS:
#     def __init__(self, use_local_gtts=True, tts_server_url=None):
#         """
#         TTS 초기화
#         :param use_local_gtts: 로컬에서 gTTS를 사용할지 여부
#         :param tts_server_url: 외부 TTS 서버 URL (None일 경우 gTTS 사용)
#         """
#         self.use_local_gtts = use_local_gtts
#         self.tts_server_url = tts_server_url or os.getenv("TTS_SERVER_URL")

#     def generate_audio(self, text: str):
#         """
#         텍스트를 음성으로 변환
#         :param text: 변환할 텍스트
#         :return: StreamingResponse로 반환할 메모리 버퍼
#         """
#         if not text:
#             raise ValueError("텍스트가 비어 있습니다.")
        
#         if self.use_local_gtts:
#             # gTTS 사용
#             tts = gTTS(text=text, lang="ko")
#             audio_file = io.BytesIO()
#             tts.write_to_fp(audio_file)
#             audio_file.seek(0)
#         else:
#             # 외부 TTS API 호출
#             if not self.tts_server_url:
#                 raise ValueError("TTS 서버 URL이 설정되지 않았습니다.")
            
#             payload = {"text": text, "speaker_id": "default"}
#             response = requests.post(self.tts_server_url, json=payload)

#             if response.status_code != 200:
#                 raise ValueError(f"TTS 서버 호출 실패: {response.text}")

#             audio_file = io.BytesIO(response.content)
#             audio_file.seek(0)
        
#         return StreamingResponse(audio_file, media_type="audio/mpeg")
# load_dotenv()
# use_local_gtts = os.getenv("USE_LOCAL_GTTS", "True").lower() == "true"
# tts_server_url = os.getenv("TTS_SERVER_URL", None)
# tts = TTS(use_local_gtts=use_local_gtts, tts_server_url=tts_server_url)
import io
import re
from gtts import gTTS
from fastapi.responses import StreamingResponse


class TTS:
    def __init__(self, language="ko"):
        """
        TTS 초기화
        :param language: 음성 언어 (기본값: 'ko' - 한국어)
        """
        self.language = language

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
        
        # gTTS로 음성 생성
        tts = gTTS(text=clean_text, lang=self.language)
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)

        # StreamingResponse로 반환
        return StreamingResponse(audio_file, media_type="audio/mpeg")
