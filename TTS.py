from gtts import gTTS
import io
from fastapi import HTTPException
import requests

def generate_gTTS_audio(text: str, lang: str = "ko") -> io.BytesIO:
    """
    gTTS를 사용해 텍스트를 음성으로 변환합니다.
    """
    try:
        tts = gTTS(text=text, lang=lang)
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"gTTS 음성 생성 실패: {str(e)}")

def generate_coqui_tts_audio(text: str, tts_server_url: str, speaker_id: str = "default", style_wav: str = None) -> io.BytesIO:
    """
    Coqui TTS API를 사용해 텍스트를 음성으로 변환합니다.
    """
    payload = {
        "text": text,
        "speaker_id": speaker_id,
        "style_wav": style_wav
    }
    try:
        response = requests.post(tts_server_url, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Coqui TTS 요청 실패: {response.text}")
        
        audio_file = io.BytesIO(response.content)
        audio_file.seek(0)
        return audio_file
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coqui TTS 음성 생성 실패: {str(e)}")