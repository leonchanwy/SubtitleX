import streamlit as st
import openai
from pathlib import Path
import tempfile
import os

def set_openai_api_key(api_key):
    openai.api_key = api_key

def transcribe_audio(audio_file, model, language, prompt, response_format, temperature, timestamp_granularities=None):
    params = {
        "model": model,
        "file": audio_file,
        "language": language if language else None,
        "prompt": prompt if prompt else None,
        "response_format": response_format,
        "temperature": temperature
    }
    if timestamp_granularities:
        params["timestamp_granularities"] = timestamp_granularities
    
    transcript = openai.audio.transcriptions.create(**params)
    return transcript.text if response_format == 'json' else transcript

def translate_audio(audio_file, model, prompt, response_format, temperature):
    translation = openai.audio.translations.create(
        model=model,
        file=audio_file,
        prompt=prompt if prompt else None,
        response_format=response_format,
        temperature=temperature
    )
    return translation.text if response_format == 'json' else translation

def text_to_speech(text, model, voice, response_format, speed):
    response = openai.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,
        speed=speed
    )
    return response.content

def whisper_api_tool():

    api_key = st.text_input("輸入您的 OpenAI API Key", value=st.session_state.api_key, type="password")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        save_api_key(api_key)
        st.success("API Key 已保存")

    tab1, tab2, tab3 = st.tabs(["音頻轉錄", "音頻翻譯", "文字轉語音"])

    # 定義語言列表，將指定語言放在前面
    languages = [
        ("zh", "中文"),
        ("en", "英文"),
        ("ms", "馬來文"),
        ("ja", "日文"),
        ("de", "德文"),
        ("af", "南非荷蘭語"),
        ("ar", "阿拉伯語"),
        ("hy", "亞美尼亞語"),
        ("az", "阿塞拜疆語"),
        ("be", "白俄羅斯語"),
        ("bs", "波斯尼亞語"),
        ("bg", "保加利亞語"),
        ("ca", "加泰羅尼亞語"),
        ("hr", "克羅地亞語"),
        ("cs", "捷克語"),
        ("da", "丹麥語"),
        ("nl", "荷蘭語"),
        ("et", "愛沙尼亞語"),
        ("fi", "芬蘭語"),
        ("fr", "法語"),
        ("gl", "加利西亞語"),
        ("el", "希臘語"),
        ("he", "希伯來語"),
        ("hi", "印地語"),
        ("hu", "匈牙利語"),
        ("is", "冰島語"),
        ("id", "印尼語"),
        ("it", "義大利語"),
        ("kk", "哈薩克語"),
        ("ko", "韓語"),
        ("lv", "拉脫維亞語"),
        ("lt", "立陶宛語"),
        ("mk", "馬其頓語"),
        ("mi", "毛利語"),
        ("mr", "馬拉地語"),
        ("ne", "尼泊爾語"),
        ("no", "挪威語"),
        ("fa", "波斯語"),
        ("pl", "波蘭語"),
        ("pt", "葡萄牙語"),
        ("ro", "羅馬尼亞語"),
        ("ru", "俄語"),
        ("sr", "塞爾維亞語"),
        ("sk", "斯洛伐克語"),
        ("sl", "斯洛維尼亞語"),
        ("es", "西班牙語"),
        ("sw", "斯瓦希里語"),
        ("sv", "瑞典語"),
        ("tl", "他加祿語"),
        ("ta", "泰米爾語"),
        ("th", "泰語"),
        ("tr", "土耳其語"),
        ("uk", "烏克蘭語"),
        ("ur", "烏爾都語"),
        ("vi", "越南語"),
        ("cy", "威爾士語")
    ]

    with tab1:
        st.header("音頻轉錄")
        uploaded_file = st.file_uploader("選擇音頻文件", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key="transcribe_file_uploader")
        model = st.selectbox("模型", ["whisper-1"], key="transcribe_model")
        language = st.selectbox("語言", languages, format_func=lambda x: f"{x[1]} ({x[0]})", key="transcribe_language")
        prompt = st.text_area("提示詞 (可選)", key="transcribe_prompt")
        response_format = st.selectbox("響應格式", ["json", "text", "srt", "verbose_json", "vtt"], key="transcribe_response_format")
        temperature = st.slider("溫度", 0.0, 1.0, 0.0, 0.1, key="transcribe_temperature")
        
        # 只有當 response_format 為 verbose_json 時才顯示 timestamp_granularities 選項
        timestamp_granularities = None
        if response_format == "verbose_json":
            timestamp_options = st.multiselect(
                "時間戳精度",
                ["word", "segment"],
                default=["segment"],
                help="選擇時間戳的精度。注意：生成詞級時間戳會增加延遲。",
                key="transcribe_timestamp_granularities"
            )
            if timestamp_options:
                timestamp_granularities = timestamp_options
        
        if uploaded_file is not None and st.button("轉錄", key="transcribe_button"):
            with st.spinner("正在轉錄..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                with open(tmp_file_path, "rb") as audio_file:
                    transcript = transcribe_audio(audio_file, model, language[0], prompt, response_format, temperature, timestamp_granularities)
                
                os.unlink(tmp_file_path)
                
                st.text_area("轉錄結果", transcript, height=250, key="transcribe_result")


    with tab2:
        st.header("音頻翻譯")
        uploaded_file = st.file_uploader("選擇要翻譯的音頻文件", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key="translate_file_uploader")
        model = st.selectbox("模型", ["whisper-1"], key="translate_model")
        prompt = st.text_area("提示詞 (可選)", key="translate_prompt")
        response_format = st.selectbox("響應格式", ["json", "text", "srt", "verbose_json", "vtt"], key="translate_response_format")
        temperature = st.slider("溫度", 0.0, 1.0, 0.0, 0.1, key="translate_temperature")
        
        if uploaded_file is not None and st.button("翻譯", key="translate_button"):
            with st.spinner("正在翻譯..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                with open(tmp_file_path, "rb") as audio_file:
                    translation = translate_audio(audio_file, model, prompt, response_format, temperature)
                
                os.unlink(tmp_file_path)
                
                st.text_area("翻譯結果", translation, height=250, key="translate_result")

    with tab3:
        st.header("文字轉語音")
        text_input = st.text_area("輸入要轉換為語音的文字", height=150, key="tts_input")
        model = st.selectbox("模型", ["tts-1", "tts-1-hd"], key="tts_model")
        voice = st.selectbox("選擇聲音", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], key="tts_voice")
        response_format = st.selectbox("音頻格式", ["mp3", "opus", "aac", "flac"], key="tts_response_format")
        speed = st.slider("速度", 0.25, 4.0, 1.0, 0.25, key="tts_speed")
        
        if text_input and st.button("生成語音", key="tts_button"):
            with st.spinner("正在生成音頻..."):
                audio_content = text_to_speech(text_input, model, voice, response_format, speed)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}") as tmp_audio_file:
                    tmp_audio_file.write(audio_content)
                    tmp_audio_file_path = tmp_audio_file.name
                
                st.audio(tmp_audio_file_path, format=f"audio/{response_format}")
                
                os.unlink(tmp_audio_file_path)