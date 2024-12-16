import streamlit as st
from groq import Groq
from pathlib import Path
import tempfile
import os
import json
from datetime import timedelta

# Constants
MODEL_TURBO = "whisper-large-v3-turbo"
MODEL_TRANSCRIBE = "distil-whisper-large-v3-en"
MODEL_TRANSLATE = "whisper-large-v3"
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes

# 模型說明
MODEL_DESCRIPTIONS = {
    MODEL_TURBO: "快速多語言轉錄（準確度適中，價格優惠）",
    MODEL_TRANSCRIBE: "僅支持英語（最快速，價格最低）",
    MODEL_TRANSLATE: "多語言支持（最準確，支持翻譯）"
}

# 模型功能支持
MODEL_CAPABILITIES = {
    MODEL_TURBO: {"transcribe": True, "translate": False, "multilingual": True},
    MODEL_TRANSCRIBE: {"transcribe": True, "translate": False, "multilingual": False},
    MODEL_TRANSLATE: {"transcribe": True, "translate": True, "multilingual": True}
}

# 添加到常量部分
MODEL_PRICING = {
    MODEL_TURBO: 0.04,      # $0.04 per hour
    MODEL_TRANSCRIBE: 0.02,  # $0.02 per hour
    MODEL_TRANSLATE: 0.111   # $0.111 per hour
}

def set_groq_api_key(api_key):
    return Groq(api_key=api_key)

def transcribe_audio(client, audio_file_path, model, language, prompt, response_format, temperature):
    with open(audio_file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_file_path), file),
            model=model,
            language=language,
            prompt=prompt,
            response_format="verbose_json" if response_format == "srt" else response_format,
            temperature=temperature
        )
    if response_format == "srt":
        return json_to_srt(transcription.model_dump()), transcription.model_dump()
    return transcription.text, transcription.model_dump() if response_format in ["json", "verbose_json"] else None

def translate_audio(client, audio_file_path, model, prompt, response_format, temperature):
    with open(audio_file_path, "rb") as file:
        translation = client.audio.translations.create(
            file=(os.path.basename(audio_file_path), file),
            model=model,
            prompt=prompt,
            response_format="verbose_json" if response_format == "srt" else response_format,
            temperature=temperature
        )
    if response_format == "srt":
        return json_to_srt(translation.model_dump()), translation.model_dump()
    return translation.text, translation.model_dump() if response_format in ["json", "verbose_json"] else None

def format_time(seconds):
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def json_to_srt(json_data):
    try:
        srt_lines = []
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        if 'segments' in json_data:
            segments = json_data['segments']
        elif isinstance(json_data, list):
            segments = json_data
        else:
            return "錯誤: 無法找到分段信息"

        for i, segment in enumerate(segments, 1):
            start_time = format_time(segment['start'])
            end_time = format_time(segment['end'])
            srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n")
        return "\n".join(srt_lines)
    except KeyError as e:
        return f"錯誤: JSON中缺少必要的鍵 {str(e)}"
    except Exception as e:
        return f"錯誤: 處理JSON時發生未知錯誤 {str(e)}"

def show_pricing_info(model):
    st.info(f"當前模型價格：${MODEL_PRICING[model]}/小時")

def whisper_api_tool():
    st.title("🎤 Groq Whisper API Tool")
    
    # 添加版本信息
    st.markdown("""
    ### 功能特點
    - 支持最新的 whisper-large-v3-turbo 模型
    - 多語言轉錄和翻譯支持
    - 自動語言檢測
    - JSON 和 SRT 格式輸出
    """)
    
    api_key = st.text_input("輸入您的 Groq API Key", value=st.session_state.get('api_key', ''), type="password")
    
    if not api_key:
        st.warning("請輸入您的 Groq API Key")
        return
        
    try:
        client = set_groq_api_key(api_key)
        st.session_state.api_key = api_key
        st.success("API Key 已設置")
    except Exception as e:
        st.error(f"API Key 設置失敗: {str(e)}")
        return

    tab1, tab2 = st.tabs(["音頻轉錄", "音頻翻譯"])

    languages = [
        ("zh", "中文"),
        ("en", "英文"),
        ("ja", "日文"),
        ("ms", "馬來文"),
        ("de", "德文"),
    ]

    with tab1:
        st.header("音頻轉錄")
        uploaded_file = st.file_uploader("選擇音頻文件", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key="transcribe_file_uploader")
        
        # 模型選擇
        model = st.selectbox(
            "模型",
            [MODEL_TURBO, MODEL_TRANSCRIBE, MODEL_TRANSLATE],
            format_func=lambda x: f"{x} - {MODEL_DESCRIPTIONS[x]}",
            key="transcribe_model"
        )

        # 根據模型功能顯示語言選擇
        if MODEL_CAPABILITIES[model]["multilingual"]:
            language = st.selectbox(
                "語言",
                languages,
                format_func=lambda x: f"{x[1]} ({x[0]})",
                key="transcribe_language"
            )
        else:
            st.info("注意：選擇的模型僅支持英語")
            language = ("en", "英文")

        prompt = st.text_area(
            "提示詞 (可選)",
            help="提供上下文或指定如何拼寫不熟悉的單詞（限224個令牌）",
            key="transcribe_prompt"
        )
        response_format = st.selectbox(
            "響應格式",
            ["json", "text", "verbose_json", "srt"],
            key="transcribe_response_format"
        )
        temperature = st.slider(
            "溫度",
            0.0,
            1.0,
            0.0,
            0.1,
            help="控制輸出的隨機性（0最保守，1最創造性）",
            key="transcribe_temperature"
        )

        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"文件大小超過限制 (25MB)。當前文件大小: {uploaded_file.size / 1024 / 1024:.2f} MB")
            elif st.button("轉錄", key="transcribe_button"):
                try:
                    with st.spinner("正在轉錄..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        transcript_text, transcript_json = transcribe_audio(
                            client,
                            tmp_file_path,
                            model,
                            language[0],
                            prompt,
                            response_format,
                            temperature
                        )

                        os.unlink(tmp_file_path)

                        st.text_area("轉錄結果", transcript_text, height=250, key="transcribe_result")

                        if transcript_json:
                            st.download_button(
                                label="下載JSON數據",
                                data=json.dumps(transcript_json, ensure_ascii=False, indent=2),
                                file_name="transcription.json",
                                mime="application/json"
                            )
                            
                            srt_content = json_to_srt(transcript_json)
                            st.download_button(
                                label="下載SRT文件",
                                data=srt_content,
                                file_name="transcription.srt",
                                mime="text/plain"
                            )
                except Exception as e:
                    st.error(f"轉錄過程中發生錯誤: {str(e)}")

    with tab2:
        st.header("音頻翻譯")
        uploaded_file = st.file_uploader("選擇要翻譯的音頻文件", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key="translate_file_uploader")
        model = st.selectbox("模型", [MODEL_TRANSLATE], key="translate_model")
        prompt = st.text_area("提示詞 (可選)", key="translate_prompt", help="提供上下文或指定如何拼寫不熟悉的單詞（限224個令牌）。")
        response_format = st.selectbox("響應格式", ["json", "text", "verbose_json", "srt"], key="translate_response_format")
        temperature = st.slider("溫度", 0.0, 1.0, 0.0, 0.1, key="translate_temperature", help="指定0到1之間的值來控制翻譯輸出。")

        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"文件大小超過限制 (25MB)。當前文件大小: {uploaded_file.size / 1024 / 1024:.2f} MB")
            elif st.button("翻譯", key="translate_button"):
                try:
                    with st.spinner("正在翻譯..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        translation_text, translation_json = translate_audio(
                            client,
                            tmp_file_path,
                            model,
                            prompt,
                            response_format,
                            temperature
                        )

                        os.unlink(tmp_file_path)

                        st.text_area("翻譯結果", translation_text, height=250, key="translate_result")

                        if translation_json:
                            st.download_button(
                                label="下載JSON數據",
                                data=json.dumps(translation_json, ensure_ascii=False, indent=2),
                                file_name="translation.json",
                                mime="application/json"
                            )
                            
                            srt_content = json_to_srt(translation_json)
                            st.download_button(
                                label="下載SRT文件",
                                data=srt_content,
                                file_name="translation.srt",
                                mime="text/plain"
                            )
                except Exception as e:
                    st.error(f"翻譯過程中發生錯誤: {str(e)}")

if __name__ == "__main__":
    whisper_api_tool()
