import streamlit as st
import os
import tempfile
from pydub import AudioSegment
import gdown
import time
import datetime
from generate_subtitles import compress_audio, transcribe_audio, translate_audio
import base64
from io import BytesIO
from openai import OpenAI

def init_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False

def validate_api_key(api_key):
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except Exception:
        return False

def ai_subtitle_generator():
    # Initialize session state
    init_session_state()

    # API Key input in main interface
    api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.session_state.api_key_valid = validate_api_key(api_key)

    language_options = {
        '中文': 'zh', '英文': 'en', '馬來語': 'ms', '日文': 'ja', '韓文': 'ko', '德語': 'de', '法語': 'fr',
        '阿非利堪斯語': 'af', '阿拉伯語': 'ar', '亞美尼亞語': 'hy', '亞塞拜然語': 'az',
        '白俄羅斯語': 'be', '波士尼亞語': 'bs', '保加利亞語': 'bg', '加泰隆尼亞語': 'ca',
        '克羅埃西亞語': 'hr', '捷克語': 'cs', '丹麥語': 'da', '荷蘭語': 'nl', '愛沙尼亞語': 'et',
        '芬蘭語': 'fi', '加利西亞語': 'gl', '希臘語': 'el', '希伯來語': 'he', '印地語': 'hi',
        '匈牙利語': 'hu', '冰島語': 'is', '印度尼西亞語': 'id', '義大利語': 'it', '卡納達語': 'kn',
        '哈薩克語': 'kk', '拉脫維亞語': 'lv', '立陶宛語': 'lt', '馬其頓語': 'mk',
        '馬拉地語': 'mr', '毛利語': 'mi', '尼泊爾語': 'ne', '挪威語': 'no', '波斯語': 'fa',
        '波蘭語': 'pl', '葡萄牙語': 'pt', '羅馬尼亞語': 'ro', '俄語': 'ru', '塞爾維亞語': 'sr',
        '斯洛伐克語': 'sk', '斯洛維尼亞語': 'sl', '西班牙語': 'es', '斯瓦希里語': 'sw',
        '瑞典語': 'sv', '他加祿語': 'tl', '坦米爾語': 'ta', '泰語': 'th', '土耳其語': 'tr',
        '烏克蘭語': 'uk', '烏都語': 'ur', '越南語': 'vi', '威爾斯語': 'cy'
    }

    selected_language = st.selectbox('請選擇轉譯語言：', options=list(language_options.keys()))
    translate_to_english = st.checkbox("翻譯成英文")
    default_prompt = '繁體！'
    user_prompt = st.text_input('請輸入 Prompt 以改進轉譯品質（如果轉譯語言不是中文，要刪去預設內容）：',
                                default_prompt,
                                help='提示可幫助改善轉譯。模型會匹配提示風格。')
    temperature = st.number_input('請輸入 Temperature：', value=0.4)

    gdrive_url = st.text_input("或輸入 Google Drive 連結:")
    uploaded_file = st.file_uploader("或請上傳 MP3 或 MP4 檔案：", type=["mp3", "mp4"])

    if gdrive_url:
        output_file = "gdrive_file"
        gdown.download(gdrive_url, output_file, quiet=False, fuzzy=True)
        with open(output_file, "rb") as f:
            uploaded_file = BytesIO(f.read())
        original_filename = "gdrive_file"
    elif uploaded_file is not None:
        original_filename = os.path.splitext(uploaded_file.name)[0]

    if uploaded_file is not None and st.session_state.api_key_valid:
        total_start_time = time.time()

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_name = temp_file.name

        with st.spinner("壓縮音訊中..."):
            start_time = time.time()
            compressed_file = compress_audio(temp_file_name)
            elapsed_time = time.time() - start_time
            st.write(f"壓縮音訊所需時間：{elapsed_time:.2f} 秒")

        if translate_to_english:
            with st.spinner("生成字幕並翻譯成英文中..."):
                start_time = time.time()
                srt_file = f"{original_filename}_en.srt"
                translate_audio(compressed_file, srt_file, user_prompt, st.session_state.api_key, temperature)
                elapsed_time = time.time() - start_time
                st.write(f"生成字幕並翻譯成英文所需時間：{elapsed_time:.2f} 秒")
        else:
            with st.spinner("生成字幕中..."):
                start_time = time.time()
                srt_file = f"{original_filename}_{language_options[selected_language]}.srt"
                transcribe_audio(compressed_file, srt_file, language_options[selected_language], user_prompt, st.session_state.api_key, temperature)
                elapsed_time = time.time() - start_time
                st.write(f"生成字幕所需時間：{elapsed_time:.2f} 秒")

        total_elapsed_time = time.time() - total_start_time
        st.write(f"總共所需時間：{total_elapsed_time:.2f} 秒")

        st.success("字幕檔案已生成！")

        with open(srt_file, 'r', encoding='utf-8') as f:
            srt_data = f.read()
        srt_bytes = srt_data.encode('utf-8')
        b64 = base64.b64encode(srt_bytes).decode()

        href = f'<a href="data:file/srt;base64,{b64}" download="{srt_file}" target="_blank">點擊此處下載字幕檔案</a>'
        st.markdown(href, unsafe_allow_html=True)

        st.markdown("以下是一些實用連結：")
        st.markdown("- [合併兩個字幕](https://subtitletools.com/merge-subtitles-online)")
        st.markdown("- [把雙行字幕變成英文大小50、中文大小75](https://colab.research.google.com/drive/16I1BLSC_LR6EFZOWGXBSJwIjJ4LfTq9s?usp=sharing)")
        st.markdown("- [生成內容摘要SRT](https://colab.research.google.com/drive/1VgfPTfmbU2kjJ7nMXkmNMWcVbvOyqX0N?usp=sharing)")
    elif not st.session_state.api_key_valid:
        st.warning("請先輸入有效的 OpenAI API Key。")

if __name__ == "__main__":
    ai_subtitle_generator()