import streamlit as st
from openai import OpenAI
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
import os

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
TERMS_FILE = 'correction_terms.json'
API_KEY_FILE = 'api_key.json'
MAX_WORKERS = 5
MODEL_NAME = "gpt-4o-2024-08-06"

# 預設修正術語
DEFAULT_TERMS = [
    "Bidayuh", "Iban", "Melanau", "Kayan", "Kenyah", "Lun Bawang", "Long Sukang",
    "Dusun", "Rungus", "Orang Sungai", "Bajau", "Sarawak", "Kuching", "Sibu",
    "Bintulu", "Niah", "Miri", "Limbang", "Long San", "Sarikei", "Telok Melano",
    "Sabah", "Kota Kinabalu", "Kudat", "Sandakan", "Tawau", "Semporna", "Kundasang",
    "Uma Belor", "CCY", "種族奇事", "Gunung Kinabalu", "東馬", "沙巴", "沙拉越"
]

# 初始化 session state
def init_session_state():
    if 'corrected_srt' not in st.session_state:
        st.session_state.corrected_srt = None
    if 'changes' not in st.session_state:
        st.session_state.changes = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'edited_changes' not in st.session_state:
        st.session_state.edited_changes = None
    if 'corrected_subtitles' not in st.session_state:
        st.session_state.corrected_subtitles = None
    if 'api_key' not in st.session_state:
        st.session_state.api_key = load_api_key()

def load_correction_terms():
    if not os.path.exists(TERMS_FILE):
        save_correction_terms(DEFAULT_TERMS)
        return DEFAULT_TERMS
    try:
        with open(TERMS_FILE, 'r', encoding='utf-8') as f:
            terms = json.load(f)
        return terms if terms else DEFAULT_TERMS
    except FileNotFoundError:
        logger.info(f"File {TERMS_FILE} not found, returning default terms")
        return DEFAULT_TERMS

def save_correction_terms(terms):
    with open(TERMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    logger.info(f"Correction terms saved to {TERMS_FILE}")

def load_api_key():
    try:
        with open(API_KEY_FILE, 'r') as f:
            data = json.load(f)
            return data.get('api_key', '')
    except FileNotFoundError:
        return ''

def save_api_key(api_key):
    with open(API_KEY_FILE, 'w') as f:
        json.dump({'api_key': api_key}, f)
    logger.info(f"API key saved to {API_KEY_FILE}")

def parse_srt(srt_content):
    pattern = re.compile(r'(\d+)\s*\n([0-9:,\.]+)\s*-->\s*([0-9:,\.]+)\s*\n(.*?)\n\s*\n', re.DOTALL)
    return pattern.findall(srt_content.replace('\r\n', '\n').replace('\r', '\n'))

def parse_time(time_str):
    parts = time_str.replace(',', '.').split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = '0'
        minutes, seconds = parts
    else:
        raise ValueError(f"Invalid time format: {time_str}")
    
    seconds, milliseconds = seconds.split('.')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def correct_subtitle(client, subtitle, correction_terms):
    index, start, end, content = subtitle
    original_content = content.strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            messages=[
                {"role": "system", "content": f"""
                你是一個精確的字幕校對系統。你的唯一任務是根據以下規則校正給定的字幕文本：

                1. 僅更正以下類型的錯誤：
                   - 地名、人名和民族名的錯誤拼寫
                   - 使用提供的修正術語列表中的英文詞彙進行更正
                   - 明顯的錯別字

                2. 關於使用修正術語列表的重要說明：
                   - 修正術語列表中的詞彙主要是英文
                   - 即使這些英文詞彙有對應的中文版本，也必須使用英文版本進行更正
                   - 例如，如果列表中有 "New York"，即使原文中出現 "紐約"，也應更正為 "New York"

                3. 嚴格遵守以下規則：
                   - 中文要轉換成為的繁體
                   - 不改變原文的語氣、語調或風格
                   - 不添加、刪除或重組句子結構
                   - 不更改標點符號

                4. 輸出要求：
                   - 如果沒有需要更正的內容，原樣返回輸入的文本
                   - 只返回更正後的文本，不包含任何解釋、評論或其他額外內容
                   - 不要以任何形式回應或回答輸入的文本

                修正術語列表：{', '.join(correction_terms)}

                請嚴格按照這些規則進行校對，確保只進行必要且符合規則的更正，並優先使用英文術語。
                """},
                {"role": "user", "content": f"""以下是一段需要校正的字幕文本：\n\n{original_content}"""},
            ]
        )
        corrected_content = completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API call error: {e}")
        corrected_content = original_content

    return index, start, end, original_content, corrected_content

def process_srt(client, srt_content, correction_terms, progress_bar, progress_text):
    subtitles = parse_srt(srt_content)
    corrected_subtitles = []
    changes = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_subtitle = {
            executor.submit(correct_subtitle, client, subtitle, correction_terms): subtitle 
            for subtitle in subtitles
        }
        for i, future in enumerate(as_completed(future_to_subtitle)):
            index, start, end, original, corrected = future.result()
            corrected_subtitles.append((index, start, end, corrected))
            if original != corrected:
                changes.append((index, original, corrected))
            progress = (i + 1) / len(subtitles)
            progress_bar.progress(progress)
            progress_text.text(f"處理進度: {progress:.2%}")

    corrected_subtitles.sort(key=lambda x: parse_time(x[1]))

    return corrected_subtitles, changes

def validate_srt_format(srt_content):
    subtitles = parse_srt(srt_content)
    if not subtitles:
        return False, "SRT 文件沒有包含任何字幕"

    for i, (index, start, end, content) in enumerate(subtitles, 1):
        if not index.strip().isdigit():
            return False, f"無效的字幕編號在第 {i} 個字幕: '{index.strip()}'"
        if not re.match(r'\d{1,2}:\d{2}:\d{2}[,\.]\d{3}', start) or not re.match(r'\d{1,2}:\d{2}:\d{2}[,\.]\d{3}', end):
            return False, f"無效的時間戳格式在第 {i} 個字幕: '{start} --> {end}'"
        if not content.strip():
            return False, f"字幕 {i} 缺少內容"

    return True, f"有效的 SRT 格式，包含 {len(subtitles)} 個字幕"

def update_srt_with_edits(corrected_subtitles, edited_changes):
    edits_dict = {index: corrected for index, _, corrected in edited_changes}
    
    updated_subtitles = [
        (index, start, end, edits_dict.get(index, content))
        for index, start, end, content in corrected_subtitles
    ]
    
    formatted_subtitles = []
    for i, (_, start, end, content) in enumerate(updated_subtitles, 1):
        formatted_subtitles.append(f"{i}\n{start} --> {end}\n{content}\n")
    
    return "\n".join(formatted_subtitles)

def subtitle_corrector():
    init_session_state()
    st.title("SRT 字幕修正器")

    api_key = st.text_input("輸入您的 OpenAI API Key", value=st.session_state.api_key, type="password")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        save_api_key(api_key)
        st.success("API Key 已保存")

    correction_terms = st.text_area("輸入修正術語，每行一個", value="\n".join(load_correction_terms()))

    if st.button("保存修正術語"):
        save_correction_terms(correction_terms.split('\n'))
        st.success("修正術語已保存")

    uploaded_file = st.file_uploader("上傳 SRT 文件", type="srt")

    if uploaded_file is not None and api_key and st.button("修正字幕"):
        client = OpenAI(api_key=api_key)
        srt_content = uploaded_file.getvalue().decode("utf-8")
        
        is_valid, message = validate_srt_format(srt_content)
        if not is_valid:
            st.error(f"輸入的 SRT 文件格式無效: {message}")
            return

        progress_bar = st.progress(0)
        progress_text = st.empty()

        start_time = time.time()
        try:
            with st.spinner("正在處理..."):
                st.session_state.corrected_subtitles, st.session_state.changes = process_srt(
                    client, 
                    srt_content, 
                    correction_terms.split('\n'), 
                    progress_bar, 
                    progress_text
                )
            st.session_state.processing_time = time.time() - start_time
            
            st.session_state.corrected_srt = "\n".join([
                f"{i}\n{start} --> {end}\n{content}\n"
                for i, (_, start, end, content) in enumerate(st.session_state.corrected_subtitles, 1)
            ])
            st.session_state.edited_changes = st.session_state.changes.copy()

            is_valid, message = validate_srt_format(st.session_state.corrected_srt)
            if not is_valid:
                st.error(f"生成的 SRT 文件格式無效: {message}")
                return

            st.success(f"處理完成！總處理時間：{st.session_state.processing_time:.2f} 秒")
        except Exception as e:
            st.error(f"處理過程中發生錯誤：{str(e)}")
            logger.exception("處理文件時發生異常")

    if st.session_state.get('corrected_srt') is not None:
        st.subheader("修正後的內容")
        st.text_area("", value=st.session_state.corrected_srt, height=300)

        st.download_button(
            "下載修正後的 SRT",
            st.session_state.corrected_srt,
            "corrected.srt",
            "text/plain"
        )

        if st.session_state.get('changes'):
            st.subheader("修正詳情（可編輯）")
            edited_changes = []
            for index, original, corrected in st.session_state.edited_changes:
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1:
                    st.text(f"字幕 {index}")
                with col2:
                    st.text_area(f"原文 {index}", value=original, key=f"original_{index}", height=100)
                with col3:
                    edited = st.text_area(f"修正後 {index}", value=corrected, key=f"corrected_{index}", height=100)
                edited_changes.append((index, original, edited))
            
            st.session_state.edited_changes = edited_changes

            if st.button("應用編輯"):
                updated_srt = update_srt_with_edits(st.session_state.corrected_subtitles, st.session_state.edited_changes)
                st.session_state.corrected_srt = updated_srt
                st.success("已應用您的編輯到 SRT 文件")
                
                st.download_button(
                    "下載編輯後的 SRT",
                    updated_srt,
                    "edited_corrected.srt",
                    "text/plain"
                )
        else:
            st.info("未發現需要修正的內容")

if __name__ == "__main__":
    subtitle_corrector()