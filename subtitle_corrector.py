import streamlit as st
from openai import OpenAI
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
TERMS_FILE = 'correction_terms.json'
MAX_WORKERS = 5
MODEL_NAME = "gpt-4o-2024-08-06"

# 初始化 session state
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

def load_correction_terms():
    try:
        with open(TERMS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.info(f"File {TERMS_FILE} not found, returning empty list")
        return []

def save_correction_terms(terms):
    with open(TERMS_FILE, 'w', encoding='utf-8') as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    logger.info(f"Correction terms saved to {TERMS_FILE}")

def parse_srt(srt_content):
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.*\n)*?)\n')
    return pattern.findall(srt_content)

def parse_time(time_str):
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000

def correct_subtitle(client, subtitle, correction_terms):
    index, start, end, content = subtitle
    original_content = content.strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.2,
            messages=[
                {"role": "system", "content": f"""
                你是一位精準的字幕校對專家。你的任務是檢查並修正字幕中的錯誤，嚴格遵守以下規則：
                1. 重點更正地名、人名和民族名的錯誤。
                2. 修正明顯的錯別字。
                3. 僅使用提供的修正術語列表中的英文詞彙進行更正。即使術語有對應的中文名稱，也只使用英文版本進行更正。
                4. 保持原有的繁簡體形式，絕對不進行繁簡轉換。
                5. 不改變原文的語氣、語調或風格。
                6. 不添加、刪除或重組句子結構以及標點符號。
                7. 如果沒有需要更正的內容，完全保留原文。

                修正術語列表： {', '.join(correction_terms)}

                請嚴格按照這些規則進行校對，確保只進行必要且符合規則的更正。
                """},
                {"role": "user", "content": original_content},
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

    # 根據開始時間對字幕進行排序
    corrected_subtitles.sort(key=lambda x: parse_time(x[1]))

    return corrected_subtitles, changes

def validate_srt_format(srt_content):
    lines = srt_content.split('\n')
    i = 0
    subtitle_count = 0

    while i < len(lines):
        # 跳過空行
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i >= len(lines):
            break

        # 檢查字幕編號，允許任何非空字符串
        if not lines[i].strip():
            return False, f"預期字幕編號，但在第 {i+1} 行找到空行"
        subtitle_count += 1
        i += 1

        # 檢查是否還有足夠的行數
        if i >= len(lines):
            return False, f"字幕 {subtitle_count} 不完整，缺少時間戳或內容"

        # 檢查時間戳格式，使用更寬鬆的正則表達式
        if not re.search(r'\d+:\d+:\d+[,\.]\d+\s*-->\s*\d+:\d+:\d+[,\.]\d+', lines[i], re.IGNORECASE):
            return False, f"無效的時間戳格式在第 {i+1} 行: '{lines[i].strip()}'"
        i += 1

        # 檢查字幕內容
        content_found = False
        while i < len(lines):
            if lines[i].strip():
                content_found = True
                i += 1
            elif content_found:
                break
            else:
                i += 1
        
        if not content_found:
            return False, f"字幕 {subtitle_count} 缺少內容"

    if subtitle_count == 0:
        return False, "SRT 文件沒有包含任何字幕"

    return True, f"有效的 SRT 格式，包含 {subtitle_count} 個字幕"

def update_srt_with_edits(corrected_subtitles, edited_changes):
    # 創建一個字典，鍵為字幕索引，值為編輯後的內容
    edits_dict = {index: corrected for index, _, corrected in edited_changes}
    
    # 更新字幕內容
    updated_subtitles = [
        (index, start, end, edits_dict.get(index, content))
        for index, start, end, content in corrected_subtitles
    ]
    
    # 格式化更新後的字幕
    formatted_subtitles = []
    for i, (_, start, end, content) in enumerate(updated_subtitles, 1):
        formatted_subtitles.append(f"{i}\n{start} --> {end}\n{content}\n")
    
    return "\n".join(formatted_subtitles)

def subtitle_corrector():
    st.title("SRT 字幕修正器")

    api_key = st.text_input("輸入您的 OpenAI API Key", type="password")
    correction_terms = st.text_area("輸入修正術語，每行一個", value="\n".join(load_correction_terms()))

    if st.button("保存修正術語"):
        save_correction_terms(correction_terms.split('\n'))
        st.success("修正術語已保存")

    uploaded_file = st.file_uploader("上傳 SRT 文件", type="srt")

    if uploaded_file is not None and api_key and st.button("修正字幕"):
        client = OpenAI(api_key=api_key)
        srt_content = uploaded_file.getvalue().decode("utf-8")
        
        # 驗證輸入的 SRT 格式
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
            
            # 格式化修正後的字幕
            st.session_state.corrected_srt = "\n".join([
                f"{i}\n{start} --> {end}\n{content}\n"
                for i, (_, start, end, content) in enumerate(st.session_state.corrected_subtitles, 1)
            ])
            st.session_state.edited_changes = st.session_state.changes.copy()  # 初始化編輯後的更改

            # 驗證輸出的 SRT 格式
            is_valid, message = validate_srt_format(st.session_state.corrected_srt)
            if not is_valid:
                st.error(f"生成的 SRT 文件格式無效: {message}")
                return

            st.success(f"處理完成！總處理時間：{st.session_state.processing_time:.2f} 秒")
        except Exception as e:
            st.error(f"處理過程中發生錯誤：{str(e)}")
            logger.exception("處理文件時發生異常")

    if st.session_state.corrected_srt is not None:
        st.subheader("修正後的內容")
        st.text_area("", value=st.session_state.corrected_srt, height=300)

        st.download_button(
            "下載修正後的 SRT",
            st.session_state.corrected_srt,
            "corrected.srt",
            "text/plain"
        )

        if st.session_state.changes:
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