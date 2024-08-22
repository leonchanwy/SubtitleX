import streamlit as st
from openai import OpenAI
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging
import opencc

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
            messages=[
                {"role": "system", "content": f"""
Please correct the following subtitle content. Only correct errors related to the provided correction terms.
If there's nothing to correct, return the original text. Do not add any explanations or comments.

Correction terms: {', '.join(correction_terms)}
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
                changes.append((original, corrected))
            progress = (i + 1) / len(subtitles)
            progress_bar.progress(progress)
            progress_text.text(f"處理進度: {progress:.2%}")

    # 根據開始時間對字幕進行排序
    corrected_subtitles.sort(key=lambda x: parse_time(x[1]))

    # 重新編號字幕
    formatted_subtitles = []
    for i, (_, start, end, content) in enumerate(corrected_subtitles, 1):
        formatted_subtitles.append(f"{i}\n{start} --> {end}\n{content}\n")

    return "\n".join(formatted_subtitles), changes

def validate_srt_format(srt_content):
    lines = srt_content.split('\n')
    i = 0
    while i < len(lines):
        # 跳过空行
        if not lines[i].strip():
            i += 1
            continue
        
        # 檢查字幕編號
        if not lines[i].strip().isdigit():
            return False, f"Invalid subtitle number at line {i+1}"
        i += 1
        
        # 檢查時間戳格式
        if i >= len(lines) or not re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', lines[i]):
            return False, f"Invalid timestamp format at line {i+1}"
        i += 1
        
        # 檢查字幕內容
        if i >= len(lines):
            return False, "Missing subtitle content"
        while i < len(lines) and lines[i].strip():
            i += 1
        i += 1
    
    return True, "Valid SRT format"


def convert_simplified_to_traditional(text):
    converter = opencc.OpenCC('s2t')  # s2t 表示简体到繁体
    return converter.convert(text)

def subtitle_corrector():
    st.title("SRT 字幕修正器")

    st.markdown("""
    ### 使用說明
    1. 輸入您的 OpenAI API Key。
    2. 添加或編輯修正術語，每行一個。
    3. 上傳 SRT 格式的字幕文件。
    4. 點擊"修正字幕"按鈕開始處理。
    5. 查看修正結果和詳情。
    6. 下載修正後的 SRT 文件。
    """)

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
                st.session_state.corrected_srt, st.session_state.changes = process_srt(
                    client, 
                    srt_content, 
                    correction_terms.split('\n'), 
                    progress_bar, 
                    progress_text
                )
            
            # 将修正后的内容从简体转换为繁体
            st.session_state.corrected_srt = convert_simplified_to_traditional(st.session_state.corrected_srt)

            st.session_state.processing_time = time.time() - start_time

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
            st.subheader("修正詳情")
            changes_df = pd.DataFrame(st.session_state.changes, columns=["原文", "修正後"])
            st.dataframe(changes_df)
        else:
            st.info("未發現需要修正的內容")

if __name__ == "__main__":
    subtitle_corrector()