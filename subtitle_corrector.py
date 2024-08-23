import streamlit as st
from openai import OpenAI
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import logging

# 设置日志
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
    # 更灵活的正则表达式，可以匹配更多样的时间戳格式
    pattern = re.compile(r'(\d+)\s*\n([0-9:,\.]+)\s*-->\s*([0-9:,\.]+)\s*\n(.*?)\n\s*\n', re.DOTALL)
    return pattern.findall(srt_content.replace('\r\n', '\n').replace('\r', '\n'))

def parse_time(time_str):
    # 处理时间戳中的不同分隔符（逗号或点）
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
            temperature=0.2,
            messages=[
                {"role": "system", "content": f"""
                你是一位精准的字幕校对专家。你的任务是检查并修正字幕中的错误，严格遵守以下规则：
                1. 重点更正地名、人名和民族名的错误。
                2. 修正明显的错别字。
                3. 仅使用提供的修正术语列表中的英文词汇进行更正。即使术语有对应的中文名称，也只使用英文版本进行更正。
                4. 保持原有的繁简体形式，绝对不进行繁简转换。
                5. 不改变原文的语气、语调或风格。
                6. 不添加、删除或重组句子结构以及标点符号。
                7. 如果没有需要更正的内容，完全保留原文。

                修正术语列表： {', '.join(correction_terms)}

                请严格按照这些规则进行校对，确保只进行必要且符合规则的更正。
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
            progress_text.text(f"处理进度: {progress:.2%}")

    # 根据开始时间对字幕进行排序
    corrected_subtitles.sort(key=lambda x: parse_time(x[1]))

    return corrected_subtitles, changes

def validate_srt_format(srt_content):
    subtitles = parse_srt(srt_content)
    if not subtitles:
        return False, "SRT 文件没有包含任何字幕"

    for i, (index, start, end, content) in enumerate(subtitles, 1):
        if not index.strip().isdigit():
            return False, f"无效的字幕编号在第 {i} 个字幕: '{index.strip()}'"
        if not re.match(r'\d{1,2}:\d{2}:\d{2}[,\.]\d{3}', start) or not re.match(r'\d{1,2}:\d{2}:\d{2}[,\.]\d{3}', end):
            return False, f"无效的时间戳格式在第 {i} 个字幕: '{start} --> {end}'"
        if not content.strip():
            return False, f"字幕 {i} 缺少内容"

    return True, f"有效的 SRT 格式，包含 {len(subtitles)} 个字幕"

def update_srt_with_edits(corrected_subtitles, edited_changes):
    # 创建一个字典，键为字幕索引，值为编辑后的内容
    edits_dict = {index: corrected for index, _, corrected in edited_changes}
    
    # 更新字幕内容
    updated_subtitles = [
        (index, start, end, edits_dict.get(index, content))
        for index, start, end, content in corrected_subtitles
    ]
    
    # 格式化更新后的字幕
    formatted_subtitles = []
    for i, (_, start, end, content) in enumerate(updated_subtitles, 1):
        formatted_subtitles.append(f"{i}\n{start} --> {end}\n{content}\n")
    
    return "\n".join(formatted_subtitles)

def subtitle_corrector():
    st.title("SRT 字幕修正器")

    api_key = st.text_input("输入您的 OpenAI API Key", type="password")
    correction_terms = st.text_area("输入修正术语，每行一个", value="\n".join(load_correction_terms()))

    if st.button("保存修正术语"):
        save_correction_terms(correction_terms.split('\n'))
        st.success("修正术语已保存")

    uploaded_file = st.file_uploader("上传 SRT 文件", type="srt")

    if uploaded_file is not None and api_key and st.button("修正字幕"):
        client = OpenAI(api_key=api_key)
        srt_content = uploaded_file.getvalue().decode("utf-8")
        
        # 验证输入的 SRT 格式
        is_valid, message = validate_srt_format(srt_content)
        if not is_valid:
            st.error(f"输入的 SRT 文件格式无效: {message}")
            return

        progress_bar = st.progress(0)
        progress_text = st.empty()

        start_time = time.time()
        try:
            with st.spinner("正在处理..."):
                st.session_state.corrected_subtitles, st.session_state.changes = process_srt(
                    client, 
                    srt_content, 
                    correction_terms.split('\n'), 
                    progress_bar, 
                    progress_text
                )
            st.session_state.processing_time = time.time() - start_time
            
            # 格式化修正后的字幕
            st.session_state.corrected_srt = "\n".join([
                f"{i}\n{start} --> {end}\n{content}\n"
                for i, (_, start, end, content) in enumerate(st.session_state.corrected_subtitles, 1)
            ])
            st.session_state.edited_changes = st.session_state.changes.copy()  # 初始化编辑后的更改

            # 验证输出的 SRT 格式
            is_valid, message = validate_srt_format(st.session_state.corrected_srt)
            if not is_valid:
                st.error(f"生成的 SRT 文件格式无效: {message}")
                return

            st.success(f"处理完成！总处理时间：{st.session_state.processing_time:.2f} 秒")
        except Exception as e:
            st.error(f"处理过程中发生错误：{str(e)}")
            logger.exception("处理文件时发生异常")

    if st.session_state.corrected_srt is not None:
        st.subheader("修正后的内容")
        st.text_area("", value=st.session_state.corrected_srt, height=300)

        st.download_button(
            "下载修正后的 SRT",
            st.session_state.corrected_srt,
            "corrected.srt",
            "text/plain"
        )

        if st.session_state.changes:
            st.subheader("修正详情（可编辑）")
            edited_changes = []
            for index, original, corrected in st.session_state.edited_changes:
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1:
                    st.text(f"字幕 {index}")
                with col2:
                    st.text_area(f"原文 {index}", value=original, key=f"original_{index}", height=100)
                with col3:
                    edited = st.text_area(f"修正后 {index}", value=corrected, key=f"corrected_{index}", height=100)
                edited_changes.append((index, original, edited))
            
            st.session_state.edited_changes = edited_changes

            if st.button("应用编辑"):
                updated_srt = update_srt_with_edits(st.session_state.corrected_subtitles, st.session_state.edited_changes)
                st.session_state.corrected_srt = updated_srt
                st.success("已应用您的编辑到 SRT 文件")
                
                st.download_button(
                    "下载编辑后的 SRT",
                    updated_srt,
                    "edited_corrected.srt",
                    "text/plain"
                )
        else:
            st.info("未发现需要修正的内容")

if __name__ == "__main__":
    subtitle_corrector()