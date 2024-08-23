import streamlit as st
import re
from openai import OpenAI
import configparser
import os
import time
from pydantic import BaseModel
from typing import List, Callable
import logging
from io import StringIO

# 定義數據模型
class SubtitleBlock(BaseModel):
    number: int
    timestamp: str
    original: str
    first_language: str
    second_language: str

class TranslatedSubtitles(BaseModel):
    subtitles: List[SubtitleBlock]

class SubtitleTranslationRequest(BaseModel):
    subtitle_blocks: List[str]
    first_language: str
    second_language: str
    instructions: str

# 配置加載函數
def load_configuration():
    config = configparser.ConfigParser()
    config_file = 'settings.cfg'

    if os.path.exists(config_file):
        config.read(config_file, encoding='utf-8-sig')
    else:
        config['settings'] = {'openai_api_key': '', 'default_first_language': '', 'default_second_language': ''}
        with open(config_file, 'w', encoding='utf-8') as f:
            config.write(f)

    return config

# OpenAI 客戶端初始化
@st.cache_resource
def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

# SRT 解析函數
def parse_srt_block(block):
    lines = block.strip().split('\n')
    if len(lines) < 3:
        return None
    number = int(lines[0].strip().lstrip('\ufeff'))  # 移除可能的 BOM
    timestamp = lines[1]
    content = '\n'.join(lines[2:])
    return SubtitleBlock(number=number, timestamp=timestamp, original=content, first_language="", second_language="")

# 文本分塊函數
def split_text_into_chunks(text, max_chunk_size=4000):
    blocks = text.strip().split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0

    for block in blocks:
        block_size = len(block)
        if current_size + block_size > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [block]
            current_size = block_size
        else:
            current_chunk.append(block)
            current_size += block_size

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks

# 字幕翻譯函數
def translate_subtitle_chunk(text, client, first_language, second_language, translation_instructions, logger):
    subtitle_blocks = [parse_srt_block(block) for block in text.strip().split('\n\n') if parse_srt_block(block)]
    
    request = SubtitleTranslationRequest(
        subtitle_blocks=[f"{block.number}\n{block.timestamp}\n{block.original}" for block in subtitle_blocks],
        first_language=first_language,
        second_language=second_language,
        instructions=translation_instructions
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"正在發送請求到 OpenAI API (嘗試 {attempt + 1}/{max_retries})")
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": f"""
翻譯以下 SRT 字幕塊為 {first_language} 和 {second_language}。
對於每個字幕塊：
1. 保留原始編號和時間戳。
2. 保留原始文本。
3. 將原始文本翻譯為 {first_language}。
4. 將原始文本翻譯為 {second_language}。
保持原始字幕編號和時間。{translation_instructions}
請以 JSON 格式回應，遵循以下結構：
{{
  "subtitles": [
    {{
      "number": 整數,
      "timestamp": "字符串",
      "original": "字符串",
      "first_language": "字符串",
      "second_language": "字符串"
    }}
  ]
}}
請勿在回應中包含任何介紹性或解釋性文本。
"""},
                    {"role": "user", "content": str(request.dict())}
                ],
                response_format={"type": "json_object"}
            )

            logger.info("已收到 OpenAI API 的回應")
            translated_subtitles = TranslatedSubtitles.parse_raw(completion.choices[0].message.content)
            
            return "\n\n".join([
                f"{block.number}\n{block.timestamp}\n{block.original}\n{block.first_language}\n{block.second_language}"
                for block in translated_subtitles.subtitles
            ])

        except Exception as e:
            logger.error(f"翻譯錯誤，嘗試 {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                raise

    logger.warning("達到最大重試次數，返回原始文本")
    return text

# SRT 格式驗證函數
def validate_srt_format(srt_content):
    lines = srt_content.strip().split('\n')
    errors = []
    subtitle_count = 0
    line_number = 0

    while line_number < len(lines):
        subtitle_count += 1

        if line_number >= len(lines) or not lines[line_number].strip().lstrip('\ufeff').isdigit():
            errors.append(f"錯誤：第 {line_number + 1} 行，預期為字幕編號 {subtitle_count}")
            break

        line_number += 1
        if line_number >= len(lines) or not re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', lines[line_number]):
            errors.append(f"錯誤：第 {line_number + 1} 行，時間戳格式不正確")
            break

        line_number += 1
        if line_number >= len(lines) or not lines[line_number].strip():
            errors.append(f"錯誤：第 {line_number + 1} 行，原文缺失")
            break

        line_number += 1
        if line_number >= len(lines) or not lines[line_number].strip():
            errors.append(f"錯誤：第 {line_number + 1} 行，第一語言翻譯缺失")
            break

        line_number += 1
        if line_number >= len(lines) or not lines[line_number].strip():
            errors.append(f"錯誤：第 {line_number + 1} 行，第二語言翻譯缺失")
            break

        line_number += 1
        if line_number < len(lines) and lines[line_number].strip():
            errors.append(f"錯誤：第 {line_number + 1} 行，預期為空行")
            break

        line_number += 1

    if not errors:
        return True, f"SRT 文件格式正確。共有 {subtitle_count} 個字幕。"
    else:
        return False, "\n".join(errors)

# 處理 SRT 文件的主函數
def process_srt_file(file, client, first_language, second_language, translation_instructions, logger, progress_callback: Callable[[float], None]):
    start_time = time.time()
    content = file.getvalue().decode("utf-8-sig")  # 使用 utf-8-sig 來處理可能的 BOM
    text_chunks = split_text_into_chunks(content)
    total_chunks = len(text_chunks)

    translated_text = ""
    for i, chunk in enumerate(text_chunks):
        logger.info(f"正在處理第 {i+1}/{total_chunks} 塊")
        translated_chunk = translate_subtitle_chunk(chunk, client, first_language, second_language, translation_instructions, logger)
        translated_text += f"{translated_chunk}\n\n"
        progress_callback((i + 1) / total_chunks)

    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"總處理時間: {processing_time:.2f} 秒")
    return translated_text, processing_time

# 主 Streamlit 應用函數
def bilingual_srt_translator():
    st.title("雙語字幕翻譯器")

    st.write("這個工具可以幫助您將 SRT 格式的字幕文件翻譯成雙語字幕。")

    # 設置日誌
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    log_output = StringIO()
    handler = logging.StreamHandler(log_output)
    logger.addHandler(handler)
    log_container = st.empty()

    # 加載配置
    config = load_configuration()
    default_api_key = config.get('settings', 'openai_api_key', fallback='')
    default_first_language = config.get('settings', 'default_first_language', fallback='traditional chinese')
    default_second_language = config.get('settings', 'default_second_language', fallback='malay')

    # 用戶輸入
    api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")
    first_language = st.text_input("第一語言", value=default_first_language)
    second_language = st.text_input("第二語言", value=default_second_language)
    translation_instructions = st.text_area(
        "自定義翻譯提示",
        "第一個語言是台灣式口語加上髒話、第二個語言要極度口語"
    )

    # 初始化 OpenAI 客戶端
    try:
        client = initialize_openai_client(api_key)
    except Exception as e:
        st.error(f"初始化 OpenAI 客戶端時出錯: {str(e)}")
        return

    # 文件上傳
    uploaded_file = st.file_uploader("選擇一個 SRT 文件", type="srt")

    if uploaded_file is not None:
        if st.button("翻譯"):
            if not first_language or not second_language:
                st.error("請在翻譯之前輸入兩種語言。")
            else:
                try:
                    with st.spinner("正在翻譯..."):
                        logger.info(f"開始翻譯: {uploaded_file.name}")
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def update_progress(progress):
                            progress_bar.progress(progress)
                            status_text.text(f"已完成 {progress:.1%}")

                        translated_srt, processing_time = process_srt_file(
                            uploaded_file, client, first_language, second_language, 
                            translation_instructions, logger, update_progress
                        )

                        # 驗證翻譯後的 SRT
                        is_valid, validation_message = validate_srt_format(translated_srt)
                        if is_valid:
                            logger.info("翻譯成功完成")
                            st.success("翻譯完成！SRT 格式驗證通過。")
                            st.info(validation_message)
                        else:
                            logger.warning("翻譯完成，但 SRT 格式驗證失敗")
                            st.warning("翻譯完成，但 SRT 格式可能有問題。")
                            st.error(validation_message)

                        # 顯示翻譯預覽
                        st.subheader("翻譯預覽")
                        st.text_area("", value=translated_srt[:1000] + "...", height=300)

                        # 下載按鈕
                        new_filename = f"{uploaded_file.name.split('.')[0]}_bilingual.srt"
                        st.download_button(
                            label="下載雙語 SRT",
                            data=translated_srt,
                            file_name=new_filename,
                            mime="text/plain"
                        )

                        # 顯示處理時間
                        st.markdown(f"<p style='font-size: 10px; text-align: right;'>總處理時間：{processing_time:.2f} 秒</p>", unsafe_allow_html=True)

                except Exception as e:
                    logger.error(f"翻譯過程中發生錯誤: {str(e)}")
                    st.error(f"翻譯過程中發生錯誤: {str(e)}")

    # 顯示日誌
    if st.checkbox("顯示詳細日誌"):
        log_container.code(log_output.getvalue())
    else:
        log_container.empty()

if __name__ == "__main__":
    bilingual_srt_translator()