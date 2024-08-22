import streamlit as st
import re
from openai import OpenAI
import configparser
import chardet
import os
import time

def load_config():
    config = configparser.ConfigParser()
    config_file = 'settings.cfg'

    if os.path.exists(config_file):
        with open(config_file, 'rb') as f:
            content = f.read()
            encoding = chardet.detect(content)['encoding']
        config.read(config_file, encoding=encoding)
    else:
        config['option'] = {'openai-apikey': '', 'target-language': ''}
        with open(config_file, 'w') as f:
            config.write(f)

    return config

@st.cache_resource
def get_openai_client(api_key):
    return OpenAI(api_key=api_key)

def split_text(text):
    blocks = re.split(r'(\n\s*\n)', text)
    short_text_list = []
    short_text = ""
    for block in blocks:
        if len(short_text + block) <= 1024:
            short_text += block
        else:
            short_text_list.append(short_text)
            short_text = block
    short_text_list.append(short_text)
    return short_text_list

def is_translation_valid(original_text, translated_text):
    original_blocks = original_text.strip().split('\n\n')
    translated_blocks = translated_text.strip().split('\n\n')

    if len(original_blocks) != len(translated_blocks):
        return False

    for orig_block, trans_block in zip(original_blocks, translated_blocks):
        orig_lines = orig_block.split('\n')
        trans_lines = trans_block.split('\n')

        if len(trans_lines) != 4:
            return False

        if orig_lines[0] != trans_lines[0] or orig_lines[1] != trans_lines[1]:
            return False

    return True

def translate_text(text, client, first_language, second_language, custom_prompt):
    max_retries = 3
    for _ in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"""
Translate the following SRT subtitle block to {first_language} and {second_language}.
Follow this exact format for each subtitle:

[subtitle number]
[timestamp]
[{first_language} translation]
[{second_language} translation]

Maintain the original subtitle numbering and timing. {custom_prompt}
Do not include any introductory or explanatory text in your response.
"""},
                    {"role": "user", "content": text},
                    {"role": "user", "content": "translate:"}
                ]
            )
            translated_text = completion.choices[0].message.content

            filtered_lines = []
            current_block = []
            for line in translated_text.split('\n'):
                line = line.strip()
                if line and not line.startswith("Sure,") and not line.startswith("以下是"):
                    current_block.append(line)
                    if len(current_block) == 4:
                        filtered_lines.extend(current_block)
                        filtered_lines.append('')
                        current_block = []

            filtered_text = '\n'.join(filtered_lines).strip()

            if is_translation_valid(text, filtered_text):
                return filtered_text
        except Exception as e:
            st.error(f"翻譯錯誤：{e}")
    return text

def validate_srt(srt_content):
    lines = srt_content.strip().split('\n')
    errors = []
    subtitle_count = 0
    line_number = 0

    while line_number < len(lines):
        subtitle_count += 1

        if line_number >= len(lines) or not lines[line_number].strip().isdigit():
            errors.append(f"錯誤：第 {line_number + 1} 行，預期為字幕編號 {subtitle_count}")
            break

        line_number += 1
        if line_number >= len(lines) or not re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', lines[line_number]):
            errors.append(f"錯誤：第 {line_number + 1} 行，時間戳格式不正確")
            break

        line_number += 1
        if line_number >= len(lines) or not lines[line_number].strip():
            errors.append(f"錯誤：第 {line_number + 1} 行，第一語言字幕缺失")
            break

        line_number += 1
        if line_number >= len(lines) or not lines[line_number].strip():
            errors.append(f"錯誤：第 {line_number + 1} 行，第二語言字幕缺失")
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

def process_srt_file(file, client, first_language, second_language, custom_prompt):
    start_time = time.time()
    content = file.getvalue().decode("utf-8")
    short_text_list = split_text(content)

    translated_text = ""
    progress_bar = st.progress(0)
    for i, short_text in enumerate(short_text_list):
        translated_short_text = translate_text(short_text, client, first_language, second_language, custom_prompt)
        translated_text += f"{translated_short_text}\n\n"
        progress_bar.progress((i + 1) / len(short_text_list))

    end_time = time.time()
    processing_time = end_time - start_time
    return translated_text, processing_time

def bilingual_srt_translator():
    st.title("雙語字幕翻譯器")

    st.write("這個工具可以幫助您將 SRT 格式的字幕文件翻譯成雙語字幕。")

    with st.expander("點擊展開查看詳細說明"):
        st.markdown("""
        ### 應用簡介
        這個工具專為處理雙語字幕翻譯而設計，能夠將單一語言的 SRT 字幕文件轉換為包含兩種語言的雙語字幕：

        1. **雙語翻譯**：將原始字幕翻譯成兩種指定的語言。
        2. **保持格式**：保留原始 SRT 文件的時間戳和編號，確保與視頻同步。
        3. **自定義提示**：允許用戶提供自定義翻譯提示，以獲得更符合特定需求的翻譯結果。

        ### 主要功能
        - 上傳 SRT 格式的字幕文件
        - 指定兩種目標翻譯語言
        - 提供自定義翻譯提示
        - 使用 OpenAI API 進行高質量翻譯
        - 驗證翻譯後的 SRT 格式
        - 提供翻譯結果預覽
        - 下載雙語 SRT 文件

        ### 使用步驟
        1. 輸入您的 OpenAI API Key。
        2. 指定第一語言和第二語言。
        3. （可選）提供自定義翻譯提示。
        4. 上傳需要翻譯的 SRT 文件。
        5. 點擊「翻譯」按鈕開始處理。
        6. 查看翻譯預覽和驗證結果。
        7. 下載雙語 SRT 文件。

        ### 注意事項
        - 確保您有有效的 OpenAI API Key。
        - 上傳的 SRT 文件應為標準格式。
        - 翻譯過程可能需要一些時間，請耐心等待。
        - 自定義提示可以幫助您獲得更符合特定風格或需求的翻譯。
        - 翻譯後的 SRT 文件使用 UTF-8 編碼，確保與大多數現代系統兼容。
        """)

    # Initialize session state
    if 'translated_srt' not in st.session_state:
        st.session_state.translated_srt = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None

    # Load configuration
    config = load_config()
    default_api_key = config.get('option', 'openai-apikey', fallback='')

    api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")
    first_language = st.text_input("第一語言", value="traditional chinese")
    second_language = st.text_input("第二語言", value="malay")
    custom_prompt = st.text_area(
        "自定義翻譯提示",
        "第一個語言是台灣式口語加上髒話、第二個語言要極度口語"
    )

    # Initialize OpenAI client
    client = get_openai_client(api_key)

    # File upload
    uploaded_file = st.file_uploader("選擇一個 SRT 文件", type="srt")

    # Translation button
    if uploaded_file is not None:
        if st.button("翻譯"):
            if not first_language or not second_language:
                st.error("請在翻譯之前輸入兩種語言。")
            else:
                with st.spinner("正在翻譯..."):
                    st.session_state.translated_srt, st.session_state.processing_time = process_srt_file(
                        uploaded_file, client, first_language, second_language, custom_prompt)

                    # Validate translated SRT
                    is_valid, validation_message = validate_srt(st.session_state.translated_srt)
                    if is_valid:
                        st.success("翻譯完成！SRT 格式驗證通過。")
                        st.info(validation_message)
                    else:
                        st.warning("翻譯完成，但 SRT 格式可能有問題。")
                        st.error(validation_message)

    # Display results and download button (if translation is available)
    if st.session_state.translated_srt is not None:
        st.subheader("翻譯預覽")
        st.text_area("", value=st.session_state.translated_srt[:1000] + "...", height=300)

        # Download button
        st.download_button(
            label="下載雙語 SRT",
            data=st.session_state.translated_srt,
            file_name="dual_language.srt",
            mime="text/plain"
        )

    # Display processing time at the bottom in small font
    if st.session_state.processing_time is not None:
        st.markdown(f"<p style='font-size: 10px; text-align: right;'>總處理時間：{st.session_state.processing_time:.2f} 秒</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    bilingual_srt_translator()