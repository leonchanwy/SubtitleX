import streamlit as st
import re
from openai import OpenAI
import configparser
import chardet
import os
import time

def load_configuration():
    config = configparser.ConfigParser()
    config_file = 'settings.cfg'

    if os.path.exists(config_file):
        with open(config_file, 'rb') as f:
            content = f.read()
            encoding = chardet.detect(content)['encoding']
        config.read(config_file, encoding=encoding)
    else:
        config['settings'] = {'openai_api_key': ''}
        with open(config_file, 'w') as f:
            config.write(f)

    return config

@st.cache_resource
def initialize_openai_client(api_key):
    return OpenAI(api_key=api_key)

def split_text_into_chunks(text, max_chunk_size=1024):
    blocks = re.split(r'(\n\s*\n)', text)
    chunks = []
    current_chunk = ""
    for block in blocks:
        if len(current_chunk + block) <= max_chunk_size:
            current_chunk += block
        else:
            chunks.append(current_chunk)
            current_chunk = block
    chunks.append(current_chunk)
    return chunks

def correct_subtitle_chunk(text, client, correction_terms):
    max_retries = 3
    for _ in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": f"""
Correct the following SRT subtitle block using the provided correction terms.
Maintain the original subtitle numbering and timing.
Only correct errors related to the provided terms. Do not make any other changes.
Correction terms: {', '.join(correction_terms)}
"""},
                    {"role": "user", "content": text},
                    {"role": "user", "content": "correct:"}
                ]
            )
            corrected_text = completion.choices[0].message.content
            return corrected_text.strip()
        except Exception as e:
            st.error(f"修正錯誤：{e}")
    return text

def process_srt_file(file, client, correction_terms):
    start_time = time.time()
    content = file.getvalue().decode("utf-8")
    text_chunks = split_text_into_chunks(content)

    corrected_text = ""
    progress_bar = st.progress(0)
    for i, chunk in enumerate(text_chunks):
        corrected_chunk = correct_subtitle_chunk(chunk, client, correction_terms)
        corrected_text += f"{corrected_chunk}\n\n"
        progress_bar.progress((i + 1) / len(text_chunks))

    end_time = time.time()
    processing_time = end_time - start_time
    return corrected_text, processing_time

def subtitle_corrector():
    st.title("字幕錯字修正器")

    st.write("這個工具可以幫助您修正 SRT 格式字幕文件中的錯字。")

    with st.expander("點擊展開查看詳細說明"):
        st.markdown("""
        ### 應用簡介
        這個工具專為修正字幕中的錯字而設計：

        1. **錯字修正**：根據提供的正確名詞列表，修正字幕中的錯誤。
        2. **保持格式**：保留原始 SRT 文件的時間戳和編號。
        3. **自定義詞彙**：允許用戶提供自定義的正確名詞列表。

        ### 主要功能
        - 上傳 SRT 格式的字幕文件
        - 提供正確名詞列表
        - 使用 OpenAI API 進行高質量錯字修正
        - 提供修正結果預覽
        - 下載修正後的 SRT 文件

        ### 使用步驟
        1. 輸入您的 OpenAI API Key。
        2. 提供正確名詞列表。
        3. 上傳需要修正的 SRT 文件。
        4. 點擊「修正」按鈕開始處理。
        5. 查看修正預覽。
        6. 下載修正後的 SRT 文件。

        ### 注意事項
        - 確保您有有效的 OpenAI API Key。
        - 上傳的 SRT 文件應為標準格式。
        - 修正過程可能需要一些時間，請耐心等待。
        - 修正後的 SRT 文件使用 UTF-8 編碼。
        """)

    # Initialize session state
    if 'corrected_srt' not in st.session_state:
        st.session_state.corrected_srt = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None

    # Load configuration
    config = load_configuration()
    default_api_key = config.get('settings', 'openai_api_key', fallback='')

    api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")
    correction_terms = st.text_area(
        "正確名詞列表（每行一個）",
        "Bidayuh\nIban\nMelanau\nKayan\nKenyah\nLun Bawang\nLong Sukang\nDusun\nRungus\nOrang Sungai\nBajau\nSarawak\nKuching\nSibu\nBintulu\nNiah\nMiri\nLimbang\nLong San\nSarikei\nTelok Melano\nSabah\nKota Kinabalu\nKudat\nSandakan\nTawau\nSemporna\nKundasang\nUma Belor\nCCY\n種族奇事\nGunung Kinabalu\nRoad Trip\n東馬\n沙巴\n沙拉越"
    ).split('\n')

    # Initialize OpenAI client
    client = initialize_openai_client(api_key)

    # File upload
    uploaded_file = st.file_uploader("選擇一個 SRT 文件", type="srt")

    # Save original filename
    original_filename = None
    if uploaded_file is not None:
        original_filename = uploaded_file.name

    # Correction button
    if uploaded_file is not None:
        if st.button("修正"):
            with st.spinner("正在修正..."):
                st.session_state.corrected_srt, st.session_state.processing_time = process_srt_file(
                    uploaded_file, client, correction_terms)
                st.success("修正完成！")

    # Display results and download button (if correction is available)
    if st.session_state.corrected_srt is not None:
        st.subheader("修正預覽")
        st.text_area("", value=st.session_state.corrected_srt[:1000] + "...", height=300)

        # Generate new filename
        if original_filename:
            new_filename = f"{os.path.splitext(original_filename)[0]}_corrected.srt"
        else:
            new_filename = "corrected.srt"

        # Download button
        st.download_button(
            label="下載修正後的 SRT",
            data=st.session_state.corrected_srt,
            file_name=new_filename,
            mime="text/plain"
        )

    # Display processing time at the bottom in small font
    if st.session_state.processing_time is not None:
        st.markdown(f"<p style='font-size: 10px; text-align: right;'>總處理時間：{st.session_state.processing_time:.2f} 秒</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    subtitle_corrector()