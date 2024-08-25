import streamlit as st
import re
from openai import OpenAI
import time
import os

def init_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'translated_srt' not in st.session_state:
        st.session_state.translated_srt = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.session_state.api_key)

def split_text_into_chunks(text, max_chunk_size=1024):
    return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

def translate_subtitle_chunk(text, client, first_language, second_language, translation_instructions):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": f"""
Translate the following SRT subtitle block to {first_language} and {second_language}.
Follow this exact format for each subtitle:

[subtitle number]
[timestamp]
[{first_language} translation]
[{second_language} translation]

Maintain the original subtitle numbering and timing. {translation_instructions}
Do not include any introductory or explanatory text in your response.
"""},
                {"role": "user", "content": text},
                {"role": "user", "content": "translate:"}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"翻譯錯誤：{e}")
        return text

def process_srt_file(file, client, first_language, second_language, translation_instructions):
    start_time = time.time()
    content = file.getvalue().decode("utf-8")
    text_chunks = split_text_into_chunks(content)

    translated_text = ""
    progress_bar = st.progress(0)
    for i, chunk in enumerate(text_chunks):
        translated_chunk = translate_subtitle_chunk(chunk, client, first_language, second_language, translation_instructions)
        translated_text += f"{translated_chunk}\n\n"
        progress_bar.progress((i + 1) / len(text_chunks))

    end_time = time.time()
    return translated_text, end_time - start_time

def validate_srt_format(srt_content):
    lines = srt_content.strip().split('\n')
    subtitle_count = sum(1 for line in lines if line.strip().isdigit())
    is_valid = all(re.match(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n.+\n.+\n', block) 
                   for block in re.split(r'\n\s*\n', srt_content))
    return is_valid, f"SRT 文件格式{'正確' if is_valid else '不正確'}。共有 {subtitle_count} 個字幕。"

def bilingual_srt_translator():

    st.write("這個工具可以幫助您將 SRT 格式的字幕文件翻譯成雙語字幕。")

    init_session_state()

    with st.expander("使用說明"):
        st.markdown("""
        1. 輸入您的 OpenAI API Key。
        2. 指定第一語言和第二語言。
        3. 提供自定義翻譯提示（可選）。
        4. 上傳 SRT 文件。
        5. 點擊「翻譯」按鈕開始處理。
        6. 查看翻譯預覽和驗證結果。
        7. 下載雙語 SRT 文件。
        """)

    st.session_state.api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
    client = get_openai_client()

    col1, col2 = st.columns(2)
    with col1:
        first_language = st.text_input("第一語言", value="traditional chinese")
    with col2:
        second_language = st.text_input("第二語言", value="english")

    translation_instructions = st.text_area("自定義翻譯提示", "第一個語言要用台灣式口語，第二個語言要用美式口語")

    uploaded_file = st.file_uploader("選擇一個 SRT 文件", type="srt")

    if uploaded_file and st.button("翻譯"):
        if not st.session_state.api_key:
            st.error("請輸入有效的 OpenAI API Key。")
        elif not first_language or not second_language:
            st.error("請輸入兩種語言。")
        else:
            with st.spinner("正在翻譯..."):
                st.session_state.translated_srt, st.session_state.processing_time = process_srt_file(
                    uploaded_file, client, first_language, second_language, translation_instructions)
                
                is_valid, validation_message = validate_srt_format(st.session_state.translated_srt)
                if is_valid:
                    st.success("翻譯完成！SRT 格式驗證通過。")
                else:
                    st.warning("翻譯完成，但 SRT 格式可能有問題。")
                st.info(validation_message)

    if st.session_state.translated_srt:
        st.subheader("翻譯預覽")
        st.text_area("", value=st.session_state.translated_srt[:1000] + "...", height=300)

        new_filename = f"{os.path.splitext(uploaded_file.name)[0]}_bilingual.srt" if uploaded_file else "bilingual.srt"
        st.download_button("下載雙語 SRT", st.session_state.translated_srt, file_name=new_filename, mime="text/plain")

    if st.session_state.processing_time:
        st.markdown(f"<p style='font-size: 10px; text-align: right;'>處理時間：{st.session_state.processing_time:.2f} 秒</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    bilingual_srt_translator()