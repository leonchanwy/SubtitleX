import streamlit as st
from openai import OpenAI
import time
from ai_subtitle_generator import ai_subtitle_generator
from subtitle_time_sync import subtitle_time_sync
from bilingual_subtitle_resizer import bilingual_subtitle_resizer
from bilingual_srt_translator import bilingual_srt_translator
from multi_language_subtitle_translator import multi_language_subtitle_translator
from subtitle_corrector import subtitle_corrector
from whisper_api_tool import whisper_api_tool

def init_session_state():
    if 'api_key' not in st.session_state:
        st.session_state['api_key'] = ""
    if 'api_key_valid' not in st.session_state:
        st.session_state['api_key_valid'] = False
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'name' not in st.session_state:
        st.session_state['name'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None

# def validate_api_key(api_key):
#     try:
#         client = OpenAI(api_key=api_key)
#         client.models.list()
#         return True
#     except Exception:
#         return False

# def api_key_input():
#     api_key = st.sidebar.text_input(
#         "輸入您的 OpenAI API Key",
#         value=st.session_state['api_key'],
#         type="password",
#         key="api_key_input"
#     )
    
#     if api_key != st.session_state['api_key']:
#         st.session_state['api_key'] = api_key
#         if api_key:
#             if validate_api_key(api_key):
#                 st.sidebar.success("API Key 有效")
#                 st.session_state['api_key_valid'] = True
#             else:
#                 st.sidebar.error("無效的 API Key")
#                 st.session_state['api_key_valid'] = False
#         else:
#             st.session_state['api_key_valid'] = False

def main():
    st.set_page_config(page_title="剪接神器", layout="wide")
    init_session_state()

    st.markdown(
        """
        <script>
        var api_key = localStorage.getItem('openai_api_key');
        if (api_key) {
            document.querySelector('input[type="password"]').value = api_key;
            document.querySelector('input[type="password"]').dispatchEvent(new Event('input'));
        }
        </script>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("導航")

    # api_key_input()

    # if not st.session_state['api_key_valid']:
    #     st.warning("請在側邊欄輸入有效的 OpenAI API Key 以使用需要 API 的功能")

    page = st.sidebar.selectbox(
        "選擇功能",
        ["AI 生成字幕", "字幕時間同步", "雙語字幕大小調整", 
         "雙語字幕翻譯", "多語言字幕翻譯", "字幕錯字修正",
         "Whisper API Tool"]
    )

    if page == "AI 生成字幕":
        ai_subtitle_generator()
    elif page == "字幕時間同步":
        subtitle_time_sync()
    elif page == "雙語字幕大小調整":
        bilingual_subtitle_resizer()
    elif page == "雙語字幕翻譯":
        bilingual_srt_translator()
    elif page == "多語言字幕翻譯":
        multi_language_subtitle_translator()
    elif page == "字幕錯字修正":
        subtitle_corrector()
    elif page == "Whisper API Tool":
        whisper_api_tool()

    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 剪接神器. All rights reserved.")

if __name__ == "__main__":
    main()