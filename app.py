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
        st.session_state.api_key = ""
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False

def validate_api_key(api_key):
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True
    except Exception:
        return False

def api_key_input():
    # 從localStorage獲取API Key
    api_key = st.sidebar.text_input(
        "輸入您的 OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        key="api_key_input"
    )
    
    message_placeholder = st.sidebar.empty()
    
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        if api_key:
            if validate_api_key(api_key):
                message_placeholder.success("API Key 有效")
                st.session_state.api_key_valid = True
                # 將有效的API Key保存到localStorage
                st.markdown(
                    """
                    <script>
                    localStorage.setItem('openai_api_key', '{}');
                    </script>
                    """.format(api_key),
                    unsafe_allow_html=True
                )
                time.sleep(0.5)
                message_placeholder.empty()
            else:
                message_placeholder.error("無效的 API Key")
                st.session_state.api_key_valid = False
        else:
            st.session_state.api_key_valid = False

def main():
    st.set_page_config(page_title="剪接神器", layout="wide")
    init_session_state()

    # 從localStorage讀取API Key
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

    st.sidebar.title("剪接神器")

    api_key_input()

    page = st.sidebar.radio("選擇功能",
                            ("AI 生成字幕", "雙語字幕翻譯器", "終極版：雙語字幕翻譯器", 
                             "字幕時間同步器", "雙語字幕大小調整器", "字幕錯字修正器", "Whisper API 功能"),
                            captions=["把聲音轉譯成字幕", "翻譯 SRT 文件", "強化版翻譯工具", 
                                      "同步字幕與分鏡點的時間", "調整雙語字幕大小", "改錯字", 
                                      "使用 Whisper API 的功能"])

    if not st.session_state.api_key_valid:
        st.warning("請在側邊欄輸入有效的 OpenAI API Key 以使用此功能")
    else:
        if page == "AI 生成字幕":
            ai_subtitle_generator()
        elif page == "字幕時間同步器":
            subtitle_time_sync()
        elif page == "雙語字幕大小調整器":
            bilingual_subtitle_resizer()
        elif page == "雙語字幕翻譯器":
            bilingual_srt_translator()
        elif page == "終極版：雙語字幕翻譯器":
            multi_language_subtitle_translator()
        elif page == "字幕錯字修正器":
            subtitle_corrector()
        elif page == "Whisper API 功能":
            whisper_api_tool()

    st.sidebar.markdown("---")
    st.sidebar.info("© 2024 剪接神器. All rights reserved.")

if __name__ == "__main__":
    main()