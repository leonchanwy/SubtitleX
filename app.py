import streamlit as st
import ui_utils
from ai_subtitle_generator import ai_subtitle_generator
from subtitle_time_sync import subtitle_time_sync
from bilingual_subtitle_resizer import bilingual_subtitle_resizer
from bilingual_srt_translator import bilingual_srt_translator
from multi_language_subtitle_translator import multi_language_subtitle_translator
from subtitle_corrector import subtitle_corrector
from whisper_api_tool import whisper_api_tool

# --- Initialization ---
def init_session_state():
    # Global API Keys
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    if 'claude_api_key' not in st.session_state:
        st.session_state.claude_api_key = ''
    
    # Navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

# --- Sidebar Logic ---
def render_sidebar():
    with st.sidebar:
        st.title("🎬 SubtitleX")

        # Navigation with radio buttons
        pages = {
            "🏠 Home": "Home",
            "🚀 AI 字幕生成": "AI 生成字幕",
            "🌐 雙語翻譯": "雙語字幕翻譯",
            "🌍 多語言翻譯": "多語言字幕翻譯",
            "⏱️ 時間軸同步": "字幕時間同步",
            "📝 錯字修正": "字幕錯字修正",
            "📏 字幕大小調整": "雙語字幕大小調整",
            "🎙️ Whisper 測試工具": "Whisper API Tool"
        }

        # Get current index
        current_index = 0
        page_values = list(pages.values())
        if st.session_state.current_page in page_values:
            current_index = page_values.index(st.session_state.current_page)

        selected_label = st.radio(
            "功能選單",
            options=list(pages.keys()),
            index=current_index,
            label_visibility="collapsed"
        )

        # Sync selection to current_page
        if pages[selected_label] != st.session_state.current_page:
            st.session_state.current_page = pages[selected_label]
            st.rerun()

        st.markdown("---")

        # API Keys Section
        with st.expander("🔐 API Key 設定", expanded=False):
            st.caption("設定 API Key 以使用翻譯功能")
            
            new_openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                placeholder="sk-..."
            )
            if new_openai_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = new_openai_key
                st.rerun()

            new_claude_key = st.text_input(
                "Claude API Key",
                value=st.session_state.claude_api_key,
                type="password",
                placeholder="sk-ant-..."
            )
            if new_claude_key != st.session_state.claude_api_key:
                st.session_state.claude_api_key = new_claude_key
                st.rerun()

        # Compact status
        status_parts = []
        if st.session_state.openai_api_key:
            status_parts.append("✅ OpenAI")
        if st.session_state.claude_api_key:
            status_parts.append("✅ Claude")

        if status_parts:
            st.caption(" · ".join(status_parts))
        else:
            st.caption("⚠️ 請輸入 API Key")

# --- Home Dashboard ---
def render_home():
    ui_utils.render_header("👋 Welcome to SubtitleX", "Your all-in-one AI toolkit for subtitle creation and localization.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ui_utils.render_card("AI Generator", "Generate subtitles from video/audio using Whisper.", "🚀", "Launch", "btn_ai_gen"):
            st.session_state.current_page = "AI 生成字幕"
            st.rerun()
            
        if ui_utils.render_card("Time Sync", "Fix out-of-sync SRT files automatically.", "⏱️", "Launch", "btn_time_sync"):
            st.session_state.current_page = "字幕時間同步"
            st.rerun()

    with col2:
        if ui_utils.render_card("Bilingual Translator", "Translate SRTs using OpenAI or Claude.", "🌐", "Launch", "btn_bi_trans"):
            st.session_state.current_page = "雙語字幕翻譯"
            st.rerun()
            
        if ui_utils.render_card("Spell Corrector", "Fix common OCR or speech errors.", "📝", "Launch", "btn_correct"):
            st.session_state.current_page = "字幕錯字修正"
            st.rerun()

    with col3:
        if ui_utils.render_card("Multi-lang Translator", "Batch translate to multiple languages.", "🌍", "Launch", "btn_multi_trans"):
            st.session_state.current_page = "多語言字幕翻譯"
            st.rerun()
            
        if ui_utils.render_card("Subtitle Resizer", "Adjust font sizes for bilingual display.", "📏", "Launch", "btn_resize"):
            st.session_state.current_page = "雙語字幕大小調整"
            st.rerun()

# --- Main App Logic ---
def main():
    # Setup page config once
    ui_utils.setup_page("SubtitleX Toolkit")
    
    init_session_state()
    render_sidebar()
    
    # Routing
    page = st.session_state.current_page
    
    if page == "Home":
        render_home()
    elif page == "AI 生成字幕":
        # Pass control to the module, but we might need to update the module 
        # to respect the global session state for keys.
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

if __name__ == "__main__":
    main()
