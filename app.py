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
        st.image("https://img.icons8.com/clouds/100/000000/video-editing.png", width=80)
        st.title("SubtitleX Tool")
        
        # Navigation
        st.subheader("📍 Navigation")
        
        pages = {
            "🏠 Home": "Home",
            "🚀 AI Subtitle Generator": "AI 生成字幕",
            "🌐 Bilingual Translator": "雙語字幕翻譯",
            "🌍 Multi-language Translator": "多語言字幕翻譯",
            "⏱️ Time Sync": "字幕時間同步",
            "📝 Spell Corrector": "字幕錯字修正",
            "📏 Subtitle Resizer": "雙語字幕大小調整",
            "🎙️ Whisper Tool": "Whisper API Tool"
        }
        
        # Update session state when selectbox changes
        selected_label = st.selectbox(
            "Go to:",
            options=list(pages.keys()),
            index=list(pages.values()).index(st.session_state.current_page) if st.session_state.current_page in pages.values() else 0,
            key="nav_selector"
        )
        
        # Sync selection to current_page
        if st.session_state.nav_selector:
            st.session_state.current_page = pages[selected_label]

        st.markdown("---")
        
        # Global Settings (API Keys)
        with st.expander("⚙️ Global Settings", expanded=False):
            st.markdown("### API Keys")
            
            # OpenAI Key
            new_openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                help="Required for AI Subtitles and OpenAI translation."
            )
            if new_openai_key != st.session_state.openai_api_key:
                st.session_state.openai_api_key = new_openai_key
                st.rerun()

            # Claude Key
            new_claude_key = st.text_input(
                "Anthropic (Claude) API Key",
                value=st.session_state.claude_api_key,
                type="password",
                help="Required for Claude translation."
            )
            if new_claude_key != st.session_state.claude_api_key:
                st.session_state.claude_api_key = new_claude_key
                st.rerun()

        # Status Indicators
        st.markdown("### System Status")
        ui_utils.render_api_key_status("OpenAI", bool(st.session_state.openai_api_key))
        ui_utils.render_api_key_status("Claude", bool(st.session_state.claude_api_key))
        
        st.markdown("---")
        st.info("© 2025 SubtitleX. All rights reserved.")

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
