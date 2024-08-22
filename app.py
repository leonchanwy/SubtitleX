import streamlit as st
from ai_subtitle_generator import ai_subtitle_generator
from subtitle_time_sync import subtitle_time_sync
from bilingual_subtitle_resizer import bilingual_subtitle_resizer
from bilingual_srt_translator import bilingual_srt_translator
from ultra_bilingual_srt_translator import ultra_bilingual_srt_translator


def main():
    st.sidebar.title("剪接神器")

    page = st.sidebar.radio("選擇功能",
                            ("AI 生成字幕", "雙語字幕翻譯器", "字幕時間同步器", "雙語字幕大小調整器", "終極版：雙語字幕翻譯器"),
                            captions=["把聲音轉譯成字幕","翻譯 SRT 文件","同步字幕與分鏡點的時間","調整雙語字幕大小", "測試用"])

    if page == "AI 生成字幕":
        ai_subtitle_generator()
    elif page == "字幕時間同步器":
        subtitle_time_sync()
    elif page == "雙語字幕大小調整器":
        bilingual_subtitle_resizer()
    elif page == "雙語字幕翻譯器":
        bilingual_srt_translator()
    elif page == "終極版：雙語字幕翻譯器":
        ultra_bilingual_srt_translator()

if __name__ == "__main__":
    main()