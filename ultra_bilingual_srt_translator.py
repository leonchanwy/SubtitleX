import anthropic
import streamlit as st
import re
import time
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_anthropic_client(api_key):
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.error(f"創建 Anthropic 客戶端時出錯: {str(e)}")
        raise

def parse_srt(content):
    # Normalize line endings
    content = content.replace('\r\n', '\n').replace('\r', '\n')

    # Use a more lenient regular expression
    pattern = re.compile(
        r'(\d+)\s*\n'                               # Match the subtitle number
        r'(\d{2}:\d{2}:\d{2},\d{3}\s*-->.*?\s*\d{2}:\d{2}:\d{2},\d{3})\s*\n'  # Match the timestamp, allow spaces
        r'(.*?)\s*(?=\n\d+\n|\Z)',                  # Match the subtitle text, non-greedy, allow spaces
        re.S                                         # Dot matches newline, enabling multi-line matching
    )
    matches = pattern.findall(content)
    
    if not matches:
        logger.warning("無法解析 SRT 文件")
        raise ValueError("無法解析 SRT 文件。請確保文件格式正確。")
    
    return matches

def translate_subtitle(client, text, target_lang1, target_lang2, prompt1, prompt2):
    try:
        system_prompt = f"""You are a highly skilled translator with expertise in many languages.
        Your task is to identify the language of the text I provide and accurately translate it into both {target_lang1} and {target_lang2}.
        Rules:
        1. Provide direct translations without explanations or notes.
        2. Keep proper nouns and place names as they are.
        3. Preserving the meaning, tone, and nuance of the original text.
        4. Maintain proper grammar, spelling, and punctuation in the translated version.
        5. Translate each instance of repeated words.
        6. For {target_lang1}: {prompt1}
        7. For {target_lang2}: {prompt2}
        8. Maintain the style and tone of the original text, including any informal or colloquial expressions.
        9. If you encounter any words or phrases you're unsure about, translate them to the best of your ability rather than leaving them untranslated or marking them as errors.
        10. Return the translations in the format: "{target_lang1}: [translation]\n{target_lang2}: [translation]".
        11. Do not include the source language or any other text in your response."""

        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": text.strip()
                }
            ]
        )
        translation = message.content[0].text.strip()
        
        if not (f"{target_lang1}:" in translation and f"{target_lang2}:" in translation):
            raise ValueError("翻譯格式不正確")
        
        logger.info(f"成功翻譯: {text[:30]}...")
        return translation
    except Exception as e:
        logger.error(f"翻譯出錯: {str(e)}")
        return f"{target_lang1}: [翻譯錯誤]\n{target_lang2}: [翻譯錯誤]"

def translate_subtitles(client, subtitles, target_lang1, target_lang2, prompt1, prompt2, progress_bar):
    translated_subtitles = []
    total = len(subtitles)
    
    for i, (number, timestamp, text) in enumerate(subtitles, 1):
        translation = translate_subtitle(client, text, target_lang1, target_lang2, prompt1, prompt2)
        translated_subtitles.append((number, timestamp, text.strip(), translation))

        progress = (i / total)
        progress_bar.progress(progress)

    return translated_subtitles

def format_bilingual_srt(translated_subtitles, original, target_lang):
    output = ""
    for number, timestamp, original_text, translation in translated_subtitles:
        lang_translation = re.search(f"{target_lang}: (.+)", translation)
        if lang_translation:
            if original:
                output += f"{number}\n{timestamp}\n{original_text}\n{lang_translation.group(1)}\n\n"
            else:
                output += f"{number}\n{timestamp}\n{lang_translation.group(1)}\n\n"
        else:
            logger.warning(f"無法為字幕 {number} 提取 {target_lang} 翻譯")
            output += f"{number}\n{timestamp}\n{'翻譯錯誤' if not original else original_text + '\n翻譯錯誤'}\n\n"
    return output.strip()

def format_dual_target_srt(translated_subtitles, target_lang1, target_lang2):
    output = ""
    for number, timestamp, _, translation in translated_subtitles:
        lang1_translation = re.search(f"{target_lang1}: (.+)", translation)
        lang2_translation = re.search(f"{target_lang2}: (.+)", translation)
        if lang1_translation and lang2_translation:
            output += f"{number}\n{timestamp}\n{lang1_translation.group(1)}\n{lang2_translation.group(1)}\n\n"
        else:
            logger.warning(f"無法為字幕 {number} 提取 {target_lang1} 或 {target_lang2} 翻譯")
            output += f"{number}\n{timestamp}\n翻譯錯誤\n翻譯錯誤\n\n"
    return output.strip()

def ultra_bilingual_srt_translator():
    st.title("多語言字幕翻譯器")

    api_key = st.text_input("Anthropic API 金鑰", type="password")
    target_lang1 = st.text_input("目標語言 1", value="traditional chinese")
    target_lang2 = st.text_input("目標語言 2", value="malay")
    prompt1 = st.text_area("語言 1 翻譯風格提示", value="使用日常口語化的表達，保持原文的情感和語氣，包括俚語和粗話的等效表達")
    prompt2 = st.text_area("語言 2 翻譯風格提示", value="使用日常口語化的表達，保持原文的情感和語氣，包括俚語和粗話的等效表達")

    uploaded_file = st.file_uploader("選擇一個 SRT 文件", type="srt")

    if 'translated_subtitles' not in st.session_state:
        st.session_state.translated_subtitles = None

    if uploaded_file is not None and api_key and st.button("翻譯"):
        try:
            content = uploaded_file.getvalue().decode("utf-8-sig")
            subtitles = parse_srt(content)

            client = get_anthropic_client(api_key)

            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("正在翻譯..."):
                start_time = time.time()
                st.session_state.translated_subtitles = translate_subtitles(client, subtitles, target_lang1, target_lang2, prompt1, prompt2, progress_bar)
                end_time = time.time()

            processing_time = end_time - start_time
            status_text.info(f"翻譯完成！總處理時間：{processing_time:.2f} 秒")
            logger.info(f"翻譯完成，處理時間：{processing_time:.2f} 秒")

        except Exception as e:
            st.error(f"處理過程中發生錯誤: {str(e)}")
            logger.exception("翻譯過程中發生異常")

    if st.session_state.translated_subtitles is not None:
        download_option = st.selectbox(
            "選擇下載格式",
            [f"原文 + {target_lang1}", f"原文 + {target_lang2}", f"{target_lang1} + {target_lang2}", f"僅 {target_lang1}", f"僅 {target_lang2}"]
        )

        st.subheader("翻譯預覽")
        preview_srt = format_bilingual_srt(st.session_state.translated_subtitles[:5], True, target_lang1) if download_option.startswith("原文") else \
                      format_dual_target_srt(st.session_state.translated_subtitles[:5], target_lang1, target_lang2) if download_option == f"{target_lang1} + {target_lang2}" else \
                      format_bilingual_srt(st.session_state.translated_subtitles[:5], False, target_lang1 if download_option.endswith(target_lang1) else target_lang2)
        
        st.text_area("", value=preview_srt + "\n...", height=300)

        if download_option == f"原文 + {target_lang1}":
            download_srt = format_bilingual_srt(st.session_state.translated_subtitles, True, target_lang1)
            file_name = f"原文_{target_lang1}_subtitles.srt"
        elif download_option == f"原文 + {target_lang2}":
            download_srt = format_bilingual_srt(st.session_state.translated_subtitles, True, target_lang2)
            file_name = f"原文_{target_lang2}_subtitles.srt"
        elif download_option == f"{target_lang1} + {target_lang2}":
            download_srt = format_dual_target_srt(st.session_state.translated_subtitles, target_lang1, target_lang2)
            file_name = f"{target_lang1}_{target_lang2}_subtitles.srt"
        elif download_option == f"僅 {target_lang1}":
            download_srt = format_bilingual_srt(st.session_state.translated_subtitles, False, target_lang1)
            file_name = f"{target_lang1}_subtitles.srt"
        else:
            download_srt = format_bilingual_srt(st.session_state.translated_subtitles, False, target_lang2)
            file_name = f"{target_lang2}_subtitles.srt"

        st.download_button(
            label=f"下載 {download_option} 字幕",
            data=download_srt,
            file_name=file_name,
            mime="text/plain"
        )

if __name__ == "__main__":
    ultra_bilingual_srt_translator()