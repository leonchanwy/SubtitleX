import anthropic
import streamlit as st
import re
import time
import logging
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_anthropic_client(api_key: str):
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.error(f"創建 Anthropic 客戶端時出錯: {str(e)}")
        raise

def parse_srt(content: str) -> List[Tuple[str, str, str]]:
    """Parse the SRT content into a list of tuples containing (number, timestamp, text)."""
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\n((?:.+?\n)+?)(?:\n|$)', re.DOTALL)
    matches = pattern.findall(content)
    return [(match[0], match[1], match[2].strip()) for match in matches]

def translate_subtitle(client, text: str, target_lang1: str, target_lang2: str, prompt1: str, prompt2: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            system_prompt = f"""You are a professional translator. Your task is to translate the given text into both {target_lang1} and {target_lang2}.
Rules:
1. Provide direct translations without explanations or notes.
2. Keep proper nouns and place names as they are.
3. Translate each instance of repeated words.
4. For {target_lang1}: {prompt1}
5. For {target_lang2}: {prompt2}
6. Maintain the style and tone of the original text, including any informal or colloquial expressions.
7. If you encounter any words or phrases you're unsure about, translate them to the best of your ability rather than leaving them untranslated or marking them as errors.
8. Return the translations in the format: "{target_lang1}: [translation]\\n{target_lang2}: [translation]" for each input line.
9. Do not include the source language or any other text in your response.
10. Treat each line as a separate subtitle and provide translations for each line."""

            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
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
            
            # 確保每行都有兩種語言的翻譯
            lines = translation.split('\n')
            formatted_lines = []
            for i in range(0, len(lines), 2):
                lang1 = lines[i] if i < len(lines) and lines[i].startswith(f"{target_lang1}:") else f"{target_lang1}: [翻譯錯誤]"
                lang2 = lines[i+1] if i+1 < len(lines) and lines[i+1].startswith(f"{target_lang2}:") else f"{target_lang2}: [翻譯錯誤]"
                formatted_lines.extend([lang1, lang2])
            
            logger.info(f"成功翻譯: {text[:30]}...")
            return "\n".join(formatted_lines)
        except Exception as e:
            logger.error(f"翻譯嘗試 {attempt + 1} 失敗: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(f"翻譯最終失敗: {text}")
                return f"{target_lang1}: [翻譯錯誤]\n{target_lang2}: [翻譯錯誤]"
            time.sleep(2 ** attempt)  # 指數退避

def batch_translate_subtitles(client, subtitles: List[Tuple[str, str, str]], target_lang1: str, target_lang2: str, prompt1: str, prompt2: str, batch_size: int = 10) -> List[Tuple[str, str, str, str]]:
    translated_subtitles = []
    total = len(subtitles)
    cache: Dict[str, str] = {}

    for i in range(0, total, batch_size):
        batch = subtitles[i:i+batch_size]
        batch_text = "\n".join([text for _, _, text in batch])
        
        if batch_text in cache:
            translations = cache[batch_text]
        else:
            translations = translate_subtitle(client, batch_text, target_lang1, target_lang2, prompt1, prompt2)
            cache[batch_text] = translations

        translation_lines = translations.split('\n')
        for j, (number, timestamp, text) in enumerate(batch):
            if j*2+1 < len(translation_lines):
                translated_subtitles.append((number, timestamp, text.strip(), f"{translation_lines[j*2]}\n{translation_lines[j*2+1]}"))
            else:
                logger.warning(f"翻譯行數不匹配，對字幕 {number} 進行單獨翻譯")
                single_translation = translate_subtitle(client, text, target_lang1, target_lang2, prompt1, prompt2)
                translated_subtitles.append((number, timestamp, text.strip(), single_translation))

        progress = (i + len(batch)) / total
        st.session_state.progress_bar.progress(progress)

    return translated_subtitles

def format_bilingual_srt(translated_subtitles: List[Tuple[str, str, str, str]], original: bool, target_lang: str) -> str:
    """Format the translated subtitles into a bilingual SRT format."""
    srt_content = []
    for number, timestamp, original_text, translated_text in translated_subtitles:
        translations = translated_text.split('\n')
        target_translation = next((t for t in translations if t.startswith(f"{target_lang}:")), f"{target_lang}: [翻譯錯誤]")
        target_translation = target_translation.split(f"{target_lang}:", 1)[1].strip()
        
        if original:
            srt_content.append(f"{number}\n{timestamp}\n{original_text}\n{target_translation}\n")
        else:
            srt_content.append(f"{number}\n{timestamp}\n{target_translation}\n")
    return "\n".join(srt_content)

def format_dual_target_srt(translated_subtitles: List[Tuple[str, str, str, str]], target_lang1: str, target_lang2: str) -> str:
    """Format the subtitles with translations in two target languages."""
    srt_content = []
    default_translation = "[翻譯錯誤]"
    
    for number, timestamp, _, translated_text in translated_subtitles:
        translations = translated_text.split('\n')
        
        translation_1 = next((t.split(f"{target_lang1}:", 1)[1].strip() for t in translations if t.startswith(f"{target_lang1}:")), default_translation)
        translation_2 = next((t.split(f"{target_lang2}:", 1)[1].strip() for t in translations if t.startswith(f"{target_lang2}:")), default_translation)
        
        srt_content.append(f"{number}\n{timestamp}\n{translation_1}\n{translation_2}\n")
    
    return "\n".join(srt_content)

def main():
    st.title("雙語字幕翻譯器")

    if 'translated_subtitles' not in st.session_state:
        st.session_state.translated_subtitles = None

    api_key = st.text_input("輸入您的 Anthropic API 金鑰:", type="password")
    target_lang1 = st.text_input("輸入第一個目標語言:", value="繁體中文")
    target_lang2 = st.text_input("輸入第二個目標語言:", value="馬來語")
    prompt1 = st.text_area("輸入第一種語言的翻譯提示:", value="使用日常口語化的表達，保持原文的情感和語氣，包括俚語和粗話的等效表達")
    prompt2 = st.text_area("輸入第二種語言的翻譯提示:", value="使用日常口語化的表達，保持原文的情感和語氣，包括俚語和粗話的等效表達")

    uploaded_file = st.file_uploader("上傳 SRT 文件", type=["srt"])
    
    if st.button("翻譯") and uploaded_file and api_key and target_lang1 and target_lang2:
        try:
            content = uploaded_file.getvalue().decode("utf-8-sig")
            subtitles = parse_srt(content)

            client = get_anthropic_client(api_key)

            st.session_state.progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("正在翻譯..."):
                start_time = time.time()
                st.session_state.translated_subtitles = batch_translate_subtitles(client, subtitles, target_lang1, target_lang2, prompt1, prompt2)
                end_time = time.time()

            processing_time = end_time - start_time
            status_text.info(f"翻譯完成！總處理時間：{processing_time:.2f} 秒")
            logger.info(f"翻譯完成，處理時間：{processing_time:.2f} 秒")

        except Exception as e:
            st.error(f"翻譯過程中發生錯誤: {str(e)}")
            logger.exception("翻譯過程失敗")

    if st.session_state.translated_subtitles is not None:
        st.subheader("下載選項")
        download_option = st.selectbox(
            "選擇下載格式",
            [f"原文 + {target_lang1}", f"原文 + {target_lang2}", f"{target_lang1} + {target_lang2}", f"僅 {target_lang1}", f"僅 {target_lang2}"]
        )

        if download_option == f"原文 + {target_lang1}":
            formatted_srt = format_bilingual_srt(st.session_state.translated_subtitles, True, target_lang1)
            file_name = f"原文_{target_lang1}_subtitles.srt"
        elif download_option == f"原文 + {target_lang2}":
            formatted_srt = format_bilingual_srt(st.session_state.translated_subtitles, True, target_lang2)
            file_name = f"原文_{target_lang2}_subtitles.srt"
        elif download_option == f"{target_lang1} + {target_lang2}":
            formatted_srt = format_dual_target_srt(st.session_state.translated_subtitles, target_lang1, target_lang2)
            file_name = f"{target_lang1}_{target_lang2}_subtitles.srt"
        elif download_option == f"僅 {target_lang1}":
            formatted_srt = format_bilingual_srt(st.session_state.translated_subtitles, False, target_lang1)
            file_name = f"{target_lang1}_subtitles.srt"
        else:
            formatted_srt = format_bilingual_srt(st.session_state.translated_subtitles, False, target_lang2)
            file_name = f"{target_lang2}_subtitles.srt"

        st.download_button(
            label=f"下載 {download_option} 字幕",
            data=formatted_srt,
            file_name=file_name,
            mime="text/plain"
        )

        st.subheader("翻譯預覽")
        preview_lines = formatted_srt.split('\n')[:20]  # 只顯示前20行
        st.text_area("預覽", value="\n".join(preview_lines) + "\n...", height=300)

if __name__ == "__main__":
    main()