import streamlit as st
import anthropic
import re
import time
import logging
from typing import List, Tuple, Dict
from tenacity import retry, stop_after_attempt, wait_random_exponential

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
LANGUAGE_OPTIONS = ["繁體中文", "英文", "日文", "馬來語", "廣東話口語", "德文"]
MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 4000
TEMPERATURE = 0.2
BATCH_SIZE = 20

class SRTTranslator:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def translate_batch(self, texts: List[str], target_lang1: str, target_lang2: str, prompt1: str, prompt2: str) -> str:
        combined_texts = "\n\n".join(texts)
        system_prompt = self._create_system_prompt(target_lang1, target_lang2, prompt1, prompt2)
        
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": f"Translate:\n\n{combined_texts}"}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"API調用失敗: {str(e)}")
            raise

    def _create_system_prompt(self, target_lang1: str, target_lang2: str, prompt1: str, prompt2: str) -> str:
        return f"""You are a highly skilled translator. Translate the following subtitles into both {target_lang1} and {target_lang2}.
        Rules:
        1. Provide direct translations without explanations.
        2. Preserve proper nouns and place names.
        3. Maintain the original tone and nuance.
        4. Use correct grammar and punctuation.
        5. Translate repeated words consistently.
        6. Keep the style informal and colloquial if the original is.
        7. Translate uncertain phrases to the best of your ability.
        8. Format: "Original: [text]\n{target_lang1}: [translation]\n{target_lang2}: [translation]"
        9. Translate each subtitle separately, maintaining order.
        10. Be consistent throughout the dialogue.
        11. For {target_lang1}: {prompt1}
        12. For {target_lang2}: {prompt2}"""

    def translate_subtitles(self, subtitles: List[Tuple[str, str, str]], 
                            target_lang1: str, target_lang2: str, 
                            prompt1: str, prompt2: str, 
                            progress_callback) -> List[Dict[str, str]]:
        translated_subtitles = []
        total = len(subtitles)

        for i in range(0, total, BATCH_SIZE):
            batch = subtitles[i:i+BATCH_SIZE]
            texts = [text for _, _, text in batch]
            
            try:
                translation = self.translate_batch(texts, target_lang1, target_lang2, prompt1, prompt2)
                parsed_translations = self._parse_translation(translation, target_lang1, target_lang2)
                
                if len(parsed_translations) != len(batch):
                    logger.warning(f"翻譯結果數量與原文不匹配。批次 {i//BATCH_SIZE + 1}")
                    parsed_translations = self._align_translations(parsed_translations, batch)
                
                translated_subtitles.extend(parsed_translations)

                progress = min((i + BATCH_SIZE) / total, 1.0)
                progress_callback(progress)

            except Exception as e:
                logger.error(f"批次{i//BATCH_SIZE + 1}翻譯失敗: {str(e)}")
                translated_subtitles.extend([{'original': '[翻譯錯誤]', target_lang1: '[翻譯錯誤]', target_lang2: '[翻譯錯誤]'} for _ in batch])

            time.sleep(1)  # 避免超過API限制

        return translated_subtitles

    def _parse_translation(self, translation: str, target_lang1: str, target_lang2: str) -> List[Dict[str, str]]:
        parsed = []
        current_item = {'original': '', target_lang1: '', target_lang2: ''}
        for line in translation.strip().split('\n'):
            if line.startswith("Original:"):
                if current_item['original'] or current_item[target_lang1] or current_item[target_lang2]:
                    parsed.append(current_item)
                    current_item = {'original': '', target_lang1: '', target_lang2: ''}
                current_item['original'] = line.replace("Original:", "").strip()
            elif line.startswith(f"{target_lang1}:"):
                current_item[target_lang1] = line.replace(f"{target_lang1}:", "").strip()
            elif line.startswith(f"{target_lang2}:"):
                current_item[target_lang2] = line.replace(f"{target_lang2}:", "").strip()
        if current_item['original'] or current_item[target_lang1] or current_item[target_lang2]:
            parsed.append(current_item)
        return parsed

    def _align_translations(self, translations: List[Dict[str, str]], originals: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        aligned = []
        for original, translation in zip(originals, translations + [None] * (len(originals) - len(translations))):
            if translation:
                aligned.append(translation)
            else:
                aligned.append({'original': original[2], 'target_lang1': '[翻譯錯誤]', 'target_lang2': '[翻譯錯誤]'})
        return aligned

class SRTProcessor:
    @staticmethod
    def parse_srt(content: str) -> List[Tuple[str, str, str]]:
        content = content.replace('\r\n', '\n').strip()
        pattern = re.compile(
            r'(\d+)\n'
            r'(\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3})\n'
            r'((?:(?!\n\d+\n).)+)',
            re.DOTALL
        )
        matches = pattern.findall(content)
        if not matches:
            raise ValueError("無法解析SRT文件。請確保文件格式正確。")
        return matches

    @staticmethod
    def remove_language_tags(content: str) -> str:
        content = re.sub(r'</?[a-z]>', '', content)  # 移除 <i>, </i> 等標籤
        content = re.sub(r'\{\\an\d\}', '', content)  # 移除 {\\an8} 等標籤
        return content

    @staticmethod
    def format_srt(subtitles: List[Tuple[str, str, str]], translations: List[Dict[str, str]], 
                   format_type: str, lang1: str, lang2: str = None) -> str:
        output = []
        for (number, timestamp, original_text), translation in zip(subtitles, translations):
            output.append(f"{number}\n{timestamp}")
            if format_type == "bilingual":
                original = translation.get('original', original_text)
                translated = translation.get(lang1, '[翻譯錯誤]')
                output.append(f"{original}\n{translated}")
            elif format_type == "dual_lang":
                lang1_text = translation.get(lang1, '[翻譯錯誤]')
                lang2_text = translation.get(lang2, '[翻譯錯誤]')
                output.append(f"{lang1_text}\n{lang2_text}")
            else:  # single language
                output.append(translation.get(lang1, '[翻譯錯誤]'))
            output.append("")  # 添加空行
        return "\n".join(output).strip()

def ultra_bilingual_srt_translator():
    st.title("多語言字幕翻譯器")

    api_key = st.text_input("Anthropic API金鑰", type="password")
    target_lang1 = st.selectbox("目標語言1", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("廣東話口語"))
    target_lang2 = st.selectbox("目標語言2", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("英文"))

    prompt1 = st.text_input("語言1翻譯風格提示", value="充滿口語髒話")
    prompt2 = st.text_input("語言2翻譯風格提示", value="充滿黑人rapper口語髒話")

    uploaded_file = st.file_uploader("選擇一個SRT文件", type="srt")

    if 'translated_subtitles' not in st.session_state:
        st.session_state.translated_subtitles = None
    if 'original_subtitles' not in st.session_state:
        st.session_state.original_subtitles = None

    if uploaded_file and api_key and st.button("開始翻譯"):
        try:
            content = uploaded_file.getvalue().decode("utf-8-sig")
            content = SRTProcessor.remove_language_tags(content)
            subtitles = SRTProcessor.parse_srt(content)
            
            translator = SRTTranslator(api_key)

            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("正在翻譯..."):
                start_time = time.time()
                st.session_state.translated_subtitles = translator.translate_subtitles(
                    subtitles, target_lang1, target_lang2, prompt1, prompt2, progress_bar.progress
                )
                st.session_state.original_subtitles = subtitles
                end_time = time.time()

            processing_time = end_time - start_time
            status_text.success(f"翻譯完成！總處理時間：{processing_time:.2f}秒")

        except Exception as e:
            st.error(f"處理過程中發生錯誤: {str(e)}")
            logger.exception("翻譯過程中發生異常")

    if st.session_state.translated_subtitles:
        download_option = st.selectbox(
            "選擇下載格式",
            [f"原文 + {target_lang1}", f"原文 + {target_lang2}", f"{target_lang1} + {target_lang2}", f"僅{target_lang1}", f"僅{target_lang2}"]
        )

        st.subheader("翻譯預覽")
        preview_subtitles = st.session_state.original_subtitles[:5]
        preview_translations = st.session_state.translated_subtitles[:5]
        
        try:
            if "原文" in download_option:
                lang = target_lang1 if target_lang1 in download_option else target_lang2
                preview_srt = SRTProcessor.format_srt(preview_subtitles, preview_translations, "bilingual", lang)
                full_srt = SRTProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "bilingual", lang)
                file_name = f"原文_{lang}.srt"
            elif "+" in download_option:
                preview_srt = SRTProcessor.format_srt(preview_subtitles, preview_translations, "dual_lang", target_lang1, target_lang2)
                full_srt = SRTProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "dual_lang", target_lang1, target_lang2)
                file_name = f"{target_lang1}_{target_lang2}.srt"
            else:
                lang = target_lang1 if target_lang1 in download_option else target_lang2
                preview_srt = SRTProcessor.format_srt(preview_subtitles, preview_translations, "single", lang)
                full_srt = SRTProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "single", lang)
                file_name = f"{lang}.srt"
            
            st.text_area("翻譯預覽", value=preview_srt + "\n...", height=300)

            st.download_button(
                label=f"下載{download_option}字幕",
                data=full_srt,
                file_name=file_name,
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"生成預覽或下載文件時發生錯誤: {str(e)}")
            logger.exception("生成預覽或下載文件時發生異常")

if __name__ == "__main__":
    ultra_bilingual_srt_translator()