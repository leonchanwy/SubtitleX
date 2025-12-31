import streamlit as st
import time
import logging
import re
from typing import List, Tuple, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from anthropic import Anthropic, APIConnectionError, APIStatusError
import ui_utils

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 8000
TEMPERATURE = 0.1
BATCH_SIZE = 50
LANGUAGE_OPTIONS = ["繁體中文", "英文", "日文", "馬來語", "廣東話口語", "德文"]

class SubtitleProcessor:
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
            # Fallback for other formats
            pattern = re.compile(
                r'(\d+)\s*\n'
                r'(\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}).*?\n'
                r'([\s\S]*?)'
                r'(?=\n+\d+\s*\n\d{1,2}:\d{2}:\d{2}[,.]\d{3}|\Z)',
                re.MULTILINE
            )
            matches = []
            for match in pattern.finditer(content):
                raw_id, timestamp, text = match.groups()
                matches.append((raw_id.strip(), timestamp.strip(), text.strip()))
        
        if not matches:
             raise ValueError("無法解析SRT文件。請確保文件格式正確。")
        return matches

    @staticmethod
    def clean_text(content: str) -> str:
        content = re.sub(r'</?[a-z]>', '', content)
        content = re.sub(r'\{\\an\d\}', '', content)
        return content

    @staticmethod
    def format_srt(subtitles: List[Tuple[str, str, str]], translations: List[Dict[str, str]], 
                   format_type: str, lang1: str, lang2: str = None) -> str:
        output = []
        for (number, timestamp, original_text), translation in zip(subtitles, translations):
            output.append(f"{number}\n{timestamp}")
            if format_type == "bilingual":
                original = translation.get('original', original_text)
                translated = translation.get(lang1, f'[缺失翻譯: {original}]')
                output.append(f"{original}\n{translated}")
            elif format_type == "dual_lang":
                lang1_text = translation.get(lang1, f'[缺失翻譯: {original_text}]')
                lang2_text = translation.get(lang2, f'[Missing translation: {original_text}]')
                output.append(f"{lang1_text}\n{lang2_text}")
            else:  # 單一語言
                output.append(translation.get(lang1, f'[缺失翻譯: {original_text}]'))
            output.append("")
        return "\n".join(output).strip()

class SubtitleTranslator:
    def __init__(self, api_key: str):
        self.client = Anthropic(
            api_key=api_key,
            default_headers={"anthropic-version": "2023-06-01"}
        )
        self.conversation_history = []

    def _create_system_prompt(self, target_lang1: str, target_lang2: str, prompt1: str, prompt2: str) -> str:
        return f"""你是專業字幕翻譯者。將提供的字幕翻譯成{target_lang1}和{target_lang2}。

按優先順序排列的規則：
1. 格式：對每個輸入字幕，必須嚴格按照以下格式輸出：
   原文：[原文內容]
   {target_lang1}：[翻譯1]
   {target_lang2}：[翻譯2]

2. 保持字幕數量：輸出的翻譯數量必須與輸入的字幕數量完全一致。
3. 單獨翻譯每個字幕，保持順序，不合併或分割。
4. 直接翻譯，不添加解釋或評論。
5. 保留專有名詞（人名、地名等）的原文，不加方括號。
6. 保持原有語氣和口語化風格（如果原文如此）。
7. {target_lang1}風格：{prompt1}
8. {target_lang2}風格：{prompt2}

請嚴格遵循這些規則，特別是前幾條關於格式和數量的規則。"""

    def _manage_conversation_history(self, max_messages=10):
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _translate_batch(self, texts: List[str], target_lang1: str, target_lang2: str, prompt1: str, prompt2: str) -> List[Dict[str, str]]:
        system_prompt = self._create_system_prompt(target_lang1, target_lang2, prompt1, prompt2)

        self._manage_conversation_history()

        combined_texts = "\n\n".join(f"{i+1}. {text}" for i, text in enumerate(texts))
        
        messages = self.conversation_history + [
            {"role": "user", "content": f"翻譯以下字幕：\n\n{combined_texts}"}
        ]

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system_prompt,
                messages=messages
            )

            self.conversation_history.append({"role": "user", "content": f"翻譯以下字幕：\n\n{combined_texts}"})
            self.conversation_history.append({"role": "assistant", "content": response.content[0].text})

            translations = self._parse_translation(response.content[0].text, target_lang1, target_lang2)

            if len(translations) != len(texts):
                logger.warning(f"翻譯數量（{len(translations)}）與原文數量（{len(texts)}）不符")
                while len(translations) < len(texts):
                    translations.append({
                        'original': texts[len(translations)],
                        target_lang1: '[翻譯失敗]',
                        target_lang2: '[Translation failed]'
                    })

            return translations
        except (APIConnectionError, APIStatusError) as e:
            logger.error(f"API 錯誤：{str(e)}")
            raise
        except Exception as e:
            logger.error(f"未預期的錯誤：{str(e)}")
            raise

    def _parse_translation(self, content: str, target_lang1: str, target_lang2: str) -> List[Dict[str, str]]:
        parsed = []
        items = content.split('\n\n')
        for item in items:
            lines = item.strip().split('\n')
            # 寬鬆解析，只要能抓到冒號後面的內容
            try:
                original = ""
                lang1_text = ""
                lang2_text = ""
                
                for line in lines:
                    if line.startswith("原文："):
                        original = line.split("：", 1)[1].strip()
                    elif line.startswith(f"{target_lang1}："):
                        lang1_text = line.split("：", 1)[1].strip()
                    elif line.startswith(f"{target_lang2}："):
                        lang2_text = line.split("：", 1)[1].strip()
                
                if original or lang1_text or lang2_text:
                    parsed.append({
                        'original': original,
                        target_lang1: lang1_text if lang1_text else "[Missing]",
                        target_lang2: lang2_text if lang2_text else "[Missing]"
                    })
            except Exception:
                continue
                
        return parsed

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
                translations = self._translate_batch(texts, target_lang1, target_lang2, prompt1, prompt2)
                translated_subtitles.extend(translations)
            except Exception as e:
                logger.error(f"翻譯批次 {i//BATCH_SIZE + 1} 失敗：{str(e)}")
                placeholder_translations = [
                    {'original': text, target_lang1: '[翻譯失敗]', target_lang2: '[Translation failed]'}
                    for text in texts
                ]
                translated_subtitles.extend(placeholder_translations)

            progress = min((i + BATCH_SIZE) / total, 1.0)
            progress_callback(progress)

            time.sleep(1)

        return translated_subtitles

    def reset_conversation(self):
        self.conversation_history = []

def multi_language_subtitle_translator():
    ui_utils.render_header("🌍 Multi-Language Translator (Legacy)", "Translate to multiple languages using Claude (Batch Processing).")

    if not ui_utils.validate_api_inputs(["Claude"]):
        st.stop()
    
    api_key = ui_utils.get_api_key("Claude")

    col1, col2 = st.columns(2)
    with col1:
        target_lang1 = st.selectbox("Language 1", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("廣東話口語"))
        prompt1 = st.text_input("Style 1", value="口語化帶俚語")
    with col2:
        target_lang2 = st.selectbox("Language 2", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("英文"))
        prompt2 = st.text_input("Style 2", value="都市俚語")

    uploaded_file = st.file_uploader("Upload SRT", type="srt")
    
    with st.expander("Options"):
        use_continuous_conversation = st.checkbox("Continuous Context", value=True)
    
    if st.button("Reset History"):
        if 'translator' in st.session_state:
            st.session_state.translator.reset_conversation()
        st.success("History Reset")

    if uploaded_file and st.button("Start Translation", type="primary"):
        try:
            content = uploaded_file.getvalue().decode("utf-8-sig")
            content = SubtitleProcessor.clean_text(content)
            subtitles = SubtitleProcessor.parse_srt(content)
            
            if 'translator' not in st.session_state:
                st.session_state.translator = SubtitleTranslator(api_key)
            
            if not use_continuous_conversation:
                st.session_state.translator.reset_conversation()

            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Translating..."):
                start_time = time.time()
                st.session_state.translated_subtitles = st.session_state.translator.translate_subtitles(
                    subtitles, target_lang1, target_lang2, prompt1, prompt2, progress_bar.progress
                )
                st.session_state.original_subtitles = subtitles
                
            status_text.success(f"✅ Finished in {time.time() - start_time:.2f}s")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.exception("Translation error")

    if st.session_state.get('translated_subtitles'):
        st.divider()
        st.subheader("Download")
        
        dl_options = [f"Original + {target_lang1}", f"Original + {target_lang2}", f"{target_lang1} + {target_lang2}", f"Only {target_lang1}", f"Only {target_lang2}"]
        download_option = st.selectbox("Format", dl_options)

        preview_subtitles = st.session_state.original_subtitles[:5]
        preview_translations = st.session_state.translated_subtitles[:5]
        
        try:
            format_type = "single"
            l1 = target_lang1
            l2 = None
            
            if "Original" in download_option:
                format_type = "bilingual"
                l1 = target_lang1 if target_lang1 in download_option else target_lang2
            elif "+" in download_option:
                format_type = "dual_lang"
                l1 = target_lang1
                l2 = target_lang2
            else:
                l1 = target_lang1 if target_lang1 in download_option else target_lang2

            preview_srt = SubtitleProcessor.format_srt(preview_subtitles, preview_translations, format_type, l1, l2)
            full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, format_type, l1, l2)
            
            file_name = f"subtitle_{l1}_{l2 if l2 else ''}.srt".replace("__", "_")

            st.text_area("Preview", value=preview_srt + "\n...", height=200)

            st.download_button(
                label="📥 Download",
                data=full_srt,
                file_name=file_name,
                mime="text/plain"
            )

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    multi_language_subtitle_translator()
