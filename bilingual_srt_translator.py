import streamlit as st
import re
import time
import logging
from typing import List, Tuple, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
MODEL = "gpt-4o-2024-08-06"
MAX_TOKENS = 4000
TEMPERATURE = 0.1
BATCH_SIZE = 30
LANGUAGE_OPTIONS = ["繁體中文", "英文", "日文", "馬來語", "廣東話口語", "德文"]

def init_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'api_key_valid' not in st.session_state:
        st.session_state.api_key_valid = False

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
        self.client = OpenAI(api_key=api_key)
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
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"翻譯以下字幕：\n\n{combined_texts}"}
        ]

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            self.conversation_history.append({"role": "user", "content": f"翻譯以下字幕：\n\n{combined_texts}"})
            self.conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})

            translations = self._parse_translation(response.choices[0].message.content, target_lang1, target_lang2)

            if len(translations) != len(texts):
                logger.warning(f"翻譯數量（{len(translations)}）與原文數量（{len(texts)}）不符")
                while len(translations) < len(texts):
                    translations.append({
                        'original': texts[len(translations)],
                        target_lang1: '[翻譯失敗]',
                        target_lang2: '[Translation failed]'
                    })

            return translations
        except Exception as e:
            logger.error(f"API 錯誤：{str(e)}")
            raise

    def _parse_translation(self, content: str, target_lang1: str, target_lang2: str) -> List[Dict[str, str]]:
        parsed = []
        items = content.split('\n\n')
        for item in items:
            lines = item.strip().split('\n')
            if len(lines) == 3:
                original = lines[0].split('：', 1)[1].strip()
                lang1 = lines[1].split('：', 1)[1].strip()
                lang2 = lines[2].split('：', 1)[1].strip()
                parsed.append({
                    'original': original,
                    target_lang1: lang1,
                    target_lang2: lang2
                })
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

            time.sleep(1)  # 避免 API 速率限制

        return translated_subtitles

    def reset_conversation(self):
        self.conversation_history = []

def load_api_key() -> Optional[str]:
    try:
        with open('api_key.txt', 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def save_api_key(api_key: str):
    with open('api_key.txt', 'w') as file:
        file.write(api_key)

def bilingual_srt_translator():
    init_session_state()
    st.title("🌐 雙語字幕翻譯器（GPT-4o）")

    api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.session_state.api_key_valid = validate_api_key(api_key)

    col1, col2 = st.columns(2)
    with col1:
        target_lang1 = st.selectbox("目標語言 1", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("廣東話口語"))
        prompt1 = st.text_input("語言 1 翻譯風格", value="口語化帶俚語")
    with col2:
        target_lang2 = st.selectbox("目標語言 2", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("英文"))
        prompt2 = st.text_input("語言 2 翻譯風格", value="都市俚語")

    uploaded_file = st.file_uploader("選擇 SRT 文件", type="srt")

    use_continuous_conversation = st.checkbox("使用持續對話（可能提高翻譯一致性）", value=True)
    
    if st.button("重置翻譯對話歷史"):
        if 'translator' in st.session_state:
            st.session_state.translator.reset_conversation()
        st.success("翻譯對話歷史已重置")

    if 'translated_subtitles' not in st.session_state:
        st.session_state.translated_subtitles = None
    if 'original_subtitles' not in st.session_state:
        st.session_state.original_subtitles = None

    if uploaded_file and api_key and st.button("開始翻譯", key="translate_button"):
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

            with st.spinner("正在翻譯..."):
                start_time = time.time()
                st.session_state.translated_subtitles = st.session_state.translator.translate_subtitles(
                    subtitles, target_lang1, target_lang2, prompt1, prompt2, progress_bar.progress
                )
                st.session_state.original_subtitles = subtitles
                end_time = time.time()

            processing_time = end_time - start_time
            status_text.success(f"✅ 翻譯完成！總處理時間：{processing_time:.2f} 秒")

        except Exception as e:
            st.error(f"❌ 處理過程中發生錯誤：{str(e)}")
            logger.exception("翻譯過程中發生異常")

    if st.session_state.translated_subtitles:
        st.subheader("翻譯結果")
        download_option = st.selectbox(
            "選擇下載格式",
            [f"原文 + {target_lang1}", f"原文 + {target_lang2}", f"{target_lang1} + {target_lang2}", f"僅 {target_lang1}", f"僅 {target_lang2}"]
        )

        preview_subtitles = st.session_state.original_subtitles[:5]
        preview_translations = st.session_state.translated_subtitles[:5]
        
        try:
            if "原文" in download_option:
                lang = target_lang1 if target_lang1 in download_option else target_lang2
                preview_srt = SubtitleProcessor.format_srt(preview_subtitles, preview_translations, "bilingual", lang)
                full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "bilingual", lang)
                file_name = f"原文_{lang}.srt"
            elif "+" in download_option:
                preview_srt = SubtitleProcessor.format_srt(preview_subtitles, preview_translations, "dual_lang", target_lang1, target_lang2)
                full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "dual_lang", target_lang1, target_lang2)
                file_name = f"{target_lang1}_{target_lang2}.srt"
            else:
                lang = target_lang1 if target_lang1 in download_option else target_lang2
                preview_srt = SubtitleProcessor.format_srt(preview_subtitles, preview_translations, "single", lang)
                full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "single", lang)
                file_name = f"{lang}.srt"
            
            st.text_area("翻譯預覽", value=preview_srt + "\n...", height=300)

            st.download_button(
                label=f"📥 下載 {download_option} 字幕",
                data=full_srt,
                file_name=file_name,
                mime="text/plain"
            )

            missing_translations = [t for t in st.session_state.translated_subtitles if any('[缺失翻譯' in v or '[翻譯失敗]' in v for v in t.values())]
            if missing_translations:
                st.warning(f"⚠️ 注意：有 {len(missing_translations)} 個字幕未能正確翻譯。")
                if st.button("顯示未翻譯的字幕"):
                    for mt in missing_translations:
                        st.text(f"原文: {mt['original']}")
                        st.text(f"{target_lang1}: {mt[target_lang1]}")
                        st.text(f"{target_lang2}: {mt[target_lang2]}")
                        st.text("---")

        except Exception as e:
            st.error(f"❌ 生成預覽或下載文件時發生錯誤：{str(e)}")
            logger.exception("生成預覽或下載文件時發生異常")

    if st.sidebar.checkbox("啟用調試模式"):
        st.sidebar.subheader("調試信息")
        if st.session_state.translated_subtitles:
            st.sidebar.json(st.session_state.translated_subtitles[:5])
        
        if st.sidebar.button("清除翻譯緩存"):
            st.session_state.translated_subtitles = None
            st.session_state.original_subtitles = None
            st.success("翻譯緩存已清除")

    st.sidebar.title("📌 使用說明")
    st.sidebar.markdown("""
    1. 輸入您的 Open AI API 密鑰（將自動保存）
    2. 選擇兩種目標翻譯語言
    3. 設定每種語言的翻譯風格（可選）
    4. 上傳 SRT 格式的字幕文件
    5. 選擇是否使用持續對話
    6. 點擊「開始翻譯」按鈕
    7. 等待翻譯完成後，選擇下載格式並下載翻譯後的字幕文件
    """)

    st.sidebar.title("ℹ️ 關於")
    st.sidebar.info("""
    本工具使用 Open AI 的 AI 模型進行字幕翻譯。
    它支持多種語言組合，並允許自定義翻譯風格。
    如有任何問題或建議，請聯繫開發團隊。
    """)

if __name__ == "__main__":
    bilingual_srt_translator()