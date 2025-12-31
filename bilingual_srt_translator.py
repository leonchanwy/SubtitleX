import streamlit as st
import re
import time
import logging
import json
from typing import List, Tuple, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
from openai import OpenAI, RateLimitError, APIError

# 自定義例外：API 配額用盡
class QuotaExceededError(Exception):
    """當 OpenAI API 配額用盡時拋出"""
    pass

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
DEFAULT_MODEL = "gpt-4o-2024-08-06"
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
        # 更穩健的 regex：匹配 ID, 時間軸, 和內容 (直到下一個 ID 或文件結束)
        # 使用多行模式，並允許時間軸格式有些微容錯
        pattern = re.compile(
            r'(\d+)\s*\n'  # ID
            r'(\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}).*?\n'  # Timestamp
            r'([\s\S]*?)(?=\n+\d+\s*\n|\Z)', # Content
            re.MULTILINE
        )
        
        matches = []
        for match in pattern.finditer(content):
            raw_id, timestamp, text = match.groups()
            matches.append((raw_id.strip(), timestamp.strip(), text.strip()))
            
        if not matches:
            # Fallback for very simple files or different formatting
            logger.warning("Standard regex failed, trying simple block split")
            blocks = content.split('\n\n')
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3 and lines[0].isdigit() and '-->' in lines[1]:
                    matches.append((lines[0].strip(), lines[1].strip(), "\n".join(lines[2:])))

        if not matches:
            raise ValueError("無法解析SRT文件。請確保文件格式正確。")
        return matches

    @staticmethod
    def clean_text(content: str) -> str:
        content = re.sub(r'</?[a-z]+>', '', content) # Remove HTML tags
        content = re.sub(r'\{\\an\d\}', '', content) # Remove alignment tags
        return content

    @staticmethod
    def format_srt(subtitles: List[Tuple[str, str, str]], translations: List[Dict[str, str]], 
                   format_type: str, lang1: str, lang2: str = None) -> str:
        output = []
        # Ensure we don't index out of bounds if lengths differ (though they shouldn't now)
        limit = min(len(subtitles), len(translations))
        
        for i in range(limit):
            number, timestamp, original_text = subtitles[i]
            translation = translations[i]
            
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

    def get_available_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            # 簡單過濾出 gpt 開頭的模型，並按名稱排序
            gpt_models = [m.id for m in models.data if m.id.startswith('gpt')]
            gpt_models.sort(reverse=True)
            return gpt_models
        except Exception as e:
            logger.error(f"無法獲取模型列表: {e}")
            return []

    def _create_system_prompt(self, target_lang1: str, target_lang2: str, prompt1: str, prompt2: str) -> str:
        return f"""你是專業字幕翻譯者。將提供的字幕翻譯成{target_lang1}和{target_lang2}。

請輸出嚴格的 JSON 格式。
JSON 結構必須為：
{{
  "translations": [
    {{
      "id": "字幕ID",
      "original": "原文內容",
      "{target_lang1}": "翻譯1",
      "{target_lang2}": "翻譯2"
    }}
  ]
}}

翻譯規則：
1. 保持原有語氣和口語化風格。
2. 保留專有名詞原文。
3. {target_lang1}風格：{prompt1}
4. {target_lang2}風格：{prompt2}
5. 確保輸出的 "id" 與輸入的字幕編號完全對應。
"""

    def _manage_conversation_history(self, max_messages=10):
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def _check_quota_error(self, error: Exception) -> bool:
        """檢查是否為配額用盡錯誤"""
        error_str = str(error).lower()
        if 'insufficient_quota' in error_str or 'exceeded your current quota' in error_str:
            return True
        if hasattr(error, 'code') and error.code == 'insufficient_quota':
            return True
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_not_exception_type(QuotaExceededError)
    )
    def _translate_batch(self, batch_subtitles: List[Tuple[str, str, str]], target_lang1: str, target_lang2: str, prompt1: str, prompt2: str, model: str) -> List[Dict[str, str]]:
        system_prompt = self._create_system_prompt(target_lang1, target_lang2, prompt1, prompt2)

        self._manage_conversation_history()

        # Prepare user content as a structured list for clarity in prompt
        # Using JSON string in user prompt helps the model understand the structure
        input_data = [
            {"id": sid, "text": text} 
            for sid, _, text in batch_subtitles
        ]
        user_content = f"翻譯以下字幕 (JSON):\n{json.dumps(input_data, ensure_ascii=False, indent=2)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"}  # 強制 JSON 模式
            )

            response_content = response.choices[0].message.content
            
            # Update history
            self.conversation_history.append({"role": "user", "content": user_content})
            self.conversation_history.append({"role": "assistant", "content": response_content})

            # Parse JSON response
            try:
                parsed_json = json.loads(response_content)
                translations_list = parsed_json.get("translations", [])
                
                # Create a map for easy lookup by ID
                trans_map = {str(item.get("id")): item for item in translations_list}
                
                final_results = []
                for sid, _, original_text in batch_subtitles:
                    item = trans_map.get(str(sid))
                    if item:
                        final_results.append({
                            'original': item.get('original', original_text),
                            target_lang1: item.get(target_lang1, '[翻譯缺失]'),
                            target_lang2: item.get(target_lang2, '[Translation missing]')
                        })
                    else:
                        logger.warning(f"ID {sid} 的翻譯在回應中遺失")
                        final_results.append({
                            'original': original_text,
                            target_lang1: '[翻譯失敗]',
                            target_lang2: '[Translation failed]'
                        })
                
                return final_results

            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失敗: {e}, 內容: {response_content[:100]}...")
                raise ValueError("Model failed to return valid JSON")

        except (RateLimitError, APIError) as e:
            if self._check_quota_error(e):
                logger.error(f"❌ API 配額已用盡！錯誤：{str(e)}")
                raise QuotaExceededError(
                    "OpenAI API 配額已用盡！請檢查您的帳戶餘額和計費詳情。"
                ) from e
            logger.error(f"API 錯誤：{str(e)}")
            raise
        except Exception as e:
            if self._check_quota_error(e):
                logger.error(f"❌ API 配額已用盡！錯誤：{str(e)}")
                raise QuotaExceededError(
                    "OpenAI API 配額已用盡！請檢查您的帳戶餘額和計費詳情。"
                ) from e
            logger.error(f"API 錯誤：{str(e)}")
            raise

    def translate_subtitles(self, subtitles: List[Tuple[str, str, str]],
                            target_lang1: str, target_lang2: str,
                            prompt1: str, prompt2: str,
                            progress_callback, model: str) -> List[Dict[str, str]]:
        translated_subtitles = []
        total = len(subtitles)

        for i in range(0, total, BATCH_SIZE):
            batch = subtitles[i:i+BATCH_SIZE]
            # Pass the full batch info (including IDs) to _translate_batch
            try:
                translations = self._translate_batch(batch, target_lang1, target_lang2, prompt1, prompt2, model)
                translated_subtitles.extend(translations)
            except QuotaExceededError:
                logger.error("❌ 偵測到 API 配額用盡，立即停止翻譯！")
                raise
            except Exception as e:
                logger.error(f"翻譯批次 {i//BATCH_SIZE + 1} 失敗：{str(e)}")
                # Create placeholders for this failed batch
                for _, _, text in batch:
                    translated_subtitles.append({
                        'original': text, 
                        target_lang1: '[翻譯失敗]', 
                        target_lang2: '[Translation failed]'
                    })

            progress = min((i + BATCH_SIZE) / total, 1.0)
            progress_callback(progress)

            time.sleep(1)  # 避免 API 速率限制

        return translated_subtitles

    def reset_conversation(self):
        self.conversation_history = []

def validate_api_key(api_key: str) -> bool:
    """Validate if the API key is non-empty and has a reasonable format."""
    if not api_key or not isinstance(api_key, str):
        return False
    api_key = api_key.strip()
    return len(api_key) > 0

def bilingual_srt_translator():
    init_session_state()
    st.title("🌐 雙語字幕翻譯器（GPT-4o）")

    api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.session_state.api_key_valid = validate_api_key(api_key)
        # Reset available models when API key changes
        if 'available_models' in st.session_state:
            del st.session_state.available_models

    # 初始化可用模型列表
    if 'available_models' not in st.session_state:
        st.session_state.available_models = [DEFAULT_MODEL]

    # 嘗試獲取模型列表（當 API Key 有效且列表尚未更新時）
    if st.session_state.api_key_valid and len(st.session_state.available_models) == 1:
        try:
            temp_translator = SubtitleTranslator(st.session_state.api_key)
            fetched_models = temp_translator.get_available_models()
            if fetched_models:
                st.session_state.available_models = fetched_models
                # 確保默認模型在列表中
                if DEFAULT_MODEL not in st.session_state.available_models:
                     st.session_state.available_models.insert(0, DEFAULT_MODEL)
        except Exception as e:
            logger.warning(f"無法自動獲取模型列表: {e}")

    # 模型選擇下拉菜單
    try:
        default_index = st.session_state.available_models.index(DEFAULT_MODEL)
    except ValueError:
        default_index = 0
        
    model_name = st.selectbox(
        "選擇模型 (Select Model)", 
        options=st.session_state.available_models, 
        index=default_index,
        help="從您的 OpenAI 帳戶中獲取可用模型列表。如果無法獲取，將顯示默認模型。"
    )

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

            # 檢查是否需要創建新的 translator（首次或 API key 變更）
            if 'translator' not in st.session_state or st.session_state.get('translator_api_key') != api_key:
                st.session_state.translator = SubtitleTranslator(api_key)
                st.session_state.translator_api_key = api_key

            if not use_continuous_conversation:
                st.session_state.translator.reset_conversation()

            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner(f"正在使用 {model_name} 翻譯..."):
                start_time = time.time()
                st.session_state.translated_subtitles = st.session_state.translator.translate_subtitles(
                    subtitles, target_lang1, target_lang2, prompt1, prompt2, progress_bar.progress, model_name
                )
                st.session_state.original_subtitles = subtitles
                # 儲存翻譯時使用的語言選項
                st.session_state.translated_lang1 = target_lang1
                st.session_state.translated_lang2 = target_lang2
                end_time = time.time()

            processing_time = end_time - start_time
            status_text.success(f"✅ 翻譯完成！總處理時間：{processing_time:.2f} 秒")

        except QuotaExceededError as e:
            st.error(f"""
            ❌ **API 配額已用盡！翻譯已停止。**

            {str(e)}

            **解決方案：**
            1. 登入 [OpenAI Platform](https://platform.openai.com/account/billing)
            2. 檢查您的帳戶餘額
            3. 添加付款方式或購買更多額度
            """)
            logger.error("API 配額用盡，翻譯已停止")
        except Exception as e:
            st.error(f"❌ 處理過程中發生錯誤：{str(e)}")
            logger.exception("翻譯過程中發生異常")

    if st.session_state.translated_subtitles:
        # 使用翻譯時儲存的語言選項
        trans_lang1 = st.session_state.get('translated_lang1', target_lang1)
        trans_lang2 = st.session_state.get('translated_lang2', target_lang2)

        st.subheader("翻譯結果")
        download_option = st.selectbox(
            "選擇下載格式",
            [f"原文 + {trans_lang1}", f"原文 + {trans_lang2}", f"{trans_lang1} + {trans_lang2}", f"僅 {trans_lang1}", f"僅 {trans_lang2}"]
        )

        preview_subtitles = st.session_state.original_subtitles[:5]
        preview_translations = st.session_state.translated_subtitles[:5]

        try:
            if "原文" in download_option:
                lang = trans_lang1 if trans_lang1 in download_option else trans_lang2
                preview_srt = SubtitleProcessor.format_srt(preview_subtitles, preview_translations, "bilingual", lang)
                full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "bilingual", lang)
                file_name = f"原文_{lang}.srt"
            elif "+" in download_option:
                preview_srt = SubtitleProcessor.format_srt(preview_subtitles, preview_translations, "dual_lang", trans_lang1, trans_lang2)
                full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "dual_lang", trans_lang1, trans_lang2)
                file_name = f"{trans_lang1}_{trans_lang2}.srt"
            else:
                lang = trans_lang1 if trans_lang1 in download_option else trans_lang2
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
                        st.text(f"{trans_lang1}: {mt.get(trans_lang1, '[無]')}")
                        st.text(f"{trans_lang2}: {mt.get(trans_lang2, '[無]')}")
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
    1. 輸入您的 OpenAI API 密鑰
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