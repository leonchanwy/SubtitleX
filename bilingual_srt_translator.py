import streamlit as st
import re
import time
import logging
import json
import concurrent.futures
from typing import List, Tuple, Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type
from openai import OpenAI, RateLimitError, APIError
import anthropic
import ui_utils

# 自定義例外：API 配額用盡
class QuotaExceededError(Exception):
    """當 API 配額用盡時拋出"""
    pass

# 自定義例外：認證失敗
class AuthenticationError(Exception):
    """當 API Key 無效時拋出"""
    pass

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 4000
TEMPERATURE = 0.1
BATCH_SIZE = 30
LANGUAGE_OPTIONS = ["繁體中文", "英文", "日文", "馬來語", "廣東話口語", "德文"]
API_PROVIDERS = ["OpenAI", "Claude"]

# 翻譯失敗占位字串 (統一使用)
FAIL_MARKER_ZH = "[翻譯失敗]"
FAIL_MARKER_EN = "[Translation failed]"
MISSING_MARKER_ZH = "[翻譯缺失]"
MISSING_MARKER_EN = "[Translation missing]"
# 包含已知穩定模型及未來可能模型
CLAUDE_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

class SubtitleProcessor:
    # 常見字幕編碼 (依優先順序)
    ENCODINGS = ['utf-8-sig', 'utf-8', 'utf-16', 'cp950', 'big5', 'gb2312', 'gbk', 'shift_jis', 'latin-1']

    @staticmethod
    def decode_file(raw_bytes: bytes) -> str:
        """嘗試多種編碼解碼檔案內容"""
        for encoding in SubtitleProcessor.ENCODINGS:
            try:
                return raw_bytes.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        # 最後用 latin-1 強制解碼（不會失敗，但可能有亂碼）
        return raw_bytes.decode('latin-1', errors='replace')

    @staticmethod
    def parse_srt(content: str) -> List[Tuple[str, str, str]]:
        content = content.replace('\r\n', '\n').strip()
        pattern = re.compile(
            r'(\d+)\s*\n'  # ID
            r'(\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}).*?\n'  # Timestamp
            r'([\s\S]*?)' # Content
            r'(?=\n+\d+\s*\n\d{1,2}:\d{2}:\d{2}[,.]\d{3}|\Z)',
            re.MULTILINE
        )
        
        matches = []
        for match in pattern.finditer(content):
            raw_id, timestamp, text = match.groups()
            matches.append((raw_id.strip(), timestamp.strip(), text.strip()))
            
        if not matches:
            # Fallback for very simple files
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
        content = re.sub(r'<[^>]+>', '', content) 
        content = re.sub(r'\{\\an\d\}', '', content)
        return content

    @staticmethod
    def format_srt(subtitles: List[Tuple[str, str, str]], translations: List[Dict[str, str]], 
                   format_type: str, lang1: str, lang2: str = None) -> str:
        output = []
        
        for i, (number, timestamp, original_text) in enumerate(subtitles):
            if i < len(translations) and translations[i] is not None:
                translation = translations[i]
            else:
                translation = {'original': original_text, lang1: MISSING_MARKER_ZH, lang2: MISSING_MARKER_EN}

            output.append(f"{number}\n{timestamp}")
            if format_type == "bilingual":
                original = translation.get('original', original_text)
                translated = translation.get(lang1, MISSING_MARKER_ZH)
                output.append(f"{original}\n{translated}")
            elif format_type == "dual_lang":
                lang1_text = translation.get(lang1, MISSING_MARKER_ZH)
                lang2_text = translation.get(lang2, MISSING_MARKER_EN)
                output.append(f"{lang1_text}\n{lang2_text}")
            else:  # 單一語言
                output.append(translation.get(lang1, MISSING_MARKER_ZH))
            output.append("")
        return "\n".join(output).strip()

class SubtitleTranslator:
    def __init__(self, api_key: str, provider: str = "OpenAI"):
        self.provider = provider
        self.api_key = api_key
        self._client = None
        self.conversation_history = []

    def _get_client(self):
        if self._client:
            return self._client
        if self.provider == "OpenAI":
            self._client = OpenAI(api_key=self.api_key)
        else:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def get_available_models(self) -> List[str]:
        client = self._get_client()
        if self.provider == "Claude":
            return CLAUDE_MODELS
        try:
            models = client.models.list()
            chat_models = [
                m.id for m in models.data
                if m.id.startswith(('gpt-', 'o1', 'chatgpt-'))
            ]
            chat_models.sort(reverse=True)
            return chat_models
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

    def _manage_conversation_history(self, max_messages=2):
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]

    def _check_quota_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        if 'insufficient_quota' in error_str or 'exceeded your current quota' in error_str:
            return True
        if hasattr(error, 'code') and error.code == 'insufficient_quota':
            return True
        return False

    def _check_auth_error(self, error: Exception) -> bool:
        error_str = str(error).lower()
        if 'authentication_error' in error_str or 'invalid x-api-key' in error_str:
            return True
        if 'invalid api key' in error_str or '401' in error_str:
            return True
        if hasattr(error, 'status_code') and error.status_code == 401:
            return True
        return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_not_exception_type((QuotaExceededError, AuthenticationError))
    )
    def _translate_batch(self, batch_subtitles: List[Tuple[str, str, str]], target_lang1: str, target_lang2: str,
                         prompt1: str, prompt2: str, model: str, use_history: bool = True,
                         context_subtitles: List[Tuple[str, str, str]] = None) -> List[Dict[str, str]]:

        if self.provider == "OpenAI":
            local_client = OpenAI(api_key=self.api_key)
        else:
            local_client = anthropic.Anthropic(api_key=self.api_key)

        system_prompt = self._create_system_prompt(target_lang1, target_lang2, prompt1, prompt2)

        input_data = [
            {"id": sid, "text": text}
            for sid, _, text in batch_subtitles
        ]

        user_content_parts = []
        if context_subtitles:
            context_text = "\n".join([f"[{sid}] {text}" for sid, _, text in context_subtitles])
            user_content_parts.append(f"【前情提要 - 僅供參考，不需翻譯】\n{context_text}\n")

        user_content_parts.append(f"翻譯以下字幕 (JSON):\n{json.dumps(input_data, ensure_ascii=False, indent=2)}")
        user_content_raw = "\n".join(user_content_parts)
        user_content_with_instruction = user_content_raw + "\n\n請直接輸出符合格式的 JSON，不要包含任何解釋、備註或 markdown 標籤。"

        try:
            if self.provider == "Claude":
                claude_messages = []
                if use_history:
                    self._manage_conversation_history(max_messages=2)
                    claude_messages.extend(self.conversation_history)
                
                claude_messages.append({"role": "user", "content": user_content_with_instruction})
                response_content = self._call_claude_api(system_prompt, claude_messages, model, client=local_client)
            else:
                openai_messages = [{"role": "system", "content": system_prompt}]
                if use_history:
                    self._manage_conversation_history(max_messages=2)
                    openai_messages.extend(self.conversation_history)

                openai_messages.append({"role": "user", "content": user_content_with_instruction})
                schema = self._build_translation_schema(target_lang1, target_lang2)
                response_content = self._call_openai_api(openai_messages, model, client=local_client, response_format=schema)

            if use_history:
                self.conversation_history.append({"role": "user", "content": user_content_raw})
                self.conversation_history.append({"role": "assistant", "content": response_content})

            return self._parse_translation_response(response_content, batch_subtitles, target_lang1, target_lang2)

        except (RateLimitError, APIError, anthropic.RateLimitError, anthropic.APIError, Exception) as e:
            if self._check_auth_error(e):
                logger.error(f"❌ API Key 無效！")
                raise AuthenticationError("API Key 無效，請檢查您的 API Key 是否正確。") from e
            if self._check_quota_error(e):
                logger.error(f"❌ API 配額已用盡！")
                raise QuotaExceededError("API 配額已用盡！") from e
            logger.error(f"API 錯誤：{str(e)}")
            raise

    def _build_translation_schema(self, target_lang1: str, target_lang2: str) -> dict:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "translation_response",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "translations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "original": {"type": "string"},
                                    target_lang1: {"type": "string"},
                                    target_lang2: {"type": "string"}
                                },
                                "required": ["id", "original", target_lang1, target_lang2],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["translations"],
                    "additionalProperties": False
                }
            }
        }

    def _call_openai_api(self, messages: List[Dict], model: str, client: OpenAI,
                         response_format: dict = None) -> str:
        if response_format is None:
            response_format = {"type": "json_object"}

        def make_request(fmt):
            params = {
                "model": model,
                "messages": messages,
                "temperature": TEMPERATURE,
                "response_format": fmt
            }
            try:
                return client.chat.completions.create(**params, max_completion_tokens=MAX_TOKENS)
            except TypeError:
                return client.chat.completions.create(**params, max_tokens=MAX_TOKENS)

        try:
            response = make_request(response_format)
        except Exception as e:
            error_str = str(e).lower()
            if response_format.get("type") == "json_schema" and (
                "response_format" in error_str or
                "json_schema" in error_str or
                "not supported" in error_str
            ):
                logger.warning(f"模型 {model} 不支援 json_schema，降級為 json_object")
                response = make_request({"type": "json_object"})
            else:
                raise

        return response.choices[0].message.content

    def _call_claude_api(self, system_prompt: str, messages: List[Dict], model: str, client: anthropic.Anthropic) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

    def _extract_json(self, text: str) -> str:
        text = text.strip()
        idx = text.find('{')
        if idx == -1:
            return text

        balance = 0
        in_string = False
        escape = False

        for i in range(idx, len(text)):
            char = text[i]
            if escape:
                escape = False
                continue
            if char == '\\' and in_string:
                escape = True
                continue
            if char == '"' and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if char == '{':
                    balance += 1
                elif char == '}':
                    balance -= 1
                    if balance == 0:
                        return text[idx : i+1]
        return text

    def _parse_translation_response(self, response_content: str, batch_subtitles: List[Tuple[str, str, str]],
                                     target_lang1: str, target_lang2: str) -> List[Dict[str, str]]:
        try:
            json_str = self._extract_json(response_content)
            parsed_json = json.loads(json_str)
            translations_list = parsed_json.get("translations", [])
            trans_map = {str(item.get("id")): item for item in translations_list}

            final_results = []
            for sid, _, original_text in batch_subtitles:
                item = trans_map.get(str(sid))
                if item:
                    final_results.append({
                        'original': item.get('original', original_text),
                        target_lang1: item.get(target_lang1, MISSING_MARKER_ZH),
                        target_lang2: item.get(target_lang2, MISSING_MARKER_EN)
                    })
                else:
                    final_results.append({
                        'original': original_text,
                        target_lang1: FAIL_MARKER_ZH,
                        target_lang2: FAIL_MARKER_EN
                    })
            return final_results
        except Exception as e:
            logger.error(f"解析失敗: {e}. Content: {response_content[:200]}...")
            raise ValueError("Model failed to return valid JSON")

    def translate_subtitles(self, subtitles: List[Tuple[str, str, str]],
                            target_lang1: str, target_lang2: str,
                            prompt1: str, prompt2: str,
                            progress_callback, model: str,
                            use_continuous_conversation: bool = True,
                            status_callback=None) -> Tuple[List[Dict[str, str]], List[Dict]]:
        
        self.reset_conversation()
        translated_subtitles = [None] * len(subtitles)
        total = len(subtitles)
        batches = [(i, subtitles[i:i+BATCH_SIZE]) for i in range(0, total, BATCH_SIZE)]
        completed_count = 0
        batch_statuses = []

        for i, (start_idx, batch) in enumerate(batches):
            batch_statuses.append({
                "batch_num": i + 1,
                "start": start_idx + 1,
                "end": min(start_idx + len(batch), total),
                "status": "pending",
                "error": None
            })

        def update_batch_status(batch_idx, status, error=None):
            batch_statuses[batch_idx]["status"] = status
            batch_statuses[batch_idx]["error"] = error
            if status_callback:
                status_callback(batch_statuses)

        if not use_continuous_conversation:
            max_workers = 5
            context_size = 3

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {}
                for batch_idx, (start_idx, batch) in enumerate(batches):
                    if start_idx > 0:
                        context_start = max(0, start_idx - context_size)
                        context = subtitles[context_start:start_idx]
                    else:
                        context = None

                    update_batch_status(batch_idx, "running")
                    future = executor.submit(
                        self._translate_batch,
                        batch, target_lang1, target_lang2, prompt1, prompt2, model, False, context
                    )
                    future_to_batch[future] = (batch_idx, start_idx, batch)

                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_idx, start_idx, batch = future_to_batch[future]
                    try:
                        results = future.result()
                        for j, res in enumerate(results):
                            if start_idx + j < total:
                                translated_subtitles[start_idx + j] = res
                        update_batch_status(batch_idx, "success")
                    except Exception as e:
                        error_msg = str(e)[:100]
                        logger.error(f"Batch {batch_idx + 1} failed: {e}")
                        update_batch_status(batch_idx, "failed", error_msg)
                        for j, (_, _, text) in enumerate(batch):
                            if start_idx + j < total:
                                translated_subtitles[start_idx + j] = {
                                    'original': text,
                                    target_lang1: FAIL_MARKER_ZH,
                                    target_lang2: FAIL_MARKER_EN
                                }
                    completed_count += len(batch)
                    progress_callback(min(completed_count / total, 1.0))
        else:
            for batch_idx, (start_idx, batch) in enumerate(batches):
                update_batch_status(batch_idx, "running")
                try:
                    results = self._translate_batch(batch, target_lang1, target_lang2, prompt1, prompt2, model, True)
                    for j, res in enumerate(results):
                        if start_idx + j < total:
                            translated_subtitles[start_idx + j] = res
                    update_batch_status(batch_idx, "success")
                except Exception as e:
                    error_msg = str(e)[:100]
                    logger.error(f"Batch {batch_idx + 1} failed: {e}")
                    update_batch_status(batch_idx, "failed", error_msg)
                    for j, (_, _, text) in enumerate(batch):
                        if start_idx + j < total:
                            translated_subtitles[start_idx + j] = {
                                'original': text,
                                target_lang1: FAIL_MARKER_ZH,
                                target_lang2: FAIL_MARKER_EN
                            }
                completed_count += len(batch)
                progress_callback(min(completed_count / total, 1.0))
                time.sleep(0.5)

        final_output = []
        for i, item in enumerate(translated_subtitles):
            if item is None:
                orig = subtitles[i][2] if i < len(subtitles) else ""
                final_output.append({'original': orig, target_lang1: '[System Error]', target_lang2: '[System Error]'})
            else:
                final_output.append(item)

        return final_output, batch_statuses

    def reset_conversation(self):
        self.conversation_history = []

def bilingual_srt_translator():
    ui_utils.render_header("🌐 Bilingual Subtitle Translator", "Translate your subtitles into two languages simultaneously using AI.")
    
    # Check API keys first
    if not ui_utils.validate_api_inputs(["OpenAI", "Claude"]):
        st.stop()

    # Provider Selection
    col_api, col_model = st.columns(2)
    with col_api:
        api_provider = st.selectbox(
            "API Provider",
            options=API_PROVIDERS,
            help="Choose between OpenAI (GPT) or Anthropic (Claude)."
        )
    
    # Get API key from session state
    api_key = ui_utils.get_api_key(api_provider)
    
    # Initialize Translator to get models
    try:
        temp_translator = SubtitleTranslator(api_key, api_provider)
        available_models = temp_translator.get_available_models()
        if not available_models:
             # Fallback defaults if list fails
            available_models = [DEFAULT_CLAUDE_MODEL] if api_provider == "Claude" else [DEFAULT_OPENAI_MODEL]
    except Exception:
        available_models = [DEFAULT_CLAUDE_MODEL] if api_provider == "Claude" else [DEFAULT_OPENAI_MODEL]

    with col_model:
        model_name = st.selectbox("Model", options=available_models)

    # Language Settings
    st.subheader("Language Settings")
    col1, col2 = st.columns(2)
    with col1:
        target_lang1 = st.selectbox("Target Language 1", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("廣東話口語"))
        prompt1 = st.text_input("Style for Lang 1", value="口語化帶俚語")
    with col2:
        target_lang2 = st.selectbox("Target Language 2", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("英文"))
        prompt2 = st.text_input("Style for Lang 2", value="都市俚語")

    # File Upload
    uploaded_file = st.file_uploader("Upload SRT File", type="srt")
    
    # Advanced Options
    with st.expander("Advanced Options"):
        use_continuous_conversation = st.checkbox(
            "Use Continuous Context", 
            value=True,
            help="Better consistency but slower. Uncheck for parallel processing (faster)."
        )

    if uploaded_file and st.button("Start Translation", type="primary"):
        try:
            content = SubtitleProcessor.decode_file(uploaded_file.getvalue())
            content = SubtitleProcessor.clean_text(content)
            subtitles = SubtitleProcessor.parse_srt(content)

            # Initialize Translator
            translator = SubtitleTranslator(api_key, api_provider)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            batch_status_container = st.empty()

            def display_batch_status(statuses):
                status_icons = {"pending": "⏳", "running": "🔄", "success": "✅", "failed": "❌"}
                lines = []
                # Only show active or failed/completed batches to save space
                active_statuses = [s for s in statuses if s["status"] != "pending"]
                # Show last 5 updates
                for s in active_statuses[-5:]:
                    icon = status_icons.get(s["status"], "❓")
                    line = f"{icon} Batch {s['batch_num']} ({s['start']}-{s['end']})"
                    if s["status"] == "failed":
                        line += f": {s['error']}"
                    lines.append(line)
                if len(active_statuses) > 5:
                    lines.insert(0, f"... ({len(active_statuses)-5} previous batches hidden)")
                batch_status_container.text("\n".join(lines))

            with st.spinner(f"Translating with {api_provider} {model_name}..."):
                start_time = time.time()
                translated, batch_statuses = translator.translate_subtitles(
                    subtitles, target_lang1, target_lang2, prompt1, prompt2,
                    progress_bar.progress, model_name, use_continuous_conversation,
                    status_callback=display_batch_status
                )
                
                # Store results in session state
                st.session_state.translated_subtitles = translated
                st.session_state.original_subtitles = subtitles
                st.session_state.translated_lang1 = target_lang1
                st.session_state.translated_lang2 = target_lang2
                
                processing_time = time.time() - start_time
                status_text.success(f"✅ Finished in {processing_time:.2f}s")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.exception("Translation failed")

    # Results & Download
    if st.session_state.get('translated_subtitles'):
        st.divider()
        st.subheader("Download Results")
        
        # Helper variables
        t_lang1 = st.session_state.get('translated_lang1', target_lang1)
        t_lang2 = st.session_state.get('translated_lang2', target_lang2)
        orig_subs = st.session_state.get('original_subtitles')
        trans_subs = st.session_state.get('translated_subtitles')

        download_format = st.selectbox(
            "Format",
            [f"Original + {t_lang1}", f"Original + {t_lang2}", f"{t_lang1} + {t_lang2}", f"Only {t_lang1}", f"Only {t_lang2}"]
        )
        
        # Generate SRT content based on selection
        try:
            format_type = "single"
            l1 = t_lang1
            l2 = None
            
            if "Original" in download_format:
                format_type = "bilingual"
                l1 = t_lang1 if t_lang1 in download_format else t_lang2
            elif "+" in download_format:
                format_type = "dual_lang"
                l1 = t_lang1
                l2 = t_lang2
            else:
                l1 = t_lang1 if t_lang1 in download_format else t_lang2

            full_srt = SubtitleProcessor.format_srt(orig_subs, trans_subs, format_type, l1, l2)
            file_name = f"subtitle_{l1}_{l2 if l2 else ''}.srt".replace("__", "_")

            col_preview, col_dl = st.columns([2, 1])
            with col_preview:
                st.text_area("Preview (First 10 lines)", value="\n".join(full_srt.split("\n")[:20]), height=200)
            
            with col_dl:
                st.download_button(
                    label="📥 Download SRT",
                    data=full_srt,
                    file_name=file_name,
                    mime="text/plain",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error generating download: {e}")

if __name__ == "__main__":
    bilingual_srt_translator()