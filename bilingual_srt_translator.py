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
DEFAULT_OPENAI_MODEL = "gpt-5.2"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"
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
    "claude-opus-4-5",
    "claude-haiku-4-5",
]

def init_session_state():
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    if 'claude_api_key' not in st.session_state:
        st.session_state.claude_api_key = ''
    if 'api_provider' not in st.session_state:
        st.session_state.api_provider = 'OpenAI'

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
        # 修復：更嚴格的 Lookahead，要求下一塊必須是 "數字 + 換行 + 時間軸"
        pattern = re.compile(
            r'(\d+)\s*\n'  # ID
            r'(\d{1,2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{1,2}:\d{2}:\d{2}[,.]\d{3}).*?\n'  # Timestamp
            r'([\s\S]*?)' # Content
            r'(?=\n+\d+\s*\n\d{1,2}:\d{2}:\d{2}[,.]\d{3}|\Z)', # Lookahead checking for ID AND Timestamp
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
        # 修復：移除所有 HTML 標籤（包括帶屬性的）
        content = re.sub(r'<[^>]+>', '', content) 
        content = re.sub(r'\{\\an\d\}', '', content) # Remove alignment tags
        return content

    @staticmethod
    def format_srt(subtitles: List[Tuple[str, str, str]], translations: List[Dict[str, str]], 
                   format_type: str, lang1: str, lang2: str = None) -> str:
        output = []
        
        # 安全長度限制 (以字幕數量為準，不足補空)
        # 不使用 min(len(subtitles), len(translations)) 以避免截斷
        
        for i, (number, timestamp, original_text) in enumerate(subtitles):
            # 安全獲取翻譯，若無則提供錯誤占位
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
        # 初始化時不強制建立 client，改在呼叫時處理或使用 _get_client
        self._client = None
        self.conversation_history = []

    def _get_client(self):
        """獲取或建立 Client 實例 (非線程安全，僅供單線程或主線程使用)"""
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
            # 過濾出 Chat Completions 可用的模型 (gpt-*, o*, chatgpt-*)
            chat_models = [
                m.id for m in models.data
                if m.id.startswith(('gpt-', 'o1', 'o3', 'o4', 'chatgpt-'))
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
        """
        管理對話歷史。
        改為預設保留較少的對話(2則：一問一答)，避免 Context Window 爆炸。
        """
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

    def _check_auth_error(self, error: Exception) -> bool:
        """檢查是否為認證錯誤（API Key 無效）"""
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

        # 建立局部 client 以確保線程安全 (並行模式下)
        if self.provider == "OpenAI":
            local_client = OpenAI(api_key=self.api_key)
        else:
            local_client = anthropic.Anthropic(api_key=self.api_key)

        system_prompt = self._create_system_prompt(target_lang1, target_lang2, prompt1, prompt2)

        input_data = [
            {"id": sid, "text": text}
            for sid, _, text in batch_subtitles
        ]

        # 構建用戶內容（可含上下文）
        user_content_parts = []

        # 並行模式下加入前情提要
        if context_subtitles:
            context_text = "\n".join([f"[{sid}] {text}" for sid, _, text in context_subtitles])
            user_content_parts.append(f"【前情提要 - 僅供參考，不需翻譯】\n{context_text}\n")

        user_content_parts.append(f"翻譯以下字幕 (JSON):\n{json.dumps(input_data, ensure_ascii=False, indent=2)}")
        user_content_raw = "\n".join(user_content_parts)

        # 強調 JSON 指令
        user_content_with_instruction = user_content_raw + "\n\n請直接輸出符合格式的 JSON，不要包含任何解釋、備註或 markdown 標籤。"

        try:
            if self.provider == "Claude":
                # 構造消息列表
                claude_messages = []
                if use_history:
                    # 注意：在多線程環境下修改 self.conversation_history 會有競爭條件
                    # 但此處 use_history=False 用於並行模式，use_history=True 用於串行模式，所以暫時安全
                    self._manage_conversation_history(max_messages=2)
                    claude_messages.extend(self.conversation_history)
                
                claude_messages.append({"role": "user", "content": user_content_with_instruction})
                response_content = self._call_claude_api(system_prompt, claude_messages, model, client=local_client)
            else:
                # OpenAI 構造
                openai_messages = [{"role": "system", "content": system_prompt}]
                if use_history:
                    self._manage_conversation_history(max_messages=2)
                    openai_messages.extend(self.conversation_history)

                openai_messages.append({"role": "user", "content": user_content_with_instruction})
                # 使用 Structured Outputs (json_schema) 確保輸出格式
                schema = self._build_translation_schema(target_lang1, target_lang2)
                response_content = self._call_openai_api(openai_messages, model, client=local_client, response_format=schema)

            # 更新歷史 (僅在啟用歷史且成功時，串行模式下)
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
        """建構 OpenAI Structured Outputs 的 JSON Schema"""
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
        """調用 OpenAI API（支援 json_schema 降級到 json_object）"""
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
                # Fallback for older SDKs
                return client.chat.completions.create(**params, max_tokens=MAX_TOKENS)

        try:
            response = make_request(response_format)
        except Exception as e:
            error_str = str(e).lower()
            # 若 json_schema 不支援，降級為 json_object
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
        """調用 Claude API"""
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE, # 新增 temperature
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text

    def _extract_json(self, text: str) -> str:
        """使用括號計數提取第一個合法的 JSON 物件（正確處理字串內的括號）"""
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
        """解析翻譯回應"""
        try:
            # 使用增強的 JSON 提取 (替代 greedy regex)
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
        """
        翻譯字幕
        返回: (翻譯結果列表, batch 狀態列表)
        """
        # 重要：每次翻譯新文件前先重置對話歷史
        self.reset_conversation()

        # 初始化為 None 的列表，長度與字幕相同
        translated_subtitles = [None] * len(subtitles)
        total = len(subtitles)
        batches = [(i, subtitles[i:i+BATCH_SIZE]) for i in range(0, total, BATCH_SIZE)]
        completed_count = 0
        batch_statuses = []  # 記錄每個 batch 的狀態

        # 初始化 batch 狀態
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
            # 並行模式 (不使用歷史，但加入前情提要)
            max_workers = 5
            context_size = 3  # 前情提要的句數

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {}
                for batch_idx, (start_idx, batch) in enumerate(batches):
                    # 取前 context_size 句作為上下文（第一個 batch 沒有上下文）
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
            # 串行模式 (持續對話)
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

        # 確保回傳結果不被過濾，若有 None (理論上不應有) 則補全
        final_output = []
        for i, item in enumerate(translated_subtitles):
            if item is None:
                # Fallback if something went wrong
                orig = subtitles[i][2] if i < len(subtitles) else ""
                final_output.append({'original': orig, target_lang1: '[System Error]', target_lang2: '[System Error]'})
            else:
                final_output.append(item)

        return final_output, batch_statuses

    def reset_conversation(self):
        self.conversation_history = []

def validate_api_key(api_key: str) -> bool:
    """基本格式驗證"""
    if not api_key or not isinstance(api_key, str):
        return False
    api_key = api_key.strip()
    return len(api_key) > 0

def verify_api_key(api_key: str, provider: str) -> Tuple[bool, str]:
    """
    實際調用 API 驗證 Key 是否有效及配額狀態
    返回: (是否有效, 狀態訊息)
    """
    if not validate_api_key(api_key):
        return False, ""

    try:
        if provider == "OpenAI":
            client = OpenAI(api_key=api_key)
            # 用最小請求測試 API Key
            client.models.list()
            return True, "✅ API Key 有效"
        else:  # Claude
            client = anthropic.Anthropic(api_key=api_key)
            # Claude 用簡單訊息測試
            client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}]
            )
            return True, "✅ API Key 有效"
    except Exception as e:
        error_str = str(e).lower()
        if 'authentication' in error_str or 'invalid' in error_str or '401' in error_str:
            return False, "❌ API Key 無效"
        elif 'insufficient_quota' in error_str or 'exceeded' in error_str or 'billing' in error_str:
            return False, "❌ API 配額不足，請檢查帳戶餘額"
        elif 'rate_limit' in error_str:
            return True, "⚠️ API Key 有效（但目前速率受限）"
        else:
            return False, f"❌ 驗證失敗: {str(e)[:50]}"

def bilingual_srt_translator():
    init_session_state()
    st.title("🌐 雙語字幕翻譯器 (Enhanced)")

    api_provider = st.selectbox(
        "選擇 API 提供商",
        options=API_PROVIDERS,
        index=API_PROVIDERS.index(st.session_state.api_provider),
        help="選擇使用 OpenAI 或 Claude API"
    )

    if api_provider != st.session_state.api_provider:
        st.session_state.api_provider = api_provider
        if 'available_models' in st.session_state:
            del st.session_state.available_models
        if 'translator' in st.session_state:
            del st.session_state.translator
        # 切換 provider 時清除驗證狀態
        if 'api_key_status' in st.session_state:
            del st.session_state.api_key_status

    # 根據 provider 使用對應的 API Key
    api_key_state = 'openai_api_key' if api_provider == 'OpenAI' else 'claude_api_key'
    current_key = st.session_state[api_key_state]

    api_key_label = f"{api_provider} API Key"
    api_key = st.text_input(api_key_label, value=current_key, type="password")

    # API Key 變更時自動驗證
    if api_key != current_key:
        st.session_state[api_key_state] = api_key
        if 'available_models' in st.session_state:
            del st.session_state.available_models
        # 自動驗證新輸入的 Key
        if api_key and len(api_key.strip()) > 10:  # 基本長度檢查
            with st.spinner("驗證 API Key..."):
                is_valid, status_msg = verify_api_key(api_key, api_provider)
                st.session_state.api_key_status = status_msg
                st.session_state.api_key_verified = is_valid
        else:
            st.session_state.api_key_status = ""
            st.session_state.api_key_verified = False

    # 顯示驗證狀態
    if st.session_state.get('api_key_status'):
        if st.session_state.get('api_key_verified', False):
            st.success(st.session_state.api_key_status)
        else:
            st.error(st.session_state.api_key_status)

    api_key_valid = validate_api_key(api_key)

    default_model = DEFAULT_CLAUDE_MODEL if api_provider == "Claude" else DEFAULT_OPENAI_MODEL

    if 'available_models' not in st.session_state:
        st.session_state.available_models = [default_model]

    # 只在驗證成功後才獲取模型列表
    if api_key_valid and st.session_state.get('api_key_verified', False) and len(st.session_state.available_models) == 1:
        try:
            temp_translator = SubtitleTranslator(api_key, api_provider)
            fetched_models = temp_translator.get_available_models()
            if fetched_models:
                st.session_state.available_models = fetched_models
                if default_model not in st.session_state.available_models:
                    st.session_state.available_models.insert(0, default_model)
        except Exception as e:
            st.warning(f"無法獲取模型列表: {e}")

    try:
        default_index = st.session_state.available_models.index(default_model)
    except ValueError:
        default_index = 0

    model_name = st.selectbox(
        "選擇模型 (Select Model)",
        options=st.session_state.available_models,
        index=default_index
    )

    col1, col2 = st.columns(2)
    with col1:
        target_lang1 = st.selectbox("目標語言 1", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("廣東話口語"))
        prompt1 = st.text_input("語言 1 翻譯風格", value="口語化帶俚語")
    with col2:
        target_lang2 = st.selectbox("目標語言 2", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("英文"))
        prompt2 = st.text_input("語言 2 翻譯風格", value="都市俚語")

    uploaded_file = st.file_uploader("選擇 SRT 文件", type="srt")

    # 改進的選項說明
    use_continuous_conversation = st.checkbox(
        "使用持續對話 (開啟=高品質上下文，關閉=極速並行翻譯)", 
        value=True,
        help="開啟：模型會記得上一段翻譯內容，術語更一致，但速度較慢。會自動管理對話長度以避免 Context Window 限制。\n關閉：同時翻譯多個片段，速度極快，但前後文關聯較弱。"
    )
    
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
            content = SubtitleProcessor.decode_file(uploaded_file.getvalue())
            content = SubtitleProcessor.clean_text(content)
            subtitles = SubtitleProcessor.parse_srt(content)

            # 確保 translator 存在且 key 正確
            needs_new_translator = (
                'translator' not in st.session_state or
                st.session_state.get('translator_api_key') != api_key or
                st.session_state.get('translator_provider') != api_provider
            )
            if needs_new_translator:
                st.session_state.translator = SubtitleTranslator(api_key, api_provider)
                st.session_state.translator_api_key = api_key
                st.session_state.translator_provider = api_provider

            if not use_continuous_conversation:
                st.session_state.translator.reset_conversation()

            progress_bar = st.progress(0)
            status_text = st.empty()
            batch_status_container = st.empty()

            # 用於顯示 batch 狀態的回調函數
            def display_batch_status(statuses):
                status_icons = {"pending": "⏳", "running": "🔄", "success": "✅", "failed": "❌"}
                lines = []
                for s in statuses:
                    icon = status_icons.get(s["status"], "❓")
                    line = f"{icon} Batch {s['batch_num']} (字幕 {s['start']}-{s['end']})"
                    if s["status"] == "failed" and s["error"]:
                        line += f": {s['error']}"
                    lines.append(line)
                batch_status_container.text("\n".join(lines))

            mode_text = "串行上下文模式" if use_continuous_conversation else "並行極速模式"
            with st.spinner(f"正在使用 {api_provider} {model_name} 翻譯 ({mode_text})..."):
                start_time = time.time()
                translated, batch_statuses = st.session_state.translator.translate_subtitles(
                    subtitles, target_lang1, target_lang2, prompt1, prompt2,
                    progress_bar.progress, model_name, use_continuous_conversation,
                    status_callback=display_batch_status
                )
                st.session_state.translated_subtitles = translated
                st.session_state.batch_statuses = batch_statuses
                st.session_state.original_subtitles = subtitles
                st.session_state.translated_lang1 = target_lang1
                st.session_state.translated_lang2 = target_lang2
                end_time = time.time()

            processing_time = end_time - start_time

            # 統計失敗的 batch
            failed_batches = [s for s in batch_statuses if s["status"] == "failed"]

            if failed_batches:
                status_text.warning(f"⚠️ 翻譯完成（{len(failed_batches)} 個批次失敗）- 處理時間：{processing_time:.2f} 秒")
                # 顯示失敗詳情
                with st.expander("查看失敗批次詳情", expanded=True):
                    for s in failed_batches:
                        st.error(f"Batch {s['batch_num']} (字幕 {s['start']}-{s['end']}): {s['error']}")
            else:
                status_text.success(f"✅ 翻譯完成！總處理時間：{processing_time:.2f} 秒 ({mode_text})")

            if len(st.session_state.translated_subtitles) != len(subtitles):
                st.warning(f"⚠️ 警告：原文有 {len(subtitles)} 句，但翻譯結果只有 {len(st.session_state.translated_subtitles)} 句。輸出可能不完整。")

        except AuthenticationError as e:
            st.error(f"❌ **API Key 無效！**\n\n{str(e)}\n\n請確認您輸入的是正確的 {api_provider} API Key。")
            logger.error("API Key 無效")
        except QuotaExceededError as e:
            billing_url = "https://console.anthropic.com/settings/billing" if api_provider == "Claude" else "https://platform.openai.com/account/billing"
            st.error(f"❌ **API 配額已用盡！**\n\n{str(e)}\n\n[前往計費設置]({billing_url})")
            logger.error("API 配額用盡")
        except Exception as e:
            st.error(f"❌ 處理過程中發生錯誤：{str(e)}")
            logger.exception("翻譯過程中發生異常")

    if st.session_state.translated_subtitles:
        trans_lang1 = st.session_state.get('translated_lang1', target_lang1)
        trans_lang2 = st.session_state.get('translated_lang2', target_lang2)

        st.subheader("翻譯結果")
        download_option = st.selectbox(
            "選擇下載格式",
            [f"原文 + {trans_lang1}", f"原文 + {trans_lang2}", f"{trans_lang1} + {trans_lang2}", f"僅 {trans_lang1}", f"僅 {trans_lang2}"]
        )

        try:
            if "原文" in download_option:
                lang = trans_lang1 if trans_lang1 in download_option else trans_lang2
                full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "bilingual", lang)
                file_name = f"原文_{lang}.srt"
            elif "+" in download_option:
                full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "dual_lang", trans_lang1, trans_lang2)
                file_name = f"{trans_lang1}_{trans_lang2}.srt"
            else:
                lang = trans_lang1 if trans_lang1 in download_option else trans_lang2
                full_srt = SubtitleProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "single", lang)
                file_name = f"{lang}.srt"

            st.text_area("翻譯預覽", value=full_srt, height=400)

            st.download_button(
                label=f"📥 下載 {download_option} 字幕",
                data=full_srt,
                file_name=file_name,
                mime="text/plain"
            )

            missing_translations = [t for t in st.session_state.translated_subtitles if t and any(
                v in (FAIL_MARKER_ZH, FAIL_MARKER_EN, MISSING_MARKER_ZH, MISSING_MARKER_EN) for v in t.values()
            )]
            if missing_translations:
                st.warning(f"⚠️ 注意：有 {len(missing_translations)} 個字幕未能正確翻譯。\n")

        except Exception as e:
            st.error(f"❌ 生成預覽或下載文件時發生錯誤：{str(e)}")
            logger.exception("生成預覽異常")

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
    1. 選擇 API 提供商（OpenAI 或 Claude）
    2. 輸入 API Key
    3. 設定目標語言和風格
    4. 上傳 SRT 文件
    5. **選擇模式**：
       - **持續對話 (勾選)**：適合需要上下文連貫的翻譯，速度較慢。
       - **並行翻譯 (不勾選)**：適合追求速度，速度極快。
    6. 點擊「開始翻譯」
    """)

if __name__ == "__main__":
    bilingual_srt_translator()
