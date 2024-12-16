import streamlit as st
import fal_client
from pathlib import Path
import tempfile
import os
import json
from datetime import timedelta
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass, asdict
import csv
import yaml
from datetime import datetime

# Constants
MODEL_VERSION = "3"
SUPPORTED_LANGUAGES = [
    ("af", "南非荷蘭語"), ("am", "阿姆哈拉語"), ("ar", "阿拉伯語"), 
    ("as", "阿薩姆語"), ("az", "阿塞拜疆語"), ("ba", "巴什基爾語"), 
    ("be", "白俄羅斯語"), ("bg", "保加利亞語"), ("bn", "孟加拉語"), 
    ("bo", "藏語"), ("br", "布列塔尼語"), ("bs", "波斯尼亞語"), 
    ("ca", "加泰羅尼亞語"), ("cs", "捷克語"), ("cy", "威爾士語"), 
    ("da", "丹麥語"), ("de", "德語"), ("el", "希臘語"), 
    ("en", "英語"), ("es", "西班牙語"), ("et", "愛沙尼亞語"), 
    ("eu", "巴斯克語"), ("fa", "波斯語"), ("fi", "芬蘭語"), 
    ("fo", "法羅語"), ("fr", "法語"), ("gl", "加利西亞語"), 
    ("gu", "古吉拉特語"), ("ha", "豪薩語"), ("haw", "夏威夷語"), 
    ("he", "希伯來語"), ("hi", "印地語"), ("hr", "克羅地亞語"), 
    ("ht", "海地克里奧爾語"), ("hu", "匈牙利語"), ("hy", "亞美尼亞語"), 
    ("id", "印尼語"), ("is", "冰島語"), ("it", "義大利語"), 
    ("ja", "日語"), ("jw", "爪哇語"), ("ka", "格魯吉亞語"), 
    ("kk", "哈薩克語"), ("km", "柬埔寨語"), ("kn", "卡納達語"), 
    ("ko", "韓語"), ("la", "拉丁語"), ("lb", "盧森堡語"), 
    ("ln", "林加拉語"), ("lo", "老撾語"), ("lt", "立陶宛語"), 
    ("lv", "拉脫維亞語"), ("mg", "馬達加斯加語"), ("mi", "毛利語"), 
    ("mk", "馬其頓語"), ("ml", "馬拉雅拉姆語"), ("mn", "蒙古語"), 
    ("mr", "馬拉地語"), ("ms", "馬來語"), ("mt", "馬耳他語"), 
    ("my", "緬甸語"), ("ne", "尼泊爾語"), ("nl", "荷蘭語"), 
    ("nn", "新挪威語"), ("no", "挪威語"), ("oc", "奧克語"), 
    ("pa", "旁遮普語"), ("pl", "波蘭語"), ("ps", "普什圖語"), 
    ("pt", "葡萄牙語"), ("ro", "羅馬尼亞語"), ("ru", "俄語"), 
    ("sa", "梵語"), ("sd", "信德語"), ("si", "僧伽羅語"), 
    ("sk", "斯洛伐克語"), ("sl", "斯洛維尼亞語"), ("sn", "紹納語"), 
    ("so", "索馬里語"), ("sq", "阿爾巴尼亞語"), ("sr", "塞爾維亞語"), 
    ("su", "巽他語"), ("sv", "瑞典語"), ("sw", "斯瓦希里語"), 
    ("ta", "泰米爾語"), ("te", "泰盧固語"), ("tg", "塔吉克語"), 
    ("th", "泰語"), ("tk", "土庫曼語"), ("tl", "他加祿語"), 
    ("tr", "土耳其語"), ("tt", "韃靼語"), ("uk", "烏克蘭語"), 
    ("ur", "烏爾都語"), ("uz", "烏茲別克語"), ("vi", "越南語"), 
    ("yi", "意第緒語"), ("yo", "約魯巴語"), ("yue", "粵語"), 
    ("zh", "中文")
]

TASK_TYPES = [
    ("transcribe", "轉錄"),
    ("translate", "翻譯為英文")
]

CHUNK_LEVELS = [
    ("segment", "分段")
]

SUPPORTED_FORMATS = ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes

@dataclass
class WhisperChunk:
    """FAL Whisper API 的分段數據結構"""
    timestamp: List[float]  # [start_time, end_time]
    text: str
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典格式"""
        return {
            "timestamp": self.timestamp,
            "text": self.text
        }
    
    def to_csv_row(self) -> List[str]:
        """轉換為 CSV 行格式"""
        return [
            f"{self.timestamp[0]:.3f}",
            f"{self.timestamp[1]:.3f}",
            self.text
        ]
    
    def to_srt_entry(self, index: int) -> str:
        """轉換為 SRT 條目格式"""
        start_time = format_time(self.timestamp[0])
        end_time = format_time(self.timestamp[1])
        return f"{index}\n{start_time} --> {end_time}\n{self.text}\n"
    
    def to_vtt_entry(self) -> str:
        """轉換為 VTT 條目格式"""
        start_time = format_time(self.timestamp[0]).replace(',', '.')
        end_time = format_time(self.timestamp[1]).replace(',', '.')
        return f"{start_time} --> {end_time}\n{self.text}\n"
    
    def to_txt_entry(self) -> str:
        """轉換為純文本格式（帶時間戳）"""
        return f"[{self.timestamp[0]:.2f}-{self.timestamp[1]:.2f}] {self.text}"

@dataclass
class WhisperResponse:
    """FAL Whisper API 的完整響應結構"""
    text: str
    chunks: List[WhisperChunk]

def set_fal_api_key(api_key: str) -> bool:
    """設置 FAL API Key"""
    os.environ["FAL_KEY"] = api_key
    return True

def format_time(seconds: float) -> str:
    """將秒數格式化為 SRT 時間格式"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def json_to_srt(json_data: Union[str, Dict[str, Any], WhisperResponse]) -> str:
    """
    將 FAL Whisper API 返回的數據轉換為 SRT 字幕格式
    
    Args:
        json_data: API 返回的數據，可以是:
            - JSON 字符串
            - 字典格式的數據
            - WhisperResponse 對象
    
    Returns:
        str: SRT 格式的字幕文本
    """
    try:
        # 處理不同的輸入類型
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        
        if isinstance(json_data, WhisperResponse):
            chunks = json_data.chunks
        elif isinstance(json_data, dict):
            chunks = [
                WhisperChunk(
                    timestamp=chunk['timestamp'],
                    text=chunk['text']
                )
                for chunk in json_data.get('chunks', [])
            ]
        else:
            return "錯誤: 輸入數據格式不正確"
            
        if not chunks:
            return "錯誤: 找不到分段數據"
            
        # 生成 SRT 格式
        srt_lines = []
        for i, chunk in enumerate(chunks, 1):
            # 格式化時間戳
            start_time = format_time(chunk.timestamp[0])
            end_time = format_time(chunk.timestamp[1])
            
            # 添加 SRT 條目
            text = chunk.text.strip()
            if text:
                srt_lines.append(f"{i}\n{start_time} --> {end_time}\n{text}\n")
        
        # 如果沒有有效的字幕行
        if not srt_lines:
            return "錯誤: 沒有找到有效的字幕數據"
            
        return "\n".join(srt_lines)
        
    except json.JSONDecodeError:
        return "錯誤: JSON 解析失敗"
    except Exception as e:
        return f"錯誤: {str(e)}"

def process_audio(
    audio_file_path: str,
    task: str = "transcribe",
    language: str = "zh",
    chunk_level: str = "segment",
    response_format: str = "srt"
) -> Tuple[str, WhisperResponse]:
    """
    統一的音頻處理函數
    
    Args:
        audio_file_path: 音頻文件路徑
        task: 任務類型 ("transcribe" 或 "translate")
        language: 音頻語言代碼 (默認 "zh")
        chunk_level: 分段級別 (默認 "segment")
        response_format: 輸出格式 (默認 "srt")
        
    Returns:
        Tuple[str, WhisperResponse]: (格式化的輸出文本, API 響應數據)
    """
    try:
        with st.spinner("正在上傳文件..."):
            audio_url = fal_client.upload_file(audio_file_path)
        
        arguments = {
            "audio_url": audio_url,
            "task": task,
            "language": language,
            "chunk_level": chunk_level,
            "version": MODEL_VERSION
        }
        
        task_name = "轉錄" if task == "transcribe" else "翻譯"
        with st.spinner(f"提交{task_name}請求..."):
            handle = fal_client.submit(
                "fal-ai/wizper",
                arguments=arguments
            )
            
        with st.spinner(f"正在處理音頻..."):
            for status in handle.iter_events(with_logs=True, interval=0.5):
                if isinstance(status, fal_client.Queued):
                    st.write(f"排隊中... 位置: {status.position}")
                elif isinstance(status, fal_client.InProgress):
                    if status.logs:
                        for log in status.logs:
                            st.write(log["message"])
                elif isinstance(status, fal_client.Completed):
                    break
            
            result = handle.get()
            response = WhisperResponse(
                text=result.get("text", ""),
                chunks=[
                    WhisperChunk(
                        timestamp=chunk['timestamp'],
                        text=chunk['text']
                    )
                    for chunk in result.get("chunks", [])
                ]
            )
            
            if response_format == "text":
                return response.text, response
            elif response_format == "srt":
                return json_to_srt(response), response
            elif response_format == "json":
                return json.dumps(result, ensure_ascii=False, indent=2), response
            return response.text, response
            
    except Exception as e:
        raise Exception(f"處理失敗: {str(e)}")

def display_file_info(file_size: int) -> None:
    """顯示文件信息"""
    st.info(f"""
    文件信息:
    - 大小: {file_size / 1024 / 1024:.2f} MB
    - 最大限制: {MAX_FILE_SIZE / 1024 / 1024} MB
    """)

def export_chunks(chunks: List[WhisperChunk], format: str, filename: str = None) -> str:
    """
    將字幕數據導出為指定格式
    
    Args:
        chunks: WhisperChunk 列表
        format: 輸出格式 ('srt', 'vtt', 'csv', 'json', 'yaml', 'txt')
        filename: 可選的文件名（不含擴展名）
    
    Returns:
        str: 格式化的輸出內容
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}"
    
    if format == "srt":
        content = "\n".join(chunk.to_srt_entry(i) for i, chunk in enumerate(chunks, 1))
        file_ext = "srt"
    
    elif format == "vtt":
        content = "WEBVTT\n\n" + "\n".join(chunk.to_vtt_entry() for chunk in chunks)
        file_ext = "vtt"
    
    elif format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Start Time", "End Time", "Text"])
        writer.writerows(chunk.to_csv_row() for chunk in chunks)
        content = output.getvalue()
        file_ext = "csv"
    
    elif format == "json":
        content = json.dumps(
            [chunk.to_dict() for chunk in chunks],
            ensure_ascii=False,
            indent=2
        )
        file_ext = "json"
    
    elif format == "yaml":
        content = yaml.dump(
            [chunk.to_dict() for chunk in chunks],
            allow_unicode=True,
            sort_keys=False
        )
        file_ext = "yaml"
    
    elif format == "txt":
        content = "\n".join(chunk.to_txt_entry() for chunk in chunks)
        file_ext = "txt"
    
    else:
        raise ValueError(f"不支持的輸出格式: {format}")
    
    return content, f"{filename}.{file_ext}"

def display_result(content: str, response: WhisperResponse, format: str):
    """顯示處理結果並提供下載選項"""
    st.subheader("處理結果")
    
    # 顯示文本結果
    st.text_area("輸出內容", content, height=250)
    
    # 生成下載文件
    output_content, filename = export_chunks(response.chunks, format)
    
    # 提供下載按鈕
    st.download_button(
        label=f"下載 {format.upper()} 文件",
        data=output_content,
        file_name=filename,
        mime=get_mime_type(format)
    )
    
    # 顯示統計信息
    st.subheader("統計信息")
    stats = {
        "總字幕數": len(response.chunks),
        "總時長": f"{response.chunks[-1].timestamp[1]:.2f} 秒",
        "平均分段長度": f"{sum(len(c.text) for c in response.chunks)/len(response.chunks):.1f} 字"
    }
    for key, value in stats.items():
        st.text(f"{key}: {value}")

def get_mime_type(format: str) -> str:
    """獲取對應格式的 MIME 類型"""
    mime_types = {
        "srt": "text/plain",
        "vtt": "text/vtt",
        "csv": "text/csv",
        "json": "application/json",
        "yaml": "application/x-yaml",
        "txt": "text/plain"
    }
    return mime_types.get(format, "text/plain")

def whisper_api_tool():
    """主應用函數"""
    st.title("🎤 FAL Whisper API Tool")
    
    # 添加版本信息
    st.markdown("""
    ### 功能特點
    - 支持 Whisper large-v3 模型
    - 支持超過 90 種語言的轉錄
    - 支持翻譯為英文
    - 支持多種輸出格式 (文本/SRT/JSON)
    - 實時處理進度顯示
    """)
    
    # API Key 設置
    api_key = st.text_input("輸入您的 FAL API Key", value=st.session_state.get('api_key', ''), type="password")
    
    if not api_key:
        st.warning("請輸入您的 FAL API Key")
        return
        
    try:
        set_fal_api_key(api_key)
        st.session_state.api_key = api_key
        st.success("API Key 已設置")
    except Exception as e:
        st.error(f"API Key 設置失敗: {str(e)}")
        return

    # 主要功能區
    tab1, tab2 = st.tabs(["音頻處理", "高級設置"])

    with tab1:
        st.header("音頻處理")
        
        # 文件上傳
        uploaded_file = st.file_uploader(
            "選擇音頻文件", 
            type=SUPPORTED_FORMATS
        )
        
        # 任務選擇
        task = st.selectbox(
            "任務類型",
            TASK_TYPES,
            format_func=lambda x: x[1]
        )[0]
        
        # 語言選擇 - 設置預設值為中文
        default_lang_index = next(
            (i for i, lang in enumerate(SUPPORTED_LANGUAGES) if lang[0] == "zh"),
            0
        )
        language = st.selectbox(
            "音頻語言",
            SUPPORTED_LANGUAGES,
            index=default_lang_index,
            format_func=lambda x: f"{x[1]} ({x[0]})"
        )[0]
        
        # 輸出格式選擇
        OUTPUT_FORMATS = [
            ("srt", "SRT 字幕"),
            ("vtt", "VTT 字幕"),
            ("txt", "純文本 (帶時間戳)"),
            ("csv", "CSV 表格"),
            ("json", "JSON 格式"),
            ("yaml", "YAML 格式")
        ]
        response_format = st.selectbox(
            "輸出格式",
            OUTPUT_FORMATS,
            format_func=lambda x: x[1]
        )[0]

        if uploaded_file is not None:
            display_file_info(uploaded_file.size)
            
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"文件大小超過限制 ({MAX_FILE_SIZE / 1024 / 1024}MB)")
            elif st.button("開始處理", key="process_button"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    result_text, result_json = process_audio(
                        tmp_file_path,
                        task=task,
                        language=language,
                        response_format=response_format
                    )

                    os.unlink(tmp_file_path)
                    
                    display_result(result_text, result_json, response_format)
                    
                except Exception as e:
                    st.error(f"處理過程中發生錯誤: {str(e)}")

    with tab2:
        st.header("高級設置")
        chunk_level = st.selectbox(
            "分段級別",
            CHUNK_LEVELS,
            format_func=lambda x: x[1]
        )[0]
        
        st.info(f"""
        當前設置:
        - 模型版本: {MODEL_VERSION}
        - 支持的文件格式: {", ".join(SUPPORTED_FORMATS)}
        - 文件大小限制: {MAX_FILE_SIZE / 1024 / 1024} MB
        """)

if __name__ == "__main__":
    whisper_api_tool()

