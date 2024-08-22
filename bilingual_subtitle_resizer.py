import streamlit as st

def escape_html(unsafe):
    return (unsafe
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#039;"))

def format_time(time):
    return time.replace(',', '.')

def srt_to_xml(srt_content, font_size_1, font_size_2):
    srt_lines = srt_content.strip().split('\n\n')
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/xml" xmlns:tts="http://www.w3.org/ns/xml#styling">
  <head>
    <styling>
      <style xml:id="style1" tts:fontSize="{font_size_1}px" tts:fontFamily="Noto Sans CJK TC" tts:fontWeight="bold" tts:textShadow="2px 2px 2px black" tts:textAlign="center" tts:displayAlign="after"/>
      <style xml:id="style2" tts:fontSize="{font_size_2}px" tts:fontFamily="Noto Sans CJK TC" tts:fontWeight="bold" tts:textShadow="2px 2px 2px black" tts:textAlign="center" tts:displayAlign="after"/>
    </styling>
  </head>
  <body>
    <div>'''

    for srt_line in srt_lines:
        lines = srt_line.split('\n')
        index = lines[0]
        start_time, end_time = map(format_time, lines[1].split(' --> '))

        num_lines = len(lines[2:])
        for i, line in enumerate(lines[2:]):
            escaped_line = escape_html(line)
            if num_lines == 1:
                style_id = "style1"
            else:
                if i == 0 and line.startswith("-"):
                    style_id = "style1"
                else:
                    style_id = "style1" if i == 0 else "style2"

            xml_content += f'''
      <p xml:id="caption{index}_{i+1}" begin="{start_time}" end="{end_time}" style="{style_id}">{escaped_line}</p>'''

    xml_content += '''
    </div>
  </body>
</tt>'''

    return xml_content

def bilingual_subtitle_resizer():
    st.title('雙語字幕大小調整器')

    st.write("這個工具可以幫助您調整雙語字幕的字體大小，並將 SRT 格式轉換為 xml 格式。")

    with st.expander("點擊展開查看詳細說明"):
        st.markdown("""
        ### 應用簡介
        這個工具專為處理雙語字幕而設計，能夠調整不同語言字幕的字體大小，並將常見的 SRT 格式轉換為更靈活的 xml 格式：

        1. **自定義字體大小**：分別為兩種語言的字幕設置不同的字體大小。
        2. **格式轉換**：將 SRT 格式的字幕文件轉換為 xml 格式。

        ### 主要功能
        - 允許用戶自定義兩種語言字幕的字體大小
        - 讀取 SRT 格式的字幕文件
        - 將 SRT 格式轉換為 xml 格式
        - 生成可下載的 xml 文件

        ### 使用步驟
        1. 設置第一語言（通常是主要語言）的字體大小。
        2. 設置第二語言的字體大小。
        3. 上傳 SRT 格式的字幕文件。
        4. 點擊「轉換文件」按鈕。
        5. 預覽生成的 xml 內容。
        6. 下載轉換後的 xml 文件。

        ### 注意事項
        - 確保上傳的 SRT 文件格式正確。
        - xml 格式更適合需要精確控制字幕樣式的場景。
        - 轉換後的 xml 文件使用 UTF-8 編碼，確保與大多數現代系統兼容。
        """)

    col1, col2 = st.columns(2)
    with col1:
        font_size_1 = st.number_input("第一行字體大小（像素）", min_value=1, max_value=100, value=71)
    with col2:
        font_size_2 = st.number_input("第二行字體大小（像素）", min_value=1, max_value=100, value=45)

    uploaded_file = st.file_uploader("選擇一個 SRT 文件", type="srt")

    if uploaded_file is not None:
        srt_content = uploaded_file.getvalue().decode("utf-8")

        if st.button('轉換文件'):
            xml_content = srt_to_xml(srt_content, font_size_1, font_size_2)
            st.text_area("xml 輸出預覽", xml_content, height=300)

            st.download_button(
                label="下載 xml 文件",
                data=xml_content,
                file_name="output.xml",
                mime="application/xml+xml"
            )

if __name__ == "__main__":
    bilingual_subtitle_resizer()