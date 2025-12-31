import streamlit as st
import os
import re
import ui_utils

def format_time(time):
    # 將SRT格式的時間轉換為XML格式
    hours, minutes, seconds = time.replace(',', '.').split(':')
    return f"{int(hours):02d}:{int(minutes):02d}:{float(seconds):06.3f}"

def escape_html(text):
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )

def srt_to_xml(srt_content, font_size_1, font_size_2):
    xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<tt xmlns="http://www.w3.org/ns/ttml" xmlns:tts="http://www.w3.org/ns/ttml#styling">
  <head>
    <styling>
      <style xml:id="style1" tts:fontSize="{font_size_1}px" tts:fontFamily="Noto Sans CJK TC" tts:fontWeight="bold" tts:textShadow="2px 2px 2px black" tts:textAlign="center" tts:displayAlign="after"/>
      <style xml:id="style2" tts:fontSize="{font_size_2}px" tts:fontFamily="Noto Sans CJK TC" tts:fontWeight="bold" tts:textShadow="2px 2px 2px black" tts:textAlign="center" tts:displayAlign="after"/>
    </styling>
  </head>
  <body>
    <div>'''

    srt_entries = re.split(r'\n\s*\n', srt_content.strip())
    
    for entry in srt_entries:
        lines = entry.strip().split('\n')
        if len(lines) < 3:
            continue  # 跳過不完整的條目
        
        index = lines[0]
        time_line = lines[1]
        text_lines = lines[2:]
        
        start_time, end_time = map(format_time, time_line.split(' --> '))
        
        for i, line in enumerate(text_lines):
            escaped_line = escape_html(line)
            style_id = "style1" if i == 0 else "style2"
            
            xml_content += f'''
      <p xml:id="caption{index}_{i+1}" begin="{start_time}" end="{end_time}" style="{style_id}">{escaped_line}</p>'''

    xml_content += '''
    </div>
  </body>
</tt>'''

    return xml_content

def bilingual_subtitle_resizer():
    ui_utils.render_header("📏 Subtitle Resizer", "Adjust font sizes for bilingual subtitles and convert to XML.")

    col1, col2 = st.columns(2)
    with col1:
        font_size_1 = st.number_input("Font Size Line 1 (px)", min_value=1, max_value=200, value=71)
    with col2:
        font_size_2 = st.number_input("Font Size Line 2 (px)", min_value=1, max_value=200, value=45)

    uploaded_file = st.file_uploader("Upload SRT File", type="srt")

    if uploaded_file is not None:
        srt_content = uploaded_file.getvalue().decode("utf-8-sig")
        original_filename = uploaded_file.name

        if st.button('Convert to XML', type="primary"):
            xml_content = srt_to_xml(srt_content, font_size_1, font_size_2)
            
            st.subheader("Preview")
            st.text_area("XML Output", xml_content, height=200)

            base_name = os.path.splitext(original_filename)[0]
            new_filename = f"{base_name}_resized.xml"

            st.download_button(
                label="📥 Download XML",
                data=xml_content.encode('utf-8'),
                file_name=new_filename,
                mime="application/xml"
            )

if __name__ == "__main__":
    bilingual_subtitle_resizer()
