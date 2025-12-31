import streamlit as st
import xml.etree.ElementTree as ET
import datetime
import re
from datetime import timedelta
import os
import tempfile
import ui_utils

# Constants
DEFAULT_MAX_DIFFERENCE = 0.5
DEFAULT_FRAME_RATE = 29.97

class XMLParser:
    @staticmethod
    def parse_xml(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        frame_rate = XMLParser._detect_frame_rate(root)
        cut_points = XMLParser._get_cut_points(root)
        return frame_rate, cut_points

    @staticmethod
    def _detect_frame_rate(root):
        rate_elements = root.findall(".//rate")
        for rate_element in rate_elements:
            timebase = rate_element.find("timebase")
            ntsc = rate_element.find("ntsc")
            if timebase is not None and timebase.text:
                try:
                    frame_rate = float(timebase.text)
                    if ntsc is not None and ntsc.text.lower() == "true":
                        frame_rate = frame_rate * 1000 / 1001  # NTSC adjustment
                    return frame_rate
                except ValueError:
                    continue
        return DEFAULT_FRAME_RATE

    @staticmethod
    def _get_cut_points(root):
        video_tracks = root.findall(".//video/track") + root.findall(".//video")
        cut_points = []
        for track in video_tracks:
            clipitems = track.findall(".//clipitem") + track.findall("clipitem")
            for clipitem in clipitems:
                start_element = clipitem.find("start")
                end_element = clipitem.find("end")
                if start_element is not None and start_element.text and start_element.text != "0":
                    cut_points.append(int(start_element.text))
                if end_element is not None and end_element.text and end_element.text != "-1":
                    cut_points.append(int(end_element.text))
        return cut_points

class SRTParser:
    @staticmethod
    def parse_srt(srt_path):
        with open(srt_path, "r", encoding="utf-8") as file:
            content = file.read()
        pattern = re.compile(r"(\d+\:\d+\:\d+,\d+) --> (\d+\:\d+\:\d+,\d+)\n(.*?)(?=\n\n|\Z)", re.DOTALL)
        return [(SRTParser._srt_time_to_timedelta(start), SRTParser._srt_time_to_timedelta(end), text.strip())
                for start, end, text in pattern.findall(content)]

    @staticmethod
    def _srt_time_to_timedelta(srt_time_str):
        hours, minutes, seconds = srt_time_str.split(':')
        seconds, milliseconds = seconds.split(',')
        return timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), milliseconds=int(milliseconds))

class SubtitleAdjuster:
    def __init__(self, frame_rate, cut_points, max_difference_seconds):
        self.frame_rate = frame_rate
        self.cut_points = [self._frame_to_timedelta(frame) for frame in cut_points]
        self.max_difference_seconds = max_difference_seconds

    def _frame_to_timedelta(self, frame):
        return timedelta(seconds=frame / self.frame_rate)

    def adjust_subtitles(self, srt_data):
        adjusted_srt_data = []
        for start_time, end_time, text in srt_data:
            new_start_time = self._get_closest_cut_time(start_time) or start_time
            new_end_time = self._get_closest_cut_time(end_time) or end_time
            adjusted_srt_data.append((new_start_time, new_end_time, text))
        return adjusted_srt_data

    def _get_closest_cut_time(self, time):
        min_difference = timedelta(hours=9999)
        closest_cut_time = None
        for cut_time in self.cut_points:
            difference = abs(cut_time - time)
            if difference < min_difference:
                min_difference = difference
                closest_cut_time = cut_time
        return closest_cut_time if min_difference <= timedelta(seconds=self.max_difference_seconds) else None

class SRTWriter:
    @staticmethod
    def write_srt(srt_data):
        srt_string = ""
        for i, (start_time, end_time, text) in enumerate(srt_data, start=1):
            srt_string += f"{i}\n"
            srt_string += f"{SRTWriter._timedelta_to_srt_time(start_time)} --> {SRTWriter._timedelta_to_srt_time(end_time)}\n"
            srt_string += f"{text}\n\n"
        return srt_string.rstrip()

    @staticmethod
    def _timedelta_to_srt_time(timedelta_obj):
        total_seconds = int(timedelta_obj.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = timedelta_obj.microseconds // 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def process_files(xml_path, srt_path, max_difference_seconds):
    frame_rate, cut_points = XMLParser.parse_xml(xml_path)
    srt_data = SRTParser.parse_srt(srt_path)
    adjuster = SubtitleAdjuster(frame_rate, cut_points, max_difference_seconds)
    adjusted_srt_data = adjuster.adjust_subtitles(srt_data)
    return SRTWriter.write_srt(adjusted_srt_data), frame_rate

def subtitle_time_sync():
    ui_utils.render_header("⏱️ Subtitle Time Sync", "Sync your SRT subtitles with video edit points using XML export data.")

    with st.expander("ℹ️ How it works"):
        st.markdown(r"""
        **Automated Synchronization:**
        1. Export an XML file of your video timeline from Premiere Pro or Final Cut.
        2. Upload that XML file along with your out-of-sync SRT.
        3. This tool will align subtitle start/end times to the nearest video cut points.
        
        **Supported Formats:**
        - XML: Final Cut Pro XML, Premiere Pro XML
        - Subtitles: Standard SRT
        """)

    col1, col2 = st.columns(2)
    with col1:
        xml_file = st.file_uploader("Upload Video XML", type="xml")
    with col2:
        srt_file = st.file_uploader("Upload SRT File", type="srt")

    if xml_file and srt_file:
        max_difference_seconds = st.slider(
            "Max Time Difference (Seconds)", 0.1, 2.0, DEFAULT_MAX_DIFFERENCE, 0.1,
            help="Maximum allowed gap between subtitle timestamp and video cut to trigger synchronization."
        )

        if st.button("Start Synchronization", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_xml:
                tmp_xml.write(xml_file.getvalue())
                tmp_xml_path = tmp_xml.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".srt") as tmp_srt:
                tmp_srt.write(srt_file.getvalue())
                tmp_srt_path = tmp_srt.name

            try:
                adjusted_content, detected_frame_rate = process_files(
                    tmp_xml_path, tmp_srt_path, max_difference_seconds)
                st.success(f"✅ Sync Complete! Detected Frame Rate: {detected_frame_rate:.2f}")
                
                original_filename = os.path.splitext(srt_file.name)[0]
                new_filename = f"{original_filename}_synced.srt"
                
                st.download_button(
                    label="📥 Download Synced SRT",
                    data=adjusted_content,
                    file_name=new_filename,
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                if os.path.exists(tmp_xml_path): os.unlink(tmp_xml_path)
                if os.path.exists(tmp_srt_path): os.unlink(tmp_srt_path)

if __name__ == "__main__":
    subtitle_time_sync()
