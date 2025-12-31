import streamlit as st
import os
import tempfile
import gdown
import time
import base64
from io import BytesIO
from pydub import AudioSegment
from openai import OpenAI
import ui_utils

def compress_audio(input_path):
    """Compress audio to MP3 32k mono to fit within API limits."""
    output_path = f"{os.path.splitext(input_path)[0]}_compressed.mp3"
    try:
        audio = AudioSegment.from_file(input_path)
        # Convert to mono and set frame rate to 16kHz (speech standard) to save space
        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(output_path, format="mp3", bitrate="32k")
        return output_path
    except Exception as e:
        raise RuntimeError(f"Audio compression failed: {e}")

def transcribe_audio(file_path, output_path, language, prompt, api_key, temperature):
    """Transcribe audio using OpenAI Whisper."""
    client = OpenAI(api_key=api_key)
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format="srt"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)

def translate_audio(file_path, output_path, prompt, api_key, temperature):
    """Translate audio to English using OpenAI Whisper."""
    client = OpenAI(api_key=api_key)
    with open(file_path, "rb") as audio_file:
        translation = client.audio.translations.create(
            model="whisper-1",
            file=audio_file,
            prompt=prompt,
            temperature=temperature,
            response_format="srt"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(translation)

def ai_subtitle_generator():
    ui_utils.render_header("🚀 AI Subtitle Generator", "Generate subtitles from audio or video files using OpenAI's Whisper model.")

    # Check for API Key
    if not ui_utils.validate_api_inputs(["OpenAI"]):
        st.stop()
    
    api_key = ui_utils.get_api_key("OpenAI")

    # Language Selection
    language_options = {
        'Chinese': 'zh', 'English': 'en', 'Malay': 'ms', 'Japanese': 'ja', 'Korean': 'ko', 
        'German': 'de', 'French': 'fr', 'Spanish': 'es', 'Italian': 'it', 'Russian': 'ru',
        'Thai': 'th', 'Vietnamese': 'vi', 'Indonesian': 'id'
    }
    
    col1, col2 = st.columns(2)
    with col1:
        selected_language = st.selectbox('Source Language', options=list(language_options.keys()))
    with col2:
        translate_to_english = st.checkbox("Translate to English")

    # Advanced Settings
    with st.expander("Advanced Settings"):
        default_prompt = '繁體！'
        user_prompt = st.text_input(
            'Prompt (Optional)',
            default_prompt,
            help='Guiding prompt for style or specific words. Remove default for non-Chinese audio.'
        )
        temperature = st.number_input('Temperature', value=0.2, min_value=0.0, max_value=1.0, step=0.1)

    # File Upload
    uploaded_file = st.file_uploader("Upload MP3/MP4", type=["mp3", "mp4", "m4a", "wav"])
    gdrive_url = st.text_input("Or paste Google Drive Link:")

    # Handle File Logic
    target_file = None
    original_filename = "subtitle"

    if gdrive_url:
        target_file = "gdrive_file"
        try:
            gdown.download(gdrive_url, target_file, quiet=False, fuzzy=True)
            with open(target_file, "rb") as f:
                uploaded_file = BytesIO(f.read())
            original_filename = "gdrive_download"
        except Exception as e:
            st.error(f"Failed to download from Drive: {e}")

    if uploaded_file:
        if not gdrive_url:
            original_filename = os.path.splitext(uploaded_file.name)[0]
        
        if st.button("Generate Subtitles", type="primary"):
            total_start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name if hasattr(uploaded_file, 'name') else 'file')[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_name = temp_file.name

                # Step 1: Compress
                status_text.text("Step 1/2: Compressing audio...")
                progress_bar.progress(20)
                compressed_file = compress_audio(temp_file_name)
                
                # Step 2: Transcribe
                status_text.text("Step 2/2: Transcribing with OpenAI...")
                progress_bar.progress(50)
                
                srt_file = f"{original_filename}.srt"
                
                if translate_to_english:
                    translate_audio(compressed_file, srt_file, user_prompt, api_key, temperature)
                else:
                    transcribe_audio(compressed_file, srt_file, language_options[selected_language], user_prompt, api_key, temperature)
                
                progress_bar.progress(100)
                status_text.success(f"✅ Completed in {time.time() - total_start_time:.2f}s")
                
                # Read and Download
                with open(srt_file, 'r', encoding='utf-8') as f:
                    srt_data = f.read()
                
                st.download_button(
                    label="📥 Download SRT",
                    data=srt_data,
                    file_name=srt_file,
                    mime="text/plain"
                )
                
                # Cleanup
                os.remove(temp_file_name)
                if os.path.exists(compressed_file):
                    os.remove(compressed_file)
                if os.path.exists(srt_file):
                    os.remove(srt_file)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    ai_subtitle_generator()
