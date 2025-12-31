import streamlit as st
import openai
from pathlib import Path
import tempfile
import os
import ui_utils

def transcribe_audio(api_key, audio_file, model, language, prompt, response_format, temperature, timestamp_granularities=None):
    client = openai.OpenAI(api_key=api_key)
    params = {
        "model": model,
        "file": audio_file,
        "language": language if language else None,
        "prompt": prompt if prompt else None,
        "response_format": response_format,
        "temperature": temperature
    }
    if timestamp_granularities:
        params["timestamp_granularities"] = timestamp_granularities
    
    transcript = client.audio.transcriptions.create(**params)
    return transcript.text if response_format == 'json' else transcript

def translate_audio(api_key, audio_file, model, prompt, response_format, temperature):
    client = openai.OpenAI(api_key=api_key)
    translation = client.audio.translations.create(
        model=model,
        file=audio_file,
        prompt=prompt if prompt else None,
        response_format=response_format,
        temperature=temperature
    )
    return translation.text if response_format == 'json' else translation

def text_to_speech(api_key, text, model, voice, response_format, speed):
    client = openai.OpenAI(api_key=api_key)
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,
        speed=speed
    )
    return response.content

def whisper_api_tool():
    ui_utils.render_header("🎙️ Whisper API Tool", "Direct access to OpenAI's audio capabilities: Transcribe, Translate, and TTS.")
    
    if not ui_utils.validate_api_inputs(["OpenAI"]):
        st.stop()
    
    api_key = ui_utils.get_api_key("OpenAI")

    tab1, tab2, tab3 = st.tabs(["Audio Transcription", "Audio Translation", "Text to Speech"])

    # Language List
    languages = [
        ("zh", "Chinese"), ("en", "English"), ("ms", "Malay"), ("ja", "Japanese"), 
        ("de", "German"), ("fr", "French"), ("es", "Spanish"), ("it", "Italian"),
        ("ko", "Korean"), ("ru", "Russian")
    ]

    with tab1:
        st.subheader("Transcribe Audio")
        uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key="transcribe_file_uploader")
        
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Model", ["whisper-1"], key="transcribe_model")
            language = st.selectbox("Language", languages, format_func=lambda x: f"{x[1]} ({x[0]})", key="transcribe_language")
        with col2:
            response_format = st.selectbox("Response Format", ["json", "text", "srt", "verbose_json", "vtt"], key="transcribe_response_format")
            temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, key="transcribe_temperature")
        
        prompt = st.text_area("Prompt (Optional)", key="transcribe_prompt")
        
        timestamp_granularities = None
        if response_format == "verbose_json":
            timestamp_options = st.multiselect(
                "Timestamp Granularities",
                ["word", "segment"],
                default=["segment"],
                help="Word-level timestamps increase latency.",
                key="transcribe_timestamp_granularities"
            )
            if timestamp_options:
                timestamp_granularities = timestamp_options
        
        if uploaded_file is not None and st.button("Transcribe", key="transcribe_button", type="primary"):
            with st.spinner("Transcribing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    with open(tmp_file_path, "rb") as audio_file:
                        transcript = transcribe_audio(api_key, audio_file, model, language[0], prompt, response_format, temperature, timestamp_granularities)
                    st.text_area("Result", transcript, height=250, key="transcribe_result")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(tmp_file_path): os.unlink(tmp_file_path)

    with tab2:
        st.subheader("Translate Audio (to English)")
        uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"], key="translate_file_uploader")
        
        col1, col2 = st.columns(2)
        with col1:
            model = st.selectbox("Model", ["whisper-1"], key="translate_model")
        with col2:
            response_format = st.selectbox("Format", ["json", "text", "srt", "verbose_json", "vtt"], key="translate_response_format")
            
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, key="translate_temperature")
        prompt = st.text_area("Prompt (Optional)", key="translate_prompt")
        
        if uploaded_file is not None and st.button("Translate", key="translate_button", type="primary"):
            with st.spinner("Translating..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    with open(tmp_file_path, "rb") as audio_file:
                        translation = translate_audio(api_key, audio_file, model, prompt, response_format, temperature)
                    st.text_area("Result", translation, height=250, key="translate_result")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    if os.path.exists(tmp_file_path): os.unlink(tmp_file_path)

    with tab3:
        st.subheader("Text to Speech")
        text_input = st.text_area("Input Text", height=100, key="tts_input")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            model = st.selectbox("Model", ["tts-1", "tts-1-hd"], key="tts_model")
        with col2:
            voice = st.selectbox("Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], key="tts_voice")
        with col3:
            response_format = st.selectbox("Format", ["mp3", "opus", "aac", "flac"], key="tts_response_format")
            
        speed = st.slider("Speed", 0.25, 4.0, 1.0, 0.25, key="tts_speed")
        
        if text_input and st.button("Generate Speech", key="tts_button", type="primary"):
            with st.spinner("Generating audio..."):
                try:
                    audio_content = text_to_speech(api_key, text_input, model, voice, response_format, speed)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{response_format}") as tmp_audio_file:
                        tmp_audio_file.write(audio_content)
                        tmp_audio_file_path = tmp_audio_file.name
                    
                    st.audio(tmp_audio_file_path, format=f"audio/{response_format}")
                    
                    st.download_button(
                        label="📥 Download Audio",
                        data=audio_content,
                        file_name=f"speech.{response_format}",
                        mime=f"audio/{response_format}"
                    )
                    
                    if os.path.exists(tmp_audio_file_path): os.unlink(tmp_audio_file_path)
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    whisper_api_tool()
