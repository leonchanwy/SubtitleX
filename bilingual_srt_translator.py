import streamlit as st
import re
from openai import OpenAI
import time
import os
import logging
from typing import List, Tuple, Dict
from tenacity import retry, stop_after_attempt, wait_random_exponential

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
LANGUAGE_OPTIONS = ["繁體中文", "英文", "日文", "馬來語", "廣東話口語", "德文"]
MODEL = "gpt-4o-2024-08-06"
MAX_TOKENS = 4000
TEMPERATURE = 0.2
BATCH_SIZE = 20

def init_session_state():
    if 'api_key' not in st.session_state:
        st.session_state.api_key = ''
    if 'translated_srt' not in st.session_state:
        st.session_state.translated_srt = None
    if 'processing_time' not in st.session_state:
        st.session_state.processing_time = None
    if 'original_subtitles' not in st.session_state:
        st.session_state.original_subtitles = None
    if 'translated_subtitles' not in st.session_state:
        st.session_state.translated_subtitles = None
        

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.session_state.api_key)

class SRTTranslator:
    def __init__(self, client):
        self.client = client

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def translate_batch(self, texts: List[str], target_lang1: str, target_lang2: str, prompt1: str, prompt2: str) -> str:
        combined_texts = "\n\n".join(texts)
        system_prompt = self._create_system_prompt(target_lang1, target_lang2, prompt1, prompt2)
        
        try:
            completion = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate:\n\n{combined_texts}"}
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    def _create_system_prompt(self, target_lang1: str, target_lang2: str, prompt1: str, prompt2: str) -> str:
        return f"""You are a highly skilled translator. Translate the following subtitles into both {target_lang1} and {target_lang2}.
        Rules:
        1. Provide direct translations without explanations.
        2. Preserve proper nouns and place names.
        3. Maintain the original tone and nuance.
        4. Use correct grammar and punctuation.
        5. Translate repeated words consistently.
        6. Keep the style informal and colloquial if the original is.
        7. Translate uncertain phrases to the best of your ability.
        8. Format: "Original: [text]\n{target_lang1}: [translation]\n{target_lang2}: [translation]"
        9. Translate each subtitle separately, maintaining order.
        10. Be consistent throughout the dialogue.
        11. For {target_lang1}: {prompt1}
        12. For {target_lang2}: {prompt2}"""

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
                translation = self.translate_batch(texts, target_lang1, target_lang2, prompt1, prompt2)
                parsed_translations = self._parse_translation(translation, target_lang1, target_lang2)
                
                if len(parsed_translations) != len(batch):
                    logger.warning(f"Translation result count does not match original. Batch {i//BATCH_SIZE + 1}")
                    parsed_translations = self._align_translations(parsed_translations, batch)
                
                translated_subtitles.extend(parsed_translations)

                progress = min((i + BATCH_SIZE) / total, 1.0)
                progress_callback(progress)

            except Exception as e:
                logger.error(f"Batch {i//BATCH_SIZE + 1} translation failed: {str(e)}")
                translated_subtitles.extend([{'original': '[Translation Error]', target_lang1: '[Translation Error]', target_lang2: '[Translation Error]'} for _ in batch])

            time.sleep(1)  # Avoid API rate limit

        return translated_subtitles

    def _parse_translation(self, translation: str, target_lang1: str, target_lang2: str) -> List[Dict[str, str]]:
        parsed = []
        current_item = {'original': '', target_lang1: '', target_lang2: ''}
        for line in translation.strip().split('\n'):
            if line.startswith("Original:"):
                if current_item['original'] or current_item[target_lang1] or current_item[target_lang2]:
                    parsed.append(current_item)
                    current_item = {'original': '', target_lang1: '', target_lang2: ''}
                current_item['original'] = line.replace("Original:", "").strip()
            elif line.startswith(f"{target_lang1}:"):
                current_item[target_lang1] = line.replace(f"{target_lang1}:", "").strip()
            elif line.startswith(f"{target_lang2}:"):
                current_item[target_lang2] = line.replace(f"{target_lang2}:", "").strip()
        if current_item['original'] or current_item[target_lang1] or current_item[target_lang2]:
            parsed.append(current_item)
        return parsed

    def _align_translations(self, translations: List[Dict[str, str]], originals: List[Tuple[str, str, str]]) -> List[Dict[str, str]]:
        aligned = []
        for original, translation in zip(originals, translations + [None] * (len(originals) - len(translations))):
            if translation:
                aligned.append(translation)
            else:
                aligned.append({'original': original[2], 'target_lang1': '[Translation Error]', 'target_lang2': '[Translation Error]'})
        return aligned

class SRTProcessor:
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
            raise ValueError("Unable to parse SRT file. Please ensure the file format is correct.")
        return matches

    @staticmethod
    def remove_language_tags(content: str) -> str:
        content = re.sub(r'</?[a-z]>', '', content)  # Remove tags like <i>, </i>
        content = re.sub(r'\{\\an\d\}', '', content)  # Remove tags like {\\an8}
        return content

    @staticmethod
    def format_srt(subtitles: List[Tuple[str, str, str]], translations: List[Dict[str, str]], 
                   format_type: str, lang1: str, lang2: str = None) -> str:
        output = []
        for (number, timestamp, original_text), translation in zip(subtitles, translations):
            output.append(f"{number}\n{timestamp}")
            if format_type == "bilingual":
                original = translation.get('original', original_text)
                translated = translation.get(lang1, '[Translation Error]')
                output.append(f"{original}\n{translated}")
            elif format_type == "dual_lang":
                lang1_text = translation.get(lang1, '[Translation Error]')
                lang2_text = translation.get(lang2, '[Translation Error]')
                output.append(f"{lang1_text}\n{lang2_text}")
            else:  # single language
                output.append(translation.get(lang1, '[Translation Error]'))
            output.append("")  # Add empty line
        return "\n".join(output).strip()

    @staticmethod
    def validate_srt_format(srt_content: str) -> Tuple[bool, str]:
        lines = srt_content.strip().split('\n')
        subtitle_count = sum(1 for line in lines if line.strip().isdigit())
        is_valid = all(re.match(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n.+(\n.+)?(\n\n|$)', block) 
                       for block in re.split(r'\n\s*\n', srt_content.strip()))
        return is_valid, f"SRT file format is {'correct' if is_valid else 'incorrect'}. There are {subtitle_count} subtitles."

def bilingual_srt_translator():
    st.title("Multi-language Subtitle Translator")

    init_session_state()

    st.session_state.api_key = st.text_input("OpenAI API Key", value=st.session_state.api_key, type="password")
    client = get_openai_client()

    target_lang1 = st.selectbox("Target Language 1", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("廣東話口語"))
    target_lang2 = st.selectbox("Target Language 2", options=LANGUAGE_OPTIONS, index=LANGUAGE_OPTIONS.index("英文"))

    prompt1 = st.text_input("Language 1 Translation Style Prompt", value="充滿口語髒話")
    prompt2 = st.text_input("Language 2 Translation Style Prompt", value="充滿黑人rapper口語髒話")

    uploaded_file = st.file_uploader("Choose an SRT file", type="srt")

    if uploaded_file and st.session_state.api_key and st.button("Start Translation"):
        try:
            content = uploaded_file.getvalue().decode("utf-8-sig")
            content = SRTProcessor.remove_language_tags(content)
            subtitles = SRTProcessor.parse_srt(content)
            
            translator = SRTTranslator(client)

            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Translating..."):
                start_time = time.time()
                st.session_state.translated_subtitles = translator.translate_subtitles(
                    subtitles, target_lang1, target_lang2, prompt1, prompt2, progress_bar.progress
                )
                st.session_state.original_subtitles = subtitles
                end_time = time.time()

            st.session_state.processing_time = end_time - start_time
            status_text.success(f"Translation complete! Total processing time: {st.session_state.processing_time:.2f} seconds")

            # Validate the translated SRT
            formatted_srt = SRTProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "bilingual", target_lang1)
            is_valid, validation_message = SRTProcessor.validate_srt_format(formatted_srt)
            
            if not is_valid:
                st.warning("The translated SRT format is invalid. Attempting to correct...")
                # Here you would implement logic to correct the SRT format or request a new translation
                # For now, we'll just show a warning
                st.error("Unable to automatically correct the SRT format. Please check the output manually.")
            
            st.info(validation_message)

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            logger.exception("An exception occurred during translation")

    if st.session_state.translated_subtitles:
        download_option = st.selectbox(
            "Choose download format",
            [f"Original + {target_lang1}", f"Original + {target_lang2}", f"{target_lang1} + {target_lang2}", f"Only {target_lang1}", f"Only {target_lang2}"]
        )

        st.subheader("Translation Preview")
        preview_subtitles = st.session_state.original_subtitles[:5]
        preview_translations = st.session_state.translated_subtitles[:5]
        
        try:
            if "Original" in download_option:
                lang = target_lang1 if target_lang1 in download_option else target_lang2
                preview_srt = SRTProcessor.format_srt(preview_subtitles, preview_translations, "bilingual", lang)
                full_srt = SRTProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "bilingual", lang)
                file_name = f"Original_{lang}.srt"
            elif "+" in download_option:
                preview_srt = SRTProcessor.format_srt(preview_subtitles, preview_translations, "dual_lang", target_lang1, target_lang2)
                full_srt = SRTProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "dual_lang", target_lang1, target_lang2)
                file_name = f"{target_lang1}_{target_lang2}.srt"
            else:
                lang = target_lang1 if target_lang1 in download_option else target_lang2
                preview_srt = SRTProcessor.format_srt(preview_subtitles, preview_translations, "single", lang)
                full_srt = SRTProcessor.format_srt(st.session_state.original_subtitles, st.session_state.translated_subtitles, "single", lang)
                file_name = f"{lang}.srt"
            
            st.text_area("Translation Preview", value=preview_srt + "\n...", height=300)

            st.download_button(
                label=f"Download {download_option} Subtitles",
                data=full_srt,
                file_name=file_name,
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"An error occurred while generating preview or download file: {str(e)}")
            logger.exception("An exception occurred while generating preview or download file")

if __name__ == "__main__":
    bilingual_srt_translator()