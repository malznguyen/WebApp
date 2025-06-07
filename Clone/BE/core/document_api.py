import logging
import os
import json
import PyPDF2
import docx2txt
from openai import OpenAI
import requests
import time
import re
from collections import Counter
import concurrent.futures 
from typing import List, Dict, Any, Optional, Tuple
from typing import Union, Tuple

logger = logging.getLogger('ImageSearchApp')  

# --- Constants for API and Chunk Processing ---
MAX_API_WORKERS = 3 
MAX_CHUNK_WORKERS = 3
MAX_PROMPT_TEXT_LENGTH = 110000
DOCUMENT_SEPARATOR = "\n\n--- DOCUMENT BREAK ---\n\n" # Separator for concatenated texts

class TextAnalysis: 
    @staticmethod
    def count_words(text: str) -> Union[int, str]:
        try:
            if not isinstance(text, str): return 0
            words = re.findall(r'\b\w+\b', text.lower())
            return len(words)
        except Exception as e:
            logger.error(f"Error counting words: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def count_sentences(text: str) -> Union[int, str]:
        try:
            if not isinstance(text, str): return 0
            # Improved sentence splitting regex
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s', text)
            sentences = [s for s in sentences if s and s.strip()]
            return len(sentences) if sentences else (1 if text.strip() else 0)
        except Exception as e:
            logger.error(f"Error counting sentences: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def count_paragraphs(text: str) -> Union[int, str]:
        try:
            if not isinstance(text, str): return 0
            paragraphs = re.split(r'\n\s*\n', text)
            paragraphs = [p for p in paragraphs if p and p.strip()]
            return len(paragraphs) if paragraphs else (1 if text.strip() else 0)
        except Exception as e:
            logger.error(f"Error counting paragraphs: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def average_word_length(text: str) -> Union[float, str]:
        try:
            if not isinstance(text, str): return 0.0
            words = re.findall(r'\b\w+\b', text.lower())
            if not words: return 0.0
            return sum(len(word) for word in words) / len(words)
        except Exception as e:
            logger.error(f"Error calculating average word length: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def average_sentence_length(text: str) -> Union[float, str]:
        try:
            if not isinstance(text, str): return 0.0
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s', text)
            sentences = [s for s in sentences if s and s.strip()]
            if not sentences: return 0.0
            total_words = sum(len(re.findall(r'\b\w+\b', s.lower())) for s in sentences)
            return total_words / len(sentences)
        except Exception as e:
            logger.error(f"Error calculating average sentence length: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def most_common_words(text: str, top_n: int = 50) -> Union[List[Tuple[str, int]], list]:
        try:
            if not isinstance(text, str): return []
            # Stop words list can be expanded or moved to a constants file
            stop_words = {
                'và', 'là', 'có', 'của', 'ở', 'tại', 'trong', 'trên', 'dưới', 'cho', 'đến', 'với', 'bởi', 'qua', 'về',
                'như', 'mà', 'thì', 'rằng', 'nhưng', 'nếu', 'hay', 'hoặc', 'khi', 'lúc', 'sau', 'trước', 'từ', 'để',
                'không', 'chưa', 'được', 'bị', 'phải', 'cần', 'nên', 'cũng', 'vẫn', 'chỉ', 'ngay', 'luôn', 'rất',
                'quá', 'hơn', 'kém', 'nhất', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'to', 'in', 'it', 'and', 'for', 'with', 'on', 'at', 'by',
                'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
                'fig', 'figure', 'table', 'section', 'chapter', 'page', 'appendix', 'index', 'image', 'example', 'et', 'al'
            }
            # Regex for words, including common accented characters
            words = re.findall(r"\b[a-zA-Zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳýỵỷỹ]+(?:[-'\u2019][a-zA-Zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳýỵỷỹ]+)*\b", text.lower())
            filtered_words = [word for word in words if word not in stop_words and len(word) > 2 and not word.isdigit()]
            if not filtered_words: return []
            return Counter(filtered_words).most_common(top_n)
        except Exception as e:
            logger.error(f"Error finding common words: {e}", exc_info=True)
            return []

def extract_text(file_path: str) -> str:
    """Extracts text from PDF, DOCX, TXT, or MD files."""
    if not file_path or not isinstance(file_path, str):
        raise ValueError("Invalid file path provided for text extraction.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.isfile(file_path): # Ensure it's a file, not a directory
        raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")
    if not os.access(file_path, os.R_OK): # Check read permissions
        raise PermissionError(f"No read permission for file: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    logger.info(f"Attempting to extract text from: '{file_name}' (format: {file_extension})")
    extracted_text = ""

    try:
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file, strict=False) # strict=False for better handling of malformed PDFs
                if pdf_reader.is_encrypted:
                    logger.warning(f"PDF '{file_name}' is encrypted. Text extraction may fail or be incomplete.")
                    # Attempt to decrypt with an empty password, common for some "protected" PDFs
                    try:
                        if pdf_reader.decrypt('') == PyPDF2.PasswordType.OWNER_PASSWORD:
                             logger.info(f"Successfully decrypted PDF '{file_name}' with empty owner password.")
                        elif pdf_reader.decrypt('') == PyPDF2.PasswordType.USER_PASSWORD:
                            logger.info(f"Successfully decrypted PDF '{file_name}' with empty user password.")
                        else: # Fails if decryption is not successful with empty password
                            raise ValueError(f"Encrypted PDF '{file_name}' requires a password.")
                    except Exception as decrypt_err: # Catch decryption specific errors
                         logger.error(f"Failed to decrypt PDF '{file_name}': {decrypt_err}")
                         raise ValueError(f"Encrypted PDF '{file_name}' requires a password or is too heavily encrypted.")


                num_pages = len(pdf_reader.pages)
                logger.debug(f"Reading {num_pages} pages from PDF '{file_name}'...")
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text: extracted_text += page_text + "\n"
                    except Exception as page_err:
                        logger.warning(f"Error reading page {page_num + 1} of PDF '{file_name}': {page_err}. Skipping page.")
                        extracted_text += f"\n[[Error reading page {page_num + 1}]]\n"
            logger.info(f"Extracted {len(extracted_text)} characters from PDF '{file_name}'.")

        elif file_extension == '.docx':
            extracted_text = docx2txt.process(file_path)
            logger.info(f"Extracted {len(extracted_text)} characters from DOCX '{file_name}'.")

        elif file_extension in ['.txt', '.md']:
            encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            text_read_success = False
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        extracted_text = file.read()
                    logger.info(f"Extracted {len(extracted_text)} characters from {file_extension.upper()} '{file_name}' (encoding: {enc}).")
                    text_read_success = True
                    break 
                except UnicodeDecodeError:
                    logger.debug(f"Failed to decode '{file_name}' with encoding {enc}, trying next...")
                except Exception as txt_err: # Catch other read errors
                    logger.error(f"Error reading text file '{file_name}' with encoding {enc}: {txt_err}", exc_info=True)
                    # Don't break, try other encodings
            if not text_read_success:
                raise IOError(f"Could not read file '{file_name}' with any of the attempted encodings.")
        else:
            raise ValueError(f"Unsupported file format: {file_extension} for file '{file_name}'")
        
        return extracted_text.strip() if extracted_text else ""

    except PyPDF2.errors.PdfReadError as pdf_err: # Specific PDF library error
        logger.error(f"PyPDF2 could not read PDF '{file_name}': {pdf_err}", exc_info=True)
        raise ValueError(f"Invalid or corrupted PDF file: {file_name}. Error: {pdf_err}")
    except Exception as e: # Catch-all for other errors during extraction process
        logger.error(f"Unexpected error during text extraction from '{file_name}': {e}", exc_info=True)
        # Re-raise as a more generic RuntimeError or a custom ExtractionError
        raise RuntimeError(f"Failed to extract text from '{file_name}': {e}")


def _get_language_full_name(lang_code: Optional[str]) -> str: 
    if not lang_code: return "original language of the document" # Default if no code
    lang_map = {
        'vi': 'Vietnamese', 'en': 'English', 'fr': 'French', 'de': 'German', 
        'es': 'Spanish', 'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch',
        'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese (Simplified)', 
        'ru': 'Russian', 'ar': 'Arabic', 'hi': 'Hindi'
    }
    return lang_map.get(lang_code.lower(), lang_code) # Fallback to code itself if not in map


def _create_summary_prompt(text_to_summarize: str, summary_level: int, language_instruction: str, 
                           is_synthesis_prompt: bool = False, is_batch_synthesis: bool = False) -> Tuple[str, str]:
    """Creates the user prompt and system role content for AI summarization/synthesis."""
    trimmed_text = text_to_summarize[:MAX_PROMPT_TEXT_LENGTH]
    if len(text_to_summarize) > MAX_PROMPT_TEXT_LENGTH:
        logger.warning(f"Prompt text was trimmed from {len(text_to_summarize)} to {MAX_PROMPT_TEXT_LENGTH} characters due to length limits.")

    lang_instr_str = language_instruction if language_instruction and isinstance(language_instruction, str) else ""
    if lang_instr_str and not lang_instr_str.strip().endswith("."): # Ensure it ends well if added
        lang_instr_str = lang_instr_str.strip() + ". " 
    else:
        lang_instr_str = lang_instr_str.strip() + " " if lang_instr_str.strip() else ""


    if is_batch_synthesis: # Specific prompt for synthesizing multiple distinct documents
        task_description = (
            f"Your task is to synthesize key information from the following collection of **distinct documents**. "
            f"Provide a comprehensive overview that integrates the main points, arguments, and findings from **all provided documents**. "
            f"The desired detail level for this synthesis is approximately {summary_level}%. {lang_instr_str}"
            f"The documents are separated by '{DOCUMENT_SEPARATOR.strip()}'. "
            f"Synthesize the following documents:\n--- START DOCUMENT COLLECTION ---\n{trimmed_text}\n--- END DOCUMENT COLLECTION ---"
        )
        system_role_content = (
            "You are an expert AI assistant specializing in synthesizing information from collections of multiple, distinct text sources. "
            "Your goal is to create a single, coherent overview. You must strictly adhere to any specified output language."
        )
    elif is_synthesis_prompt: # For synthesizing chunks of a single document or combined texts not explicitly marked as batch
        task_description = (
            f"Synthesize the key information from the following combined text. This text may be derived from multiple sections of a single document or related sources. "
            f"Provide a comprehensive overview that integrates the main points, arguments, and findings. "
            f"The desired detail level for this synthesis is {summary_level}%. {lang_instr_str}"
            f"Combined text to synthesize:\n--- START COMBINED TEXT ---\n{trimmed_text}\n--- END COMBINED TEXT ---"
        )
        system_role_content = (
            "You are an expert AI assistant specializing in synthesizing information from provided text segments. "
            "You must strictly adhere to any specified output language."
        )
    else: # Standard summary of a single document (or chunk)
        task_description = (
            f"Create a summary for the following document with a detail level of approximately {summary_level}%. {lang_instr_str}"
            f"Document to summarize:\n--- START DOCUMENT ---\n{trimmed_text}\n--- END DOCUMENT ---"
        )
        system_role_content = (
            "You are a professional AI assistant specializing in summarizing documents clearly and concisely. "
            "You must strictly adhere to any specified output language."
        )
    return task_description, system_role_content


def _handle_api_error(api_name: str, exception: Exception) -> str: 
    """Standardized error message generation for API call failures."""
    logger.error(f"Error using {api_name} API: {str(exception)}", exc_info=True)
    error_message = str(exception).lower()

    if isinstance(exception, requests.exceptions.Timeout):
        return f"{api_name} Error: Request timed out after configured limit."
    if isinstance(exception, requests.exceptions.ConnectionError):
        return f"{api_name} Error: Network connection issue. Please check your internet."
    if isinstance(exception, requests.exceptions.RequestException): # Generic requests error
         return f"{api_name} Error: A request-related error occurred ({type(exception).__name__})."

    # OpenAI specific error handling (also applicable to DeepSeek/Grok if they use similar structures)
    if hasattr(exception, 'status_code'): # Likely an openai.APIError or similar
        status_code = getattr(exception, 'status_code')
        api_error_message = getattr(exception, 'message', str(exception)) # Get detailed message if available

        if status_code == 401: return f"{api_name} Error (401): Invalid API key or authentication failure."
        if status_code == 429: return f"{api_name} Error (429): Rate limit or quota exceeded. Please try again later or check your plan."
        if status_code == 400:
            if "context_length_exceeded" in error_message or "input_too_long" in error_message:
                 return f"{api_name} Error (400): Input text is too long for the model's context window."
            return f"{api_name} Error (400): Bad request. Details: {api_error_message}"
        if status_code in [500, 502, 503, 504]:
             return f"{api_name} Error ({status_code}): Temporary server-side issue. Please try again later."
        return f"{api_name} Error ({status_code}): API returned an error. Details: {api_error_message}"

    # Fallback for general Python exceptions or errors not fitting above patterns
    if "authentication" in error_message or "incorrect api key" in error_message:
        return f"{api_name} Error: Authentication issue (possibly invalid API key)."
    if "rate limit" in error_message or "quota" in error_message:
        return f"{api_name} Error: API limit or quota likely exceeded."
    if "context_length_exceeded" in error_message:
        return f"{api_name} Error: Input text too long for model processing."

    return f"{api_name} API Error: An unexpected issue occurred ({type(exception).__name__}). Details: {str(exception)}"


def _call_ai_model(client_config: Dict[str, Any], model_name: str, messages: List[Dict[str, str]], 
                   max_tokens: int, temperature: float, timeout: int) -> str:
    """Helper to make the actual API call, abstracting client creation."""
    client = OpenAI(api_key=client_config['api_key'], base_url=client_config.get('base_url'))
    response = client.with_options(timeout=timeout).chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    if response and response.choices and response.choices[0].message and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    else:
        logger.error(f"AI model ({model_name}) API returned unexpected or empty structure: {response}")
        raise ValueError("Invalid or empty API response structure from AI model.")


def summarize_with_deepseek(text: str, api_key: str, summary_level: int = 50, target_language_code: Optional[str] = None, 
                            is_synthesis_prompt: bool = False, is_batch_synthesis: bool = False) -> str:
    if not api_key: return "DeepSeek Error: API Key not configured."
    if not text or not text.strip(): return "DeepSeek Error: Input text for summarization is empty."
    from config.constants import API_TIMEOUT_SEC 

    target_lang_full = _get_language_full_name(target_language_code)
    lang_instruction = f"IMPORTANT: Please provide the output ONLY in {target_lang_full}." if target_language_code else ""
    system_role_lang_enforce = f" You MUST provide the output exclusively in {target_lang_full}." if target_language_code else ""

    prompt_text, system_content_base = _create_summary_prompt(text, summary_level, lang_instruction, is_synthesis_prompt, is_batch_synthesis)
    final_system_content = f"{system_content_base}{system_role_lang_enforce}"

    logger.debug(f"Calling DeepSeek API ({'BATCH_SYNTHESIS' if is_batch_synthesis else ('SYNTHESIS' if is_synthesis_prompt else 'SUMMARY')}, Lang: {target_lang_full})...")
    try:
        summary = _call_ai_model(
            client_config={'api_key': api_key, 'base_url': "https://api.deepseek.com"},
            model_name="deepseek-chat",
            messages=[{"role": "system", "content": final_system_content}, {"role": "user", "content": prompt_text}],
            max_tokens=8048, temperature=0.5, timeout=API_TIMEOUT_SEC
        )
        logger.info(f"DeepSeek returned content ({target_lang_full}), {len(summary)} chars.")
        return summary
    except Exception as e:
        return _handle_api_error("DeepSeek", e)


def summarize_with_grok(text: str, api_key: str, summary_level: int = 50, target_language_code: Optional[str] = None, 
                        is_synthesis_prompt: bool = False, is_batch_synthesis: bool = False) -> str:
    if not api_key: return "Grok Error: API Key not configured."
    if not text or not text.strip(): return "Grok Error: Input text for summarization is empty."
    from config.constants import API_TIMEOUT_SEC

    target_lang_full = _get_language_full_name(target_language_code)
    lang_instruction = f"\nVERY IMPORTANT: The final output MUST be in {target_lang_full}. No other languages are acceptable." if target_language_code else ""
    system_role_lang_enforce = f" Always respond ONLY in {target_lang_full}. Strict adherence is mandatory." if target_language_code else ""
    
    prompt_text, system_content_base = _create_summary_prompt(text, summary_level, lang_instruction, is_synthesis_prompt, is_batch_synthesis)
    # Grok might benefit from a more direct system prompt, especially for synthesis
    if is_batch_synthesis or is_synthesis_prompt:
        final_system_content = f"You are an AI that synthesizes information from provided text sources.{system_role_lang_enforce}"
    else:
        final_system_content = f"{system_content_base}{system_role_lang_enforce}"

    logger.debug(f"Calling Grok API ({'BATCH_SYNTHESIS' if is_batch_synthesis else ('SYNTHESIS' if is_synthesis_prompt else 'SUMMARY')}, Lang: {target_lang_full})...")
    try:
        summary = _call_ai_model(
            client_config={'api_key': api_key, 'base_url': "https://api.x.ai/v1"},
            model_name="grok-2", 
            messages=[{"role": "system", "content": final_system_content}, {"role": "user", "content": prompt_text}],
            max_tokens=8048, temperature=0.5, timeout=API_TIMEOUT_SEC
        )
        logger.info(f"Grok returned content ({target_lang_full}), {len(summary)} chars.")
        return summary
    except Exception as e:
        return _handle_api_error("Grok", e)


def summarize_with_chatgpt(text: str, api_key: str, summary_level: int = 50, target_language_code: Optional[str] = None, 
                           is_synthesis_prompt: bool = False, is_batch_synthesis: bool = False) -> str:
    if not api_key: return "ChatGPT Error: API Key not configured."
    if not text or not text.strip(): return "ChatGPT Error: Input text for summarization is empty."
    from config.constants import API_TIMEOUT_SEC

    target_lang_full = _get_language_full_name(target_language_code)
    lang_instruction = f"CRITICAL REQUIREMENT: Provide the final output ONLY in {target_lang_full}. Adherence is paramount." if target_language_code else ""
    system_role_lang_enforce = f" You must respond exclusively in {target_lang_full}. No other language is permitted." if target_language_code else ""

    prompt_text, system_content_base = _create_summary_prompt(text, summary_level, lang_instruction, is_synthesis_prompt, is_batch_synthesis)
    final_system_content = f"{system_content_base}{system_role_lang_enforce}"

    logger.debug(f"Calling ChatGPT API ({'BATCH_SYNTHESIS' if is_batch_synthesis else ('SYNTHESIS' if is_synthesis_prompt else 'SUMMARY')}, Lang: {target_lang_full})...")
    try:
        summary = _call_ai_model(
            client_config={'api_key': api_key},
            model_name="gpt-4o", 
            messages=[{"role": "system", "content": final_system_content}, {"role": "user", "content": prompt_text}],
            max_tokens=4096, temperature=0.6, timeout=API_TIMEOUT_SEC
        )
        logger.info(f"ChatGPT returned content ({target_lang_full}), {len(summary)} chars.")
        return summary
    except Exception as e:
        return _handle_api_error("ChatGPT", e)


def _process_chunk_task(api_func, chunk_text: str, api_key_val: str, summary_level_val: int, 
                        target_lang: Optional[str], for_synthesis: bool, is_batch_synth_flag: bool, 
                        api_name_str: str, chunk_idx: int) -> str:
    """Helper function to run each chunk summarization/synthesis in a thread."""
    target_language_full_name = _get_language_full_name(target_lang)
    log_task_type = "BATCH_SYNTH_CHUNK" if is_batch_synth_flag else ("SYNTH_CHUNK" if for_synthesis else "SUMM_CHUNK")
    logger.info(f"Starting chunk {chunk_idx+1} ({log_task_type}) with {api_name_str} for language: {target_language_full_name}...")
    try:
        # Pass is_batch_synthesis flag to the AI function
        summary_part = api_func(chunk_text, api_key_val, summary_level_val, target_lang, 
                                is_synthesis_prompt=for_synthesis, is_batch_synthesis=is_batch_synth_flag)
        
        if summary_part and isinstance(summary_part, str) and not (summary_part.startswith(f"{api_name_str} Error:") or "[[[Error" in summary_part):
            logger.info(f"Successfully processed chunk {chunk_idx+1} ({log_task_type}) with {api_name_str}.")
            return summary_part
        else:
            err_msg = summary_part if summary_part else 'Empty response'
            logger.warning(f"Error/no summary for chunk {chunk_idx+1} ({log_task_type}) from {api_name_str}: {err_msg}")
            return f"[[[Error processing chunk {chunk_idx+1} with {api_name_str}: {err_msg}]]]"
    except Exception as chunk_err:
        logger.error(f"Critical error calling {api_name_str} for chunk {chunk_idx+1} ({log_task_type}): {chunk_err}", exc_info=True)
        return f"[[[Critical system error processing chunk {chunk_idx+1} ({log_task_type}) with {api_name_str}]]]"


def summarize_chunks_and_combine(api_func, api_key_val: str, api_name_str: str, text_chunks: List[str], 
                                 summary_level_val: int, target_lang: Optional[str] = None, 
                                 for_synthesis: bool = False, is_batch_synth_flag: bool = False) -> str:
    """Processes text in chunks and combines the results, potentially synthesizing them."""
    if not api_key_val: return f"{api_name_str} Error: API Key not configured."
    num_chunks = len(text_chunks)
    target_language_full_name = _get_language_full_name(target_lang)
    log_task_type = "BATCH_SYNTH_MAP_REDUCE" if is_batch_synth_flag else ("SYNTH_MAP_REDUCE" if for_synthesis else "SUMM_MAP_REDUCE")

    logger.info(f"Starting {log_task_type} with {api_name_str} for {num_chunks} chunks (Target Lang: {target_language_full_name}). Max chunk workers: {MAX_CHUNK_WORKERS}")

    chunk_results = [None] * num_chunks
    chunk_errors_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CHUNK_WORKERS) as executor:
        future_to_chunk_idx = {
            executor.submit(_process_chunk_task, api_func, chunk_text, api_key_val, summary_level_val, 
                            target_lang, for_synthesis, is_batch_synth_flag, api_name_str, i): i
            for i, chunk_text in enumerate(text_chunks) if chunk_text and chunk_text.strip()
        }
        for future in concurrent.futures.as_completed(future_to_chunk_idx):
            chunk_idx = future_to_chunk_idx[future]
            try:
                summary_part = future.result()
                chunk_results[chunk_idx] = summary_part
                if isinstance(summary_part, str) and ("Error:" in summary_part or "[[[Error" in summary_part): 
                    chunk_errors_count += 1
            except Exception as exc:
                logger.error(f"Chunk {chunk_idx+1} ({log_task_type}) processing with {api_name_str} generated an exception: {exc}", exc_info=True)
                chunk_results[chunk_idx] = f"[[[System exception processing chunk {chunk_idx+1} with {api_name_str}]]]"
                chunk_errors_count += 1
    
    valid_chunk_summaries = [s for s in chunk_results if s and isinstance(s, str) and not ("Error:" in s or "[[[Error" in s)]
    
    if not valid_chunk_summaries:
        error_message = f"{api_name_str} Error: No valid chunk summaries/syntheses were created after parallel processing. Total errors: {chunk_errors_count}/{num_chunks}"
        logger.error(error_message)
        # Optionally, combine all error messages from chunks if needed for detailed feedback
        all_error_details = "\n".join([s for s in chunk_results if s and ("Error:" in s or "[[[Error" in s)])
        return f"{error_message}\nDetails:\n{all_error_details if all_error_details else 'No specific error details from chunks.'}"

    combined_text_for_final_pass = "\n\n--- Next Section ---\n\n".join(valid_chunk_summaries)
    logger.info(f"Combined {len(valid_chunk_summaries)} valid chunk results from {api_name_str} ({chunk_errors_count} errors). Total chars for final pass: {len(combined_text_for_final_pass)}. Starting final synthesis pass...")

    if not combined_text_for_final_pass.strip(): # Should not happen if valid_chunk_summaries is not empty
        logger.error(f"Error: Combined text from {api_name_str} is empty before final pass.")
        return f"{api_name_str} Error: Combined text from valid chunks is unexpectedly empty."

    try:
        # The final pass is always a synthesis of the combined (chunk summaries/syntheses).
        # If it was a batch synthesis, this final pass is synthesizing the syntheses of chunks of multiple docs.
        final_synthesis_level = max(int(summary_level_val * 0.8), 20) # Slightly less detail for final pass
        logger.info(f"Final synthesis pass for {api_name_str} (Target Lang: {target_language_full_name}) with detail level: {final_synthesis_level}%")

        # is_synthesis_prompt=True for the final combination step.
        # is_batch_synth_flag is passed to indicate the overall nature of the task for prompting.
        final_output = api_func(combined_text_for_final_pass, api_key_val, final_synthesis_level, 
                                target_lang, is_synthesis_prompt=True, is_batch_synthesis=is_batch_synth_flag)
        
        if chunk_errors_count > 0:
            error_note = f"\n\n(Note: {chunk_errors_count} out of {num_chunks} text sections encountered errors during initial processing with {api_name_str} and were excluded from this final result.)"
            if isinstance(final_output, str) and not final_output.startswith(f"{api_name_str} Error:"):
                 final_output += error_note
            else:
                 final_output = f"{error_note}\n\nAdditionally, the final combination step resulted in: {final_output}"
        
        logger.info(f"Final synthesis pass completed for {api_name_str}.")
        return final_output
    except Exception as final_err:
        logger.error(f"Error calling {api_name_str} for final synthesis pass: {final_err}", exc_info=True)
        return f"{api_name_str} Error: Critical error during final synthesis with {api_name_str}: {final_err}"


def _process_document_with_single_api(api_name: str, api_function, api_key: str, 
                                      text_to_process: str, num_chunks: int, chunks: List[str],
                                      summary_level: int, target_language_code: Optional[str], 
                                      is_synthesis_task: bool, is_batch_synthesis_task: bool, 
                                      task_name_for_log: str) -> Tuple[str, str]:
    """Helper to process text (single doc or combined batch) with one AI API, handling chunking."""
    target_lang_full = _get_language_full_name(target_language_code)
    log_task_type = "BATCH_SYNTHESIS" if is_batch_synthesis_task else ("SYNTHESIS" if is_synthesis_task else "SUMMARY")
    logger.info(f"Starting {api_name.capitalize()} API call for '{task_name_for_log}' ({log_task_type}, Target Lang: {target_lang_full})...")
    start_time = time.time()
    
    if not api_key: 
        logger.warning(f"{api_name.capitalize()} API key not provided for '{task_name_for_log}'. Skipping.")
        return api_name, f"{api_name.capitalize()} Error: API Key not configured."

    api_result: str
    if num_chunks == 1: # Single chunk, direct call
        api_result = api_function(chunks[0], api_key, summary_level, target_language_code, 
                                  is_synthesis_prompt=is_synthesis_task, 
                                  is_batch_synthesis=is_batch_synthesis_task)
    else: # Multiple chunks, use map-reduce
        api_result = summarize_chunks_and_combine(api_function, api_key, api_name.capitalize(), chunks, 
                                                  summary_level, target_language_code, 
                                                  for_synthesis=is_synthesis_task, # for_synthesis for chunk processing
                                                  is_batch_synth_flag=is_batch_synthesis_task) # overall task type

    end_time = time.time()
    logger.info(f"{api_name.capitalize()} API call for '{task_name_for_log}' ({log_task_type}) took {end_time - start_time:.2f} seconds.")

    if isinstance(api_result, str) and api_result.startswith(f"{api_name.capitalize()} Error:"):
        logger.error(f"{api_name.capitalize()} API returned error for '{task_name_for_log}': {api_result}")
    elif not api_result: 
        logger.warning(f"{api_name.capitalize()} API returned no result for '{task_name_for_log}'.")
        api_result = f"{api_name.capitalize()} Error: No content returned from API."
    else:
        logger.info(f"{api_name.capitalize()} API call for '{task_name_for_log}' completed successfully.")
    return api_name, api_result


def process_document(file_path: Optional[str] = None,
                     input_text_to_process: Optional[str] = None,
                     is_synthesis_task: bool = False, # True if summarizing chunks of a single doc
                     deepseek_key: Optional[str] = None,
                     grok_key: Optional[str] = None,
                     chatgpt_key: Optional[str] = None,
                     summary_level: int = 50, 
                     target_language_code: Optional[str] = None) -> Dict[str, Any]:
    """Processes a single document for text extraction, analysis, and AI summarization."""
    
    task_name = os.path.basename(file_path) if file_path else "Direct Input Text"
    processing_type = "SINGLE_DOC_SYNTHESIS" if is_synthesis_task else "INDIVIDUAL_SUMMARY"
    lang_str = f"Language: {_get_language_full_name(target_language_code)}"
    logger.info(f"Starting [{processing_type}]: '{task_name}' (Detail: {summary_level}%, {lang_str})")

    results: Dict[str, Any] = {'original_text': None, 'deepseek': None, 'grok': None, 'chatgpt': None, 'analysis': None, 'error': None}
    text_for_processing: Optional[str] = None
    analysis_error_msg: Optional[str] = None

    try:
        # 1. Get Text Content
        try:
            if input_text_to_process is not None:
                text_for_processing = input_text_to_process
                if not text_for_processing or not text_for_processing.strip():
                    raise ValueError("Direct input text is empty or whitespace only.")
                logger.info(f"Using direct input text for '{task_name}', length: {len(text_for_processing)} characters.")
            elif file_path:
                text_for_processing = extract_text(file_path) # Can raise errors
                if not text_for_processing or not text_for_processing.strip():
                     raise ValueError(f"Text extraction from '{task_name}' returned empty or whitespace only content.")
                logger.info(f"Extracted {len(text_for_processing)} characters from '{task_name}'.")
            else:
                raise ValueError("Missing input: Either 'file_path' or 'input_text_to_process' must be provided.")
            results['original_text'] = text_for_processing
        except (FileNotFoundError, PermissionError, IsADirectoryError, ValueError, IOError, RuntimeError) as ext_err:
            logger.error(f"Input or text extraction error for '{task_name}': {ext_err}", exc_info=True)
            results['error'] = f"Input/Extraction Error: {ext_err}"
            results['analysis'] = {'error': f"Analysis skipped due to input/extraction error: {ext_err}"}
            return results # Critical error, cannot proceed

        # 2. Perform Text Analysis
        logger.info(f"Starting text analysis for '{task_name}'...")
        try:
            analysis_data = {
                 'word_count': TextAnalysis.count_words(text_for_processing),
                 'sentence_count': TextAnalysis.count_sentences(text_for_processing),
                 'paragraph_count': TextAnalysis.count_paragraphs(text_for_processing),
                 'avg_word_length': TextAnalysis.average_word_length(text_for_processing),
                 'avg_sentence_length': TextAnalysis.average_sentence_length(text_for_processing),
                 'common_words': TextAnalysis.most_common_words(text_for_processing, top_n=50)
             }
            analysis_errors = [f"{k}: {v}" for k, v in analysis_data.items() if v == "Error"]
            if analysis_errors:
                analysis_error_msg = "Text analysis encountered errors: " + "; ".join(analysis_errors)
                logger.error(analysis_error_msg)
                analysis_data['error'] = analysis_error_msg # Add error to analysis dict
            else:
                logger.info(f"Text analysis for '{task_name}' completed.")
            results['analysis'] = analysis_data
        except Exception as analysis_exception:
            logger.error(f"Critical error during text analysis for '{task_name}': {analysis_exception}", exc_info=True)
            analysis_error_msg = f"System error during text analysis: {analysis_exception}"
            results['analysis'] = {'error': analysis_error_msg}

        # 3. Call AI Models
        active_api_configs = []
        if deepseek_key and deepseek_key.strip(): active_api_configs.append({'name': 'deepseek', 'func': summarize_with_deepseek, 'key': deepseek_key})
        if grok_key and grok_key.strip(): active_api_configs.append({'name': 'grok', 'func': summarize_with_grok, 'key': grok_key})
        if chatgpt_key and chatgpt_key.strip(): active_api_configs.append({'name': 'chatgpt', 'func': summarize_with_chatgpt, 'key': chatgpt_key})

        if active_api_configs:
            try: from config.constants import MAX_CHUNK_SIZE 
            except ImportError: MAX_CHUNK_SIZE = 40000; logger.warning(f"MAX_CHUNK_SIZE not in constants, using default: {MAX_CHUNK_SIZE}")

            chunks = [text_for_processing[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text_for_processing), MAX_CHUNK_SIZE)]
            num_chunks = len(chunks)
            logger.debug(f"Preparing to call AI models for '{task_name}'. Number of chunks: {num_chunks} (MAX_CHUNK_SIZE: {MAX_CHUNK_SIZE})")

            overall_api_start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
                future_to_api_name = {
                    executor.submit(
                        _process_document_with_single_api, # Reusing this helper
                        call_info['name'], call_info['func'], call_info['key'],
                        text_for_processing, num_chunks, chunks, 
                        summary_level, target_language_code,
                        is_synthesis_task, False, # is_batch_synthesis_task is False for single doc
                        task_name
                    ): call_info['name'] for call_info in active_api_configs
                }
                for future in concurrent.futures.as_completed(future_to_api_name):
                    api_name_res = future_to_api_name[future]
                    try:
                        _, api_result_content = future.result() # Helper returns (name, result)
                        results[api_name_res] = api_result_content
                    except Exception as exc_future:
                        logger.error(f"AI call for {api_name_res} ('{task_name}') failed in future: {exc_future}", exc_info=True)
                        results[api_name_res] = f"{api_name_res.capitalize()} Error: System error during parallel execution ({exc_future})"
            logger.info(f"All AI model calls for '{task_name}' completed in {time.time() - overall_api_start_time:.2f}s.")
        else:
             logger.info(f"No AI API keys provided or models enabled for '{task_name}'. Skipping AI summarization.")
             results['deepseek'] = results['grok'] = results['chatgpt'] = "<Not executed: No API key or model not selected>"
        
        # Consolidate errors
        api_error_messages = [results[api_key] for api_key in ['deepseek', 'grok', 'chatgpt'] if isinstance(results.get(api_key), str) and "Error:" in results[api_key]]
        
        final_error_parts = []
        if results.get('error'): final_error_parts.append(results['error']) # From extraction
        if analysis_error_msg: final_error_parts.append(analysis_error_msg) # From analysis
        if api_error_messages: final_error_parts.extend(api_error_messages) # From APIs

        if final_error_parts:
            results['error'] = ". ".join(filter(None, final_error_parts))
        else:
            results['error'] = None # Explicitly None if no errors

        logger.info(f"Completed processing [{processing_type}] for: '{task_name}'. Overall error status: {results['error'] if results['error'] else 'None'}")
        return results

    except Exception as e_main: # Catch-all for unexpected errors in this main function
        logger.critical(f"Unexpected critical system error in process_document for '{task_name}': {e_main}", exc_info=True)
        err_msg = f"Unexpected critical system error: {e_main}"
        # Ensure all keys are present in the returned dict for UI consistency
        return {
            'original_text': text_for_processing or "", 'analysis': results.get('analysis', {'error': err_msg}),
            'deepseek': results.get('deepseek', f"DeepSeek Error: {err_msg}"), 
            'grok': results.get('grok', f"Grok Error: {err_msg}"), 
            'chatgpt': results.get('chatgpt', f"ChatGPT Error: {err_msg}"),
            'error': err_msg
        }

# --- NEW BATCH SYNTHESIS FUNCTION ---
def process_batch_synthesis(file_paths: List[str],
                            deepseek_key: Optional[str] = None,
                            grok_key: Optional[str] = None,
                            chatgpt_key: Optional[str] = None,
                            summary_level: int = 50,
                            target_language_code: Optional[str] = None) -> Dict[str, Any]:
    """
    Processes a batch of documents, extracts text from each, concatenates them,
    and then performs a synthesis using enabled AI models.
    """
    batch_task_name = f"Batch Synthesis of {len(file_paths)} documents"
    lang_str = f"Language: {_get_language_full_name(target_language_code)}"
    logger.info(f"Starting [{batch_task_name}] (Detail: {summary_level}%, {lang_str})")

    results: Dict[str, Any] = {
        'processed_files': [],
        'failed_files': [], # Tuples of (filename, error_message)
        'concatenated_text_char_count': 0,
        'deepseek_synthesis': None,
        'grok_synthesis': None,
        'chatgpt_synthesis': None,
        'overall_error': None
    }
    
    all_extracted_texts: List[str] = []
    extraction_errors: List[str] = []

    # 1. Extract text from all documents
    logger.info(f"Extracting text from {len(file_paths)} documents for batch synthesis...")
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        try:
            text = extract_text(file_path)
            if text and text.strip():
                all_extracted_texts.append(text)
                results['processed_files'].append(file_name)
            else:
                logger.warning(f"No content extracted from '{file_name}' or content was empty; skipping for batch.")
                extraction_errors.append(f"'{file_name}': No content or empty.")
                results['failed_files'].append((file_name, "No content or empty"))
        except Exception as ext_err:
            logger.error(f"Failed to extract text from '{file_name}' for batch: {ext_err}", exc_info=False) # Keep log less verbose for individual file errors in batch
            extraction_errors.append(f"'{file_name}': {ext_err}")
            results['failed_files'].append((file_name, str(ext_err)))

    if not all_extracted_texts:
        err_msg = "No text could be extracted from any of the provided documents for batch synthesis."
        logger.error(err_msg)
        results['overall_error'] = err_msg
        if extraction_errors:
             results['overall_error'] += " Extraction issues: " + "; ".join(extraction_errors)
        return results

    concatenated_text = DOCUMENT_SEPARATOR.join(all_extracted_texts)
    results['concatenated_text_char_count'] = len(concatenated_text)
    logger.info(f"Concatenated text from {len(all_extracted_texts)} documents, total chars: {results['concatenated_text_char_count']}.")

    # 2. Call AI Models for Batch Synthesis
    active_api_configs_batch = []
    if deepseek_key and deepseek_key.strip(): active_api_configs_batch.append({'name': 'deepseek', 'func': summarize_with_deepseek, 'key': deepseek_key, 'result_key': 'deepseek_synthesis'})
    if grok_key and grok_key.strip(): active_api_configs_batch.append({'name': 'grok', 'func': summarize_with_grok, 'key': grok_key, 'result_key': 'grok_synthesis'})
    if chatgpt_key and chatgpt_key.strip(): active_api_configs_batch.append({'name': 'chatgpt', 'func': summarize_with_chatgpt, 'key': chatgpt_key, 'result_key': 'chatgpt_synthesis'})

    if active_api_configs_batch:
        try: from config.constants import MAX_CHUNK_SIZE
        except ImportError: MAX_CHUNK_SIZE = 40000; logger.warning(f"MAX_CHUNK_SIZE not in constants, using default for batch: {MAX_CHUNK_SIZE}")

        chunks = [concatenated_text[i:i+MAX_CHUNK_SIZE] for i in range(0, len(concatenated_text), MAX_CHUNK_SIZE)]
        num_chunks = len(chunks)
        logger.debug(f"Preparing AI calls for batch synthesis. Number of chunks for concatenated text: {num_chunks}")

        overall_api_start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
            future_to_api_info = {
                executor.submit(
                    _process_document_with_single_api, # Reusing this helper
                    call_info['name'], call_info['func'], call_info['key'],
                    concatenated_text, num_chunks, chunks, 
                    summary_level, target_language_code,
                    True, True, # is_synthesis_task=True, is_batch_synthesis_task=True
                    batch_task_name 
                ): call_info for call_info in active_api_configs_batch # Store full call_info
            }
            for future in concurrent.futures.as_completed(future_to_api_info):
                api_info = future_to_api_info[future]
                api_name_res = api_info['name']
                result_key = api_info['result_key']
                try:
                    _, api_result_content = future.result()
                    results[result_key] = api_result_content
                except Exception as exc_future_batch:
                    logger.error(f"AI call for {api_name_res} (batch synthesis) failed in future: {exc_future_batch}", exc_info=True)
                    results[result_key] = f"{api_name_res.capitalize()} Error: System error during parallel batch execution ({exc_future_batch})"
        logger.info(f"All AI model calls for batch synthesis completed in {time.time() - overall_api_start_time:.2f}s.")
    else:
        logger.info("No AI API keys provided or models enabled for batch synthesis. Skipping AI synthesis.")
        results['deepseek_synthesis'] = results['grok_synthesis'] = results['chatgpt_synthesis'] = "<Not executed: No API key or model not selected>"

    # Consolidate errors for batch
    batch_api_error_messages = [results[api_conf['result_key']] for api_conf in active_api_configs_batch if isinstance(results.get(api_conf['result_key']), str) and "Error:" in results[api_conf['result_key']]]
    
    final_batch_error_parts = []
    if extraction_errors: final_batch_error_parts.append(f"Extraction issues with files: {'; '.join(extraction_errors)}")
    if batch_api_error_messages: final_batch_error_parts.extend(batch_api_error_messages)

    if final_batch_error_parts:
        results['overall_error'] = ". ".join(filter(None, final_batch_error_parts))
    else:
        results['overall_error'] = None

    logger.info(f"Completed [{batch_task_name}]. Overall error status: {results['overall_error'] if results['overall_error'] else 'None'}")
    return results

def process_document_async(file_path: Optional[str] = None, input_text: Optional[str] = None, 
                           settings: Optional[Dict[str, Any]] = None, progress_callback=None) -> Dict[str, Any]:
    if progress_callback: progress_callback("Starting document processing...")
    actual_settings = settings or {}
    try:
        # Ensure settings are correctly passed for single document processing
        result = process_document(
            file_path=file_path,
            input_text_to_process=input_text,
            is_synthesis_task=actual_settings.get('is_synthesis_task', False), # For single doc chunk synthesis
            deepseek_key=actual_settings.get('deepseek_key'),
            grok_key=actual_settings.get('grok_key'),
            chatgpt_key=actual_settings.get('chatgpt_key'),
            summary_level=int(actual_settings.get('summary_level', 50)),
            target_language_code=actual_settings.get('target_language_code')
        )
        if progress_callback:
            msg = f"Processing completed with issues: {result.get('error')}" if result.get('error') else "Processing completed successfully."
            progress_callback(msg)
        return result
    except Exception as e:
        error_msg = f"Async processing wrapper for single document failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if progress_callback: progress_callback(error_msg)
        return {'error': error_msg, 'original_text': '', 'deepseek': None, 'grok': None, 'chatgpt': None, 'analysis': {'error': error_msg}}


def extract_text_preview(file_path: str, max_chars: int = 1000) -> str: 
    """Quick text preview for web UI, extracts a snippet of text."""
    try:
        full_text = extract_text(file_path) # Can raise errors
        if not full_text: return "No content found in document."
        return full_text[:max_chars] + ("..." if len(full_text) > max_chars else "")
    except Exception as e:
        logger.error(f"Error creating text preview for '{os.path.basename(file_path)}': {e}", exc_info=False) # Less verbose for preview errors
        return f"Error reading file preview: {str(e)}"


def validate_document_settings(settings: Dict[str, Any]) -> Dict[str, Any]: 
    errors: List[str] = []
    
    summary_level_str = settings.get('summary_level', '50')
    try:
        summary_level = int(summary_level_str)
        if not (10 <= summary_level <= 90):
            errors.append("Summary level must be an integer between 10 and 90.")
    except (ValueError, TypeError):
        errors.append("Summary level must be a valid integer.")

    target_language_code = settings.get('target_language_code')
    if target_language_code is not None:
        if not isinstance(target_language_code, str):
            errors.append("Target language code must be a string if provided.")
        elif not re.match(r"^[a-z]{2}(-[A-Z]{2})?$", target_language_code) and target_language_code != "": # Allow empty string for default
             # This is a soft validation, as actual supported codes depend on AI
            logger.debug(f"Potentially unconventional language code format: '{target_language_code}'. API will determine validity.")
  
    return {"valid": not errors, "errors": "; ".join(errors) if errors else None}
