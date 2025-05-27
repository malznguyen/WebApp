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

logger = logging.getLogger('ImageSearchApp')  

class TextAnalysis: 
    @staticmethod
    def count_words(text):
        try:
            if not isinstance(text, str): return 0
            words = re.findall(r'\b\w+\b', text.lower())
            return len(words)
        except Exception as e:
            logger.error(f"Error counting words: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def count_sentences(text):
        try:
            if not isinstance(text, str): return 0
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
            sentences = [s for s in sentences if s and s.strip()]
            return len(sentences) if sentences else (1 if text.strip() else 0)
        except Exception as e:
            logger.error(f"Error counting sentences: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def count_paragraphs(text):
        try:
            if not isinstance(text, str): return 0
            paragraphs = re.split(r'\n\s*\n', text)
            paragraphs = [p for p in paragraphs if p and p.strip()]
            return len(paragraphs) if paragraphs else (1 if text.strip() else 0)
        except Exception as e:
            logger.error(f"Error counting paragraphs: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def average_word_length(text):
        try:
            if not isinstance(text, str): return 0
            words = re.findall(r'\b\w+\b', text.lower())
            if not words: return 0
            return sum(len(word) for word in words) / len(words)
        except Exception as e:
            logger.error(f"Error calculating average word length: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def average_sentence_length(text):
        try:
            if not isinstance(text, str): return 0
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
            sentences = [s for s in sentences if s and s.strip()]
            if not sentences: return 0
            total_words = sum(len(re.findall(r'\b\w+\b', s.lower())) for s in sentences)
            return total_words / len(sentences)
        except Exception as e:
            logger.error(f"Error calculating average sentence length: {e}", exc_info=True)
            return "Error"

    @staticmethod
    def most_common_words(text, top_n=50):
        try:
            if not isinstance(text, str): return []

            stop_words = {
                'và', 'là', 'có', 'của', 'ở', 'tại', 'trong', 'trên', 'dưới', 'cho', 'đến', 'với', 'bởi', 'qua', 'về',
                'như', 'mà', 'thì', 'rằng', 'nhưng', 'nếu', 'hay', 'hoặc', 'khi', 'lúc', 'sau', 'trước', 'từ', 'để',
                'không', 'chưa', 'được', 'bị', 'phải', 'cần', 'nên', 'cũng', 'vẫn', 'chỉ', 'ngay', 'luôn', 'rất',
                'quá', 'hơn', 'kém', 'nhất', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
                'vài', 'nhiều', 'ít', 'mọi', 'mỗi', 'các', 'những', 'này', 'kia', 'đó', 'ấy', 'đây', 'tôi', 'ta',
                'chúng tôi', 'chúng ta', 'bạn', 'anh', 'chị', 'em', 'ông', 'bà', 'họ', 'chúng nó', 'ai', 'gì', 'nào',
                'đâu', 'bao giờ', 'bao lâu', 'vì sao', 'tại sao', 'thế nào', 'ra sao', 'làm sao', 'việc', 'điều', 'thứ',
                'cách', 'lần', 'người', 'ngày', 'tháng', 'năm', 'giờ', 'phút', 'giây', 'theo', 'cùng', 'riêng', 'chung',
                'khác', 'giống', 'nhau', 'tự', 'chính', 'nhất là', 'ngoài ra', 'hơn nữa', 'tuy nhiên', 'mặc dù', 'do đó',
                'vì vậy', 'cho nên', 'để mà', 'số', 'thông tin', 'sử dụng', 'hình', 'trang', 'mục', 'phần', 'chương', 'bảng',
                'hình ảnh', 'ví dụ', 'thực hiện', 'bao gồm', 'liên quan', 'quá trình', 'kết quả', 'vấn đề', 'giải pháp',
                'đầu tiên', 'cuối cùng', 'tổng cộng', 'trung bình', 'khoảng', 'gần', 'hầu hết', 'đặc biệt', 'quan trọng',
                'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t', 'as', 'at',
                'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can',
                'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during',
                'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he',
                'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s',
                'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself',
                'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once',
                'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she',
                'she\'d', 'she\'ll', 'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the',
                'their', 'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll',
                'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'wasn\'t',
                'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where',
                'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t',
                'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves',
                'fig', 'figure', 'table', 'section', 'chapter', 'page', 'appendix', 'index', 'image', 'example'
            }

            words = re.findall(r"\b[a-zA-Zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳýỵỷỹ]+(?:[-'\u2019][a-zA-Zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳýỵỷỹ]+)*\b", text.lower())
            filtered_words = [word for word in words if word not in stop_words and len(word) > 1 and not word.isdigit()]
            if not filtered_words: return []
            return Counter(filtered_words).most_common(top_n)
        except Exception as e:
            logger.error(f"Error finding common words: {e}", exc_info=True)
            return []

def extract_text(file_path):
    if not file_path or not isinstance(file_path, str):
        raise ValueError("Invalid file path.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not os.path.isfile(file_path):
        raise IsADirectoryError(f"Path is directory, not file: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"No read permission: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    logger.info(f"Extracting text from: {file_name} (format: {file_extension})")

    try:
        if file_extension == '.pdf':
            text = ""
            with open(file_path, 'rb') as file:
                try:
                    pdf_reader = PyPDF2.PdfReader(file, strict=False)
                    if pdf_reader.is_encrypted:
                        logger.warning(f"PDF '{file_name}' is encrypted. Cannot extract.")
                        raise ValueError(f"Encrypted PDF: {file_name}")

                    num_pages = len(pdf_reader.pages)
                    logger.debug(f"Reading {num_pages} pages from PDF '{file_name}'...")
                    for page_num in range(num_pages):
                        try:
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text: text += page_text + "\n"
                        except Exception as page_err:
                            logger.warning(f"Error reading page {page_num + 1} of PDF '{file_name}': {page_err}. Skipping page.")
                            text += f"\n[[Error reading page {page_num + 1}]]\n"
                    logger.info(f"Extracted from PDF '{file_name}', {len(text)} characters.")
                    return text

                except PyPDF2.errors.PdfReadError as pdf_err:
                    logger.error(f"Critical error reading PDF '{file_name}': {pdf_err}", exc_info=True)
                    raise ValueError(f"Invalid or corrupted PDF: {file_name}")

        elif file_extension == '.docx':
            try:
                text = docx2txt.process(file_path)
                logger.info(f"Extracted from DOCX '{file_name}', {len(text)} characters.")
                return text
            except Exception as docx_err:
                 logger.error(f"Error extracting from DOCX '{file_name}': {docx_err}", exc_info=True)
                 raise ValueError(f"Cannot process DOCX file: {file_name}")

        elif file_extension in ['.txt', '.md']:
            encodings_to_try = ['utf-8', 'utf-16', 'cp1252', 'latin-1']
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        text = file.read()
                    logger.info(f"Extracted from {file_extension.upper()} '{file_name}' (encoding: {enc}), {len(text)} characters.")
                    return text
                except UnicodeDecodeError:
                    logger.debug(f"Failed encoding {enc} for '{file_name}', trying next...")
                except Exception as txt_err:
                    logger.error(f"Error reading text file '{file_name}' with encoding {enc}: {str(txt_err)}", exc_info=True)

            logger.error(f"Cannot read {file_extension.upper()} file '{file_name}' with tried encodings.")
            raise IOError(f"Cannot read file {file_name} with available encodings.")

        else:
            error_msg = f"Unsupported file format: {file_extension} ({file_name})"
            logger.error(error_msg)
            raise ValueError(error_msg)

    except (FileNotFoundError, PermissionError, IsADirectoryError, ValueError, IOError) as specific_err:
        raise specific_err
    except Exception as e:
        logger.error(f"Unexpected error extracting text from {file_name}: {str(e)}", exc_info=True)
        raise RuntimeError(f"System error extracting file: {e}")


def _get_language_full_name(lang_code): 
    lang_map = {'vi': 'Vietnamese', 'en': 'English'}
    return lang_map.get(lang_code, lang_code)


def _create_summary_prompt(text_to_summarize, summary_level, language_instruction, is_synthesis_prompt=False):
    MAX_PROMPT_TEXT_LENGTH = 110000 
    trimmed_text = text_to_summarize[:MAX_PROMPT_TEXT_LENGTH]
    if len(text_to_summarize) > MAX_PROMPT_TEXT_LENGTH:
        logger.warning(f"Prompt text was trimmed from {len(text_to_summarize)} to {len(trimmed_text)} characters.")

    if is_synthesis_prompt:
        task_description = f"""Synthesize the key information from the following combined text (derived from multiple documents on a similar topic). Provide a comprehensive overview that integrates the main points, arguments, and findings. The desired detail level for this synthesis is {summary_level}%. {language_instruction} Combined text to synthesize: --- START COMBINED TEXT --- {trimmed_text} --- END COMBINED TEXT ---"""
        system_role_content = "You are an expert AI assistant specializing in synthesizing information from multiple text sources..."
    else:
        task_description = f"""Create a summary for the following document with a detail level of {summary_level}%. {language_instruction} Document to summarize: --- START DOCUMENT --- {trimmed_text} --- END DOCUMENT ---"""
        system_role_content = "You are a professional AI assistant specializing in summarizing documents clearly..."
    return task_description, system_role_content


def _handle_api_error(api_name, exception): 
    logger.error(f"Error using {api_name} API: {str(exception)}", exc_info=True)
    error_message = str(exception).lower()

    if isinstance(exception, requests.exceptions.Timeout):
        return f"{api_name} Error: Request timeout."
    if isinstance(exception, requests.exceptions.ConnectionError):
        return f"{api_name} Error: Network connection error."
    if isinstance(exception, requests.exceptions.RequestException):
         return f"{api_name} Error: Unknown request error ({exception})."

    # Specific OpenAI/DeepSeek/Grok like error parsing
    # Check if the exception itself is an APIError from openai library
    if hasattr(exception, 'status_code'): # Likely an openai.APIError
        if exception.status_code == 401:
            return f"{api_name} Error: Invalid API key or authentication error."
        if exception.status_code == 429:
            return f"{api_name} Error: Rate limit or quota exceeded."
        if exception.status_code == 400 and "context_length_exceeded" in error_message: # Check message for context length
             return f"{api_name} Error: Input text too long for model."
        if exception.status_code == 400:
             return f"{api_name} Error: Bad request (check parameters/data - Status {exception.status_code})."
        if exception.status_code in [500, 502, 503, 504]:
             return f"{api_name} Error: Temporary server error (Status {exception.status_code}). Try again later."

    # Fallback for general error messages
    if "authentication" in error_message or "incorrect api key" in error_message:
        return f"{api_name} Error: Invalid API key or authentication error."
    if "rate limit" in error_message or "limit" in error_message or "quota" in error_message:
        return f"{api_name} Error: Rate limit or quota exceeded."
    if "context_length_exceeded" in error_message: # Double check for non-APIError exceptions
        return f"{api_name} Error: Input text too long for model."

    return f"{api_name} API unknown error: {exception}"

def summarize_with_deepseek(text, api_key, summary_level=50, target_language_code=None, is_synthesis_prompt=False):
    if not api_key: return "DeepSeek Error: API Key not configured."
    if not text or not text.strip(): return "DeepSeek Error: Empty input text."

    from config.constants import API_TIMEOUT_SEC # Lấy timeout từ constants

    target_language_full = _get_language_full_name(target_language_code) if target_language_code else "default"
    language_instruction = ""
    system_role_extra = ""
    if target_language_code:
        language_instruction = f" IMPORTANT: Please provide the output in {target_language_full}."
        system_role_extra = f" You MUST provide the output ONLY in {target_language_full}."

    prompt_text, system_content_base = _create_summary_prompt(text, summary_level, language_instruction, is_synthesis_prompt)
    system_content = f"{system_content_base}{system_role_extra}"

    logger.debug(f"Calling DeepSeek API ({'SYNTHESIS' if is_synthesis_prompt else 'SUMMARY'}...)")
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.with_options(timeout=API_TIMEOUT_SEC).chat.completions.create( # Sử dụng API_TIMEOUT_SEC
            model="deepseek-chat", # 
            messages=[{"role": "system", "content": system_content},{"role": "user", "content": prompt_text}],
            max_tokens=8048, temperature=0.5 
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            logger.info(f"DeepSeek returned {'synthesis' if is_synthesis_prompt else 'summary'} ({target_language_full}) {len(summary)} characters.")
            return summary
        else:
            logger.error(f"DeepSeek API returned unexpected structure: {response}")
            return "DeepSeek Error: Invalid API response."
    except Exception as e:
        return _handle_api_error("DeepSeek", e)


def summarize_with_grok(text, api_key, summary_level=50, target_language_code=None, is_synthesis_prompt=False):
    if not api_key: return "Grok Error: API Key not configured."
    if not text or not text.strip(): return "Grok Error: Empty input text."

    from config.constants import API_TIMEOUT_SEC 

    target_language_full = _get_language_full_name(target_language_code) if target_language_code else "default"
    language_instruction = ""
    system_role_extra = ""
    if target_language_code:
        language_instruction = f"\nVERY IMPORTANT: The final output MUST be in {target_language_full}."
        system_role_extra = f" Always respond ONLY in {target_language_full}."

    prompt_text, system_content_base = _create_summary_prompt(text, summary_level, language_instruction, is_synthesis_prompt)
    system_content = f"{system_content_base}{system_role_extra}"
    if is_synthesis_prompt:
        system_content = f"You are an AI that synthesizes information from multiple texts.{system_role_extra}"

    logger.debug(f"Calling Grok API ({'SYNTHESIS' if is_synthesis_prompt else 'SUMMARY'}...)")
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        response = client.with_options(timeout=API_TIMEOUT_SEC).chat.completions.create( 
            model="grok-2", 
            messages=[{"role": "system", "content": system_content},{"role": "user", "content": prompt_text}],
            max_tokens=8048, temperature=0.5
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            logger.info(f"Grok returned {'synthesis' if is_synthesis_prompt else 'summary'} ({target_language_full}) {len(summary)} characters.")
            return summary
        else:
            logger.error(f"Grok API returned unexpected structure: {response}")
            return "Grok Error: Invalid API response."
    except Exception as e:
        return _handle_api_error("Grok", e)


def summarize_with_chatgpt(text, api_key, summary_level=50, target_language_code=None, is_synthesis_prompt=False):
    if not api_key: return "ChatGPT Error: API Key not configured."
    if not text or not text.strip(): return "ChatGPT Error: Empty input text."

    from config.constants import API_TIMEOUT_SEC # Lấy timeout từ constants

    target_language_full = _get_language_full_name(target_language_code) if target_language_code else "default"
    language_instruction = ""
    system_role_extra = ""
    if target_language_code:
        language_instruction = f" CRITICAL REQUIREMENT: Provide the final output ONLY in {target_language_full}."
        system_role_extra = f" You must respond exclusively in {target_language_full}."

    prompt_text, system_content_base = _create_summary_prompt(text, summary_level, language_instruction, is_synthesis_prompt)
    system_content = f"{system_content_base}{system_role_extra}"

    logger.debug(f"Calling ChatGPT API ({'SYNTHESIS' if is_synthesis_prompt else 'SUMMARY'}...)")
    try:
        client = OpenAI(api_key=api_key)
        response = client.with_options(timeout=API_TIMEOUT_SEC).chat.completions.create( # Sử dụng API_TIMEOUT_SEC
            model="gpt-4o", # Hoặc "gpt-3.5-turbo" nếu muốn tiết kiệm chi phí và chấp nhận chất lượng thấp hơn
            messages=[{"role": "system", "content": system_content},{"role": "user", "content": prompt_text}],
            max_tokens=4096, 
            temperature=0.6
        )

        if response and response.choices and response.choices[0].message and response.choices[0].message.content:
            summary = response.choices[0].message.content.strip()
            logger.info(f"ChatGPT returned {'synthesis' if is_synthesis_prompt else 'summary'} ({target_language_full}) {len(summary)} characters.")
            return summary
        else:
            logger.error(f"ChatGPT API returned unexpected structure: {response}")
            return "ChatGPT Error: Invalid API response."
    except Exception as e:
        return _handle_api_error("ChatGPT", e)



MAX_API_WORKERS = 3 
MAX_CHUNK_WORKERS = 3 
def _process_chunk_task(api_func, chunk_text, api_key_val, summary_level_val, target_lang, for_synthesis, api_name_str, chunk_idx):
    """Helper function to run each chunk summarization in a thread."""
    logger.info(f"Starting chunk {chunk_idx+1} processing with {api_name_str}...")
    try:
        summary_part = api_func(chunk_text, api_key_val, summary_level_val, target_lang, is_synthesis_prompt=for_synthesis)
        if summary_part and isinstance(summary_part, str) and "Error" not in summary_part:
            logger.info(f"Successfully processed chunk {chunk_idx+1} with {api_name_str}.")
            return summary_part
        else:
            logger.warning(f"Error/no summary for chunk {chunk_idx+1} from {api_name_str}: {summary_part}")
            return f"[[[Error processing chunk {chunk_idx+1} with {api_name_str}: {summary_part}]]]"
    except Exception as chunk_err:
        logger.error(f"Critical error calling {api_name_str} for chunk {chunk_idx+1}: {chunk_err}", exc_info=True)
        return f"[[[Critical system error processing chunk {chunk_idx+1} with {api_name_str}]]]"

def summarize_chunks_and_combine(api_func, api_key_val, api_name_str, text_chunks, summary_level_val, target_lang=None, for_synthesis=False):

    if not api_key_val: return f"{api_name_str} not configured."
    num_chunks = len(text_chunks)
    logger.info(f"Starting Map-Reduce with {api_name_str} for {num_chunks} chunks (Synthesis: {for_synthesis}). Max chunk workers: {MAX_CHUNK_WORKERS}")

    chunk_summaries = [None] * num_chunks # Để giữ đúng thứ tự
    chunk_errors = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CHUNK_WORKERS) as executor:
        future_to_chunk_idx = {
            executor.submit(_process_chunk_task, api_func, chunk_text, api_key_val, summary_level_val, target_lang, for_synthesis, api_name_str, i): i
            for i, chunk_text in enumerate(text_chunks) if chunk_text and chunk_text.strip()
        }

        for future in concurrent.futures.as_completed(future_to_chunk_idx):
            chunk_idx = future_to_chunk_idx[future]
            try:
                summary_part = future.result()
                chunk_summaries[chunk_idx] = summary_part
                if "Error" in summary_part or "Critical" in summary_part: 
                    chunk_errors += 1
            except Exception as exc:
                logger.error(f"Chunk {chunk_idx+1} processing with {api_name_str} generated an exception: {exc}", exc_info=True)
                chunk_summaries[chunk_idx] = f"[[[System exception processing chunk {chunk_idx+1} with {api_name_str}]]]"
                chunk_errors += 1

    processed_chunk_summaries = [s for s in chunk_summaries if s is not None]

    if not processed_chunk_summaries or chunk_errors == len(processed_chunk_summaries): 
        logger.error(f"Error: No valid chunk summaries created with {api_name_str}. Errors: {chunk_errors}/{len(processed_chunk_summaries)}")
        return f"Error: No valid chunk summaries created with {api_name_str} after parallel processing."

    combined_summary_text = "\n\n--- Next Section ---\n\n".join(processed_chunk_summaries) 
    logger.info(f"Combined {len(processed_chunk_summaries)} chunk summaries from {api_name_str} ({chunk_errors} errors). Total characters for final pass: {len(combined_summary_text)}. Starting final pass...")

    if not combined_summary_text.strip():
        logger.error(f"Error: Combined text from {api_name_str} is empty before final pass.")
        return f"Error: Combined text from {api_name_str} is empty."

    try:
        final_summary_level_val = max(int(summary_level_val * 0.8), 20) 
        logger.info(f"Final pass for {api_name_str} with detail level: {final_summary_level_val}%")

        final_output_text = api_func(combined_summary_text, api_key_val, final_summary_level_val, target_lang, is_synthesis_prompt=True) # Final pass luôn là synthesis
        if chunk_errors > 0:
            final_output_text += f"\n\n(Note: {chunk_errors} errors occurred while processing some text sections with {api_name_str}.)"
        logger.info(f"Final pass completed for {api_name_str}.")
        return final_output_text
    except Exception as final_err:
        logger.error(f"Error calling {api_name_str} for final pass: {final_err}", exc_info=True)
        return f"Critical error during final summary/synthesis with {api_name_str}: {final_err}"


def _process_with_single_api(api_name, api_function, api_key_value, text_for_processing, num_chunks, chunks, summary_level, target_language_code, is_synthesis_task, task_name):
    """Helper function to process document with a single API, handling chunking."""
    logger.info(f"Starting {api_name.capitalize()} API call for '{task_name}'...")
    start_time = time.time()
    api_result = None
    if num_chunks == 1:
        api_result = api_function(chunks[0], api_key_value, summary_level, target_language_code, is_synthesis_prompt=is_synthesis_task)
    else:
        api_result = summarize_chunks_and_combine(api_function, api_key_value, api_name.capitalize(), chunks, summary_level, target_language_code, for_synthesis=is_synthesis_task)

    end_time = time.time()
    logger.info(f"{api_name.capitalize()} API call for '{task_name}' took {end_time - start_time:.2f} seconds.")

    if isinstance(api_result, str) and "Error" in api_result:
        logger.error(f"{api_name.capitalize()} API returned error for '{task_name}': {api_result}")
    else:
        logger.info(f"{api_name.capitalize()} API completed for '{task_name}'.")
    return api_name, api_result


def process_document(file_path=None,
                     input_text_to_process=None,
                     is_synthesis_task=False,
                     deepseek_key=None,
                     grok_key=None,
                     chatgpt_key=None,
                     summary_level=50, # Đảm bảo đây là int
                     target_language_code=None):

    task_name = os.path.basename(file_path) if file_path else "Direct Input Text"
    if is_synthesis_task and file_path:
        task_name = f"Synthesis (based on {os.path.basename(file_path)}...)"
    elif is_synthesis_task and not file_path:
        task_name = "Input Text Synthesis"

    lang_str = f"Language: {target_language_code or 'Default'}"
    processing_type = "SYNTHESIS" if is_synthesis_task else "INDIVIDUAL SUMMARY"
    logger.info(f"Starting [{processing_type}]: {task_name} (Detail: {summary_level}%, {lang_str})")

    results = {'original_text': None, 'deepseek': None, 'grok': None, 'chatgpt': None, 'analysis': None, 'error': None}
    text_for_processing = None
    analysis_error_msg = None

    try:
        try:
            if input_text_to_process is not None:
                text_for_processing = input_text_to_process
                if not text_for_processing or not text_for_processing.strip():
                    raise ValueError("Direct input text is empty.")
                logger.info(f"Using direct input text for '{task_name}', {len(text_for_processing)} characters.")
            elif file_path:
                text_for_processing = extract_text(file_path)
                if not text_for_processing or not text_for_processing.strip():
                     raise ValueError("Text extraction returned empty.")
                logger.info(f"Extracted {len(text_for_processing)} characters from {task_name}.")
            else:
                raise ValueError("Missing input: Need file_path or input_text_to_process.")
            results['original_text'] = text_for_processing
        except (FileNotFoundError, PermissionError, IsADirectoryError, ValueError, IOError, RuntimeError) as ext_err:
            logger.error(f"Input/extraction error for '{task_name}': {ext_err}", exc_info=True)
            results['error'] = f"Input error: {ext_err}"
            results['analysis'] = {'error': f"Input error: {ext_err}"}
            return results

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
                analysis_error_msg = "Analysis errors: " + "; ".join(analysis_errors)
                logger.error(analysis_error_msg)
            else:
                logger.info(f"Text analysis for '{task_name}' completed.")
            results['analysis'] = analysis_data
        except Exception as analysis_exception:
            logger.error(f"Critical error analyzing text for '{task_name}': {analysis_exception}", exc_info=True)
            analysis_error_msg = f"System error during analysis: {analysis_exception}"
            results['analysis'] = {'error': analysis_error_msg}

        active_api_calls = []
        if deepseek_key:
            active_api_calls.append({'name': 'deepseek', 'func': summarize_with_deepseek, 'key': deepseek_key})
        if grok_key:
            active_api_calls.append({'name': 'grok', 'func': summarize_with_grok, 'key': grok_key})
        if chatgpt_key:
            active_api_calls.append({'name': 'chatgpt', 'func': summarize_with_chatgpt, 'key': chatgpt_key})

        if active_api_calls:
            try:
                from config.constants import MAX_CHUNK_SIZE
            except ImportError:
                MAX_CHUNK_SIZE = 40000 # Giá trị mặc định nếu không import được
                logger.warning(f"Could not import MAX_CHUNK_SIZE from constants, using default: {MAX_CHUNK_SIZE}")

            chunks = [text_for_processing[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text_for_processing), MAX_CHUNK_SIZE)]
            num_chunks = len(chunks)
            logger.debug(f"Chunks for API processing of '{task_name}': {num_chunks} (MAX_CHUNK_SIZE: {MAX_CHUNK_SIZE})")

            overall_api_start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_API_WORKERS) as executor:
                future_to_api_name = {
                    executor.submit(
                        _process_with_single_api,
                        call_info['name'],
                        call_info['func'],
                        call_info['key'],
                        text_for_processing, # Toàn bộ text, để hàm con có thể quyết định chunking
                        num_chunks, # Số chunk đã tính
                        chunks, # Danh sách chunk đã tạo
                        summary_level,
                        target_language_code,
                        is_synthesis_task,
                        task_name
                    ): call_info['name'] for call_info in active_api_calls
                }
                for future in concurrent.futures.as_completed(future_to_api_name):
                    api_name_res = future_to_api_name[future]
                    try:
                        returned_api_name, api_result = future.result() # _process_with_single_api trả về (tên, kết quả)
                        results[returned_api_name] = api_result
                    except Exception as exc:
                        logger.error(f"API call for {api_name_res} in '{task_name}' generated an exception: {exc}", exc_info=True)
                        results[api_name_res] = f"{api_name_res.capitalize()} Error: System error during parallel execution ({exc})"
            overall_api_end_time = time.time()
            logger.info(f"All API calls for '{task_name}' completed in {overall_api_end_time - overall_api_start_time:.2f} seconds.")

        else:
             logger.info(f"No API keys provided for '{task_name}'. Skipping summary/synthesis step.")
             results['deepseek'] = "<Not executed>"
             results['grok'] = "<Not executed>"
             results['chatgpt'] = "<Not executed>"

        if analysis_error_msg and not results.get('error'): # Ghi đè lỗi nếu có lỗi phân tích
            results['error'] = analysis_error_msg

        logger.info(f"Completed processing [{processing_type}] for: {task_name}")
        return results

    except Exception as e:
        logger.critical(f"Unexpected system error in process_document for {task_name}: {str(e)}", exc_info=True)
        error_msg = f"Unexpected system error: {e}"
        # Đảm bảo trả về đủ các key để UI không bị lỗi
        final_results = {'error': error_msg,
                         'original_text': text_for_processing if text_for_processing else "",
                         'deepseek': results.get('deepseek'),
                         'grok': results.get('grok'),
                         'chatgpt': results.get('chatgpt'),
                         'analysis': results.get('analysis', {'error': error_msg})}
        # Nếu analysis chưa có lỗi, gán lỗi vào
        if 'error' not in final_results['analysis']:
            final_results['analysis']['error'] = error_msg
        return final_results


# Hàm process_document_async và các hàm khác giữ nguyên cấu trúc
def process_document_async(file_path=None, input_text=None, settings=None, progress_callback=None):
    """Web-friendly async version with progress updates"""
    if progress_callback:
        progress_callback("Starting document processing...")

    try:
        settings = settings or {}
        actual_summary_level = int(settings.get('summary_level', 50)) # Đảm bảo là int

        if progress_callback:
            # TODO: Triển khai callback chi tiết hơn từ các bước trong process_document
            pass

        result = process_document(
            file_path=file_path,
            input_text_to_process=input_text,
            is_synthesis_task=settings.get('is_synthesis_task', False), # Đổi tên key cho nhất quán với main.py
            deepseek_key=settings.get('deepseek_key'),
            grok_key=settings.get('grok_key'),
            chatgpt_key=settings.get('chatgpt_key'),
            summary_level=actual_summary_level,
            target_language_code=settings.get('target_language_code') # Đổi tên key cho nhất quán với main.py
        )

        if progress_callback:
            if result.get('error'):
                progress_callback(f"Processing completed with errors: {result.get('error')}")
            else:
                progress_callback("Processing completed successfully")

        return result

    except Exception as e:
        error_msg = f"Async processing wrapper failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if progress_callback:
            progress_callback(error_msg)
        return {'error': error_msg, 'original_text': '', 'deepseek': None, 'grok': None, 'chatgpt': None, 'analysis': {'error': error_msg}}


def extract_text_preview(file_path, max_chars=1000): 
    """Quick text preview for web UI"""
    try:
        full_text = extract_text(file_path)
        if len(full_text) <= max_chars:
            return full_text
        return full_text[:max_chars] + "..."
    except Exception as e:
        logger.error(f"Error creating preview for {file_path}: {e}")
        return f"Error reading file: {str(e)}"


def validate_document_settings(settings): 
    """Validate processing settings for web UI"""
    required_fields = ['summary_level']
    missing_fields = [field for field in required_fields if field not in settings]

    if missing_fields:
        return {"valid": False, "errors": f"Missing fields: {', '.join(missing_fields)}"}

    summary_level = settings.get('summary_level')
    try:
        summary_level = int(summary_level)
        if not (10 <= summary_level <= 90):
            raise ValueError()
    except (ValueError, TypeError):
        return {"valid": False, "errors": "Summary level must be an integer between 10 and 90"}

    return {"valid": True, "errors": None}