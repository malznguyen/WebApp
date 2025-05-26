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
    if is_synthesis_prompt:
        task_description = f"""Synthesize the key information from the following combined text (derived from multiple documents on a similar topic). Provide a comprehensive overview that integrates the main points, arguments, and findings. The desired detail level for this synthesis is {summary_level}%. {language_instruction} Combined text to synthesize: --- START COMBINED TEXT --- {text_to_summarize[:110000]} --- END COMBINED TEXT ---"""
        system_role_content = "You are an expert AI assistant specializing in synthesizing information from multiple text sources..."
    else:
        task_description = f"""Create a summary for the following document with a detail level of {summary_level}%. {language_instruction} Document to summarize: --- START DOCUMENT --- {text_to_summarize[:110000]} --- END DOCUMENT ---"""
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
    
    if "authentication" in error_message or "401" in error_message or "incorrect api key" in error_message:
        return f"{api_name} Error: Invalid API key or authentication error."
    if "rate limit" in error_message or "limit" in error_message or "quota" in error_message or "429" in error_message:
        return f"{api_name} Error: Rate limit or quota exceeded."
    if "context_length_exceeded" in error_message:
        return f"{api_name} Error: Input text too long for model."
    if "400" in error_message:
         return f"{api_name} Error: Bad request (check parameters/data)."
    if "500" in error_message or "502" in error_message or "503" in error_message:
         return f"{api_name} Error: Temporary server error ({exception}). Try again later."
    
    return f"{api_name} API unknown error: {exception}"


def summarize_with_deepseek(text, api_key, summary_level=50, target_language_code=None, is_synthesis_prompt=False):
    if not api_key: return "DeepSeek Error: API Key not configured."
    if not text or not text.strip(): return "DeepSeek Error: Empty input text."
    
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
        response = client.with_options(timeout=120.0).chat.completions.create(
            model="deepseek-chat",
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
        response = client.with_options(timeout=120.0).chat.completions.create(
            model="grok-3-beta",
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
        response = client.with_options(timeout=180.0).chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_content},{"role": "user", "content": prompt_text}],
            max_tokens=10000, temperature=0.6
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


def process_document(file_path=None,
                     input_text_to_process=None,
                     is_synthesis_task=False,
                     deepseek_key=None,
                     grok_key=None,
                     chatgpt_key=None,
                     summary_level=50,
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

        if deepseek_key or grok_key or chatgpt_key:
            try:
                from config.constants import MAX_CHUNK_SIZE
            except ImportError: 
                MAX_CHUNK_SIZE = 250000
                
            chunks = [text_for_processing[i:i+MAX_CHUNK_SIZE] for i in range(0, len(text_for_processing), MAX_CHUNK_SIZE)]
            num_chunks = len(chunks)
            logger.debug(f"Chunks for API '{task_name}': {num_chunks}")

            api_calls = {
                'deepseek': (summarize_with_deepseek, deepseek_key), 
                'grok': (summarize_with_grok, grok_key), 
                'chatgpt': (summarize_with_chatgpt, chatgpt_key)
            }

            def summarize_chunks_and_combine(api_func, api_key_val, api_name_str, target_lang=None, for_synthesis=False):
                if not api_key_val: return f"{api_name_str} not configured."
                logger.info(f"Starting Map-Reduce with {api_name_str} for {num_chunks} chunks (Synthesis: {for_synthesis})...")
                
                chunk_summaries = []
                chunk_errors = 0
                for i, chunk_text in enumerate(chunks):
                    if not chunk_text or not chunk_text.strip(): continue
                    try:
                        summary_part = api_func(chunk_text, api_key_val, summary_level, target_lang, is_synthesis_prompt=for_synthesis)
                        if summary_part and isinstance(summary_part, str) and "Error" not in summary_part: 
                            chunk_summaries.append(summary_part)
                        else: 
                            logger.warning(f"Error/no summary chunk {i+1} from {api_name_str}: {summary_part}")
                            chunk_summaries.append(f"[[[Error chunk {i+1}: {summary_part}]]]")
                            chunk_errors += 1
                    except Exception as chunk_err: 
                        logger.error(f"Error calling {api_name_str} for chunk {i+1}: {chunk_err}", exc_info=True)
                        chunk_summaries.append(f"[[[Critical error chunk {i+1}]]]")
                        chunk_errors += 1
                        
                if not chunk_summaries or chunk_errors == num_chunks: 
                    return f"Error: No chunk summaries created with {api_name_str}."
                    
                combined_summary_text = "\n\n--- Next ---\n\n".join(chunk_summaries)
                logger.info(f"Combined {len(chunk_summaries)} chunk summaries from {api_name_str} ({chunk_errors} errors). Total characters: {len(combined_summary_text)}. Starting final pass...")
                
                if not combined_summary_text.strip(): 
                    return f"Error: Combined text from {api_name_str} is empty."
                    
                try:
                    final_summary_level_val = max(10, summary_level - 10)
                    final_output_text = api_func(combined_summary_text, api_key_val, final_summary_level_val, target_lang, is_synthesis_prompt=for_synthesis)
                    if chunk_errors > 0: 
                        final_output_text += f"\n\n(Note: {chunk_errors} errors occurred while processing text sections.)"
                    return final_output_text
                except Exception as final_err: 
                    logger.error(f"Error calling {api_name_str} final pass: {final_err}", exc_info=True)
                    return f"Critical error during final summary/synthesis with {api_name_str}."

            for result_key, (api_function, api_key_value) in api_calls.items():
                if api_key_value:
                    logger.info(f"Starting {result_key.capitalize()} API call for '{task_name}'...")
                    if num_chunks == 1:
                        api_result = api_function(chunks[0], api_key_value, summary_level, target_language_code, is_synthesis_prompt=is_synthesis_task)
                    else:
                        api_result = summarize_chunks_and_combine(api_function, api_key_value, result_key.capitalize(), target_language_code, for_synthesis=is_synthesis_task)
                    
                    if isinstance(api_result, str) and "Error" in api_result:
                        logger.error(f"{result_key.capitalize()} API returned error for '{task_name}': {api_result}")
                        results[result_key] = api_result
                    else:
                         results[result_key] = api_result
                         logger.info(f"{result_key.capitalize()} API completed for '{task_name}'.")
                else:
                    results[result_key] = None

        else:
             logger.info(f"No API keys provided for '{task_name}'. Skipping summary/synthesis step.")
             results['deepseek'] = "<Not executed>"
             results['grok'] = "<Not executed>"
             results['chatgpt'] = "<Not executed>"

        if analysis_error_msg and not results.get('error'):
            results['error'] = analysis_error_msg

        logger.info(f"Completed processing [{processing_type}] for: {task_name}")
        return results

    except Exception as e:
        logger.critical(f"Unexpected system error in process_document for {task_name}: {str(e)}", exc_info=True)
        error_msg = f"Unexpected system error: {e}"
        return {'error': error_msg,
                'original_text': text_for_processing,
                'deepseek': None, 'grok': None, 'chatgpt': None,
                'analysis': {'error': error_msg}}


def process_document_async(file_path=None, input_text=None, settings=None, progress_callback=None):
    """Web-friendly async version with progress updates"""
    if progress_callback:
        progress_callback("Starting document processing...")
    
    try:
        settings = settings or {}
        
        if progress_callback:
            progress_callback("Analyzing text structure...")
        
        result = process_document(
            file_path=file_path,
            input_text_to_process=input_text,
            is_synthesis_task=settings.get('is_synthesis', False),
            deepseek_key=settings.get('deepseek_key'),
            grok_key=settings.get('grok_key'),
            chatgpt_key=settings.get('chatgpt_key'),
            summary_level=settings.get('summary_level', 50),
            target_language_code=settings.get('target_language')
        )
        
        if progress_callback:
            if result.get('error'):
                progress_callback(f"Processing completed with errors")
            else:
                progress_callback("Processing completed successfully")
        
        return result
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        if progress_callback:
            progress_callback(error_msg)
        return {'error': error_msg}


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
    if not isinstance(summary_level, int) or not (10 <= summary_level <= 90):
        return {"valid": False, "errors": "Summary level must be between 10 and 90"}
    
    return {"valid": True, "errors": None}