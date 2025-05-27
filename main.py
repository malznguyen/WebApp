import eel
import sys
import os
import json
import base64
import tempfile
import threading
import time
import traceback 
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# --- Backend Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'BE')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# --- Core Imports ---
try:
    from core.search_thread import SearchThread, search_image_sync
    from core.document_api import (
        process_document,
        extract_text_preview,
        extract_text,
        process_batch_synthesis
    )
    from core.image_processing import process_web_upload, validate_image_upload
    from core.api_client import validate_api_keys
    from config.settings import (
        SERP_API_KEY, IMGUR_CLIENT_ID, DEEPSEEK_API_KEY, GROK_API_KEY,
        CHATGPT_API_KEY, WINDOW_WIDTH, WINDOW_HEIGHT
    )
    from utils.logger import setup_logger
    from utils.helpers import ensure_dir_exists
    print("✅ All backend imports successful!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Ensure all backend modules (BE directory) are correctly placed and have no internal errors.")
    sys.exit(1)

# --- Globals & Initialization ---
logger = setup_logger()
eel.init('FE')

active_searches: Dict[str, threading.Thread] = {}
active_processing_tasks: Dict[str, threading.Thread] = {}
app_config: Optional[Dict[str, Any]] = None

# ===== UTILITY FUNCTIONS =====

def safe_int(value: Any, default: int = 0, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Safely convert a value to an integer, with optional clamping."""
    try:
        result = int(value)
        if min_val is not None: result = max(min_val, result)
        if max_val is not None: result = min(max_val, result)
        return result
    except (ValueError, TypeError):
        return default

def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert a value to a boolean."""
    if isinstance(value, bool): return value
    if isinstance(value, str): return value.lower() in ('true', '1', 'yes', 'on')
    try: return bool(value)
    except: return default

def validate_base64_data(data: str, expected_prefix: Optional[str] = None) -> bytes:
    """Validates and decodes base64 encoded data."""
    if not data or not isinstance(data, str):
        raise ValueError("Invalid or empty base64 data provided.")
    try:
        actual_data = data
        if data.startswith('data:'):
            if ',' not in data: raise ValueError("Invalid base64 data URL format: missing comma separator.")
            header, actual_data = data.split(',', 1)
            if expected_prefix and not header.startswith(expected_prefix):
                raise ValueError(f"Invalid data URL prefix. Expected '{expected_prefix}', got '{header}'.")
        decoded_bytes = base64.b64decode(actual_data)
        if not decoded_bytes: raise ValueError("Decoded base64 data is empty.")
        return decoded_bytes
    except base64.binascii.Error as b64_err:
        raise ValueError(f"Base64 decoding failed: {b64_err}")
    except Exception as e:
        raise ValueError(f"Error processing base64 data: {e}")

def create_temp_file(data: bytes, filename: str, suffix: Optional[str] = None) -> str:
    """Creates a temporary file. The file is not auto-deleted."""
    temp_file_path: Optional[str] = None
    try:
        if not data: raise ValueError("Cannot create temp file with empty data.")
        if not filename: raise ValueError("Filename hint cannot be empty for creating temp file.")
        temp_dir = ensure_dir_exists('temp')
        original_suffix = os.path.splitext(filename)[1] if '.' in filename else '.tmp'
        actual_suffix = suffix if suffix and isinstance(suffix, str) else original_suffix
        if not actual_suffix.startswith('.'): actual_suffix = '.' + actual_suffix
        base_name_part = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in os.path.splitext(os.path.basename(filename))[0])
        prefix = f"{base_name_part}_" if base_name_part else "uploaded_file_"
        with tempfile.NamedTemporaryFile(delete=False, suffix=actual_suffix, dir=temp_dir, prefix=prefix) as temp_file_obj:
            temp_file_obj.write(data)
            temp_file_path = temp_file_obj.name
        if not temp_file_path: raise IOError("Failed to obtain temporary file path after creation.")
        return temp_file_path
    except Exception as e:
        if temp_file_path and os.path.exists(temp_file_path): cleanup_temp_file(temp_file_path)
        logger.error(f"Failed to create temporary file for '{filename}': {e}", exc_info=True)
        raise IOError(f"Failed to create temporary file for '{filename}': {e}")

def cleanup_temp_file(file_path: str) -> None:
    """Safely cleans up (deletes) a temporary file if it exists."""
    try:
        if file_path and isinstance(file_path, str) and os.path.isfile(file_path):
            os.unlink(file_path)
            logger.debug(f"Successfully cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Could not delete temporary file '{file_path}': {e}", exc_info=True)

def validate_file_path(file_path: str, check_read_access: bool = True) -> bool:
    """Validates if a file path exists, is a file, and optionally is readable."""
    if not file_path or not isinstance(file_path, str): return False
    try:
        path_is_file = os.path.isfile(file_path)
        if not path_is_file: return False
        if check_read_access and not os.access(file_path, os.R_OK): return False
        return True
    except Exception as e:
        logger.debug(f"Error validating file path '{file_path}': {e}")
        return False

# ===== INITIALIZATION & CONFIGURATION =====
@eel.expose
def get_app_config() -> Dict[str, Any]:
    """Retrieves and returns the application's configuration for the frontend."""
    global app_config
    if app_config is None:
        try:
            api_keys_validation = validate_api_keys()
            app_config = {
                'status': 'ready',
                'has_serp_api': api_keys_validation.get('has_serp_api', False),
                'has_imgur': api_keys_validation.get('has_imgur', False),
                'has_deepseek': bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY.strip()),
                'has_grok': bool(GROK_API_KEY and GROK_API_KEY.strip()),
                'has_chatgpt': bool(CHATGPT_API_KEY and CHATGPT_API_KEY.strip()),
                'supported_formats': ['.pdf', '.docx', '.txt', '.md'],
                'version': '2.1.0',
                'ready': api_keys_validation.get('ready', False),
                'missing_keys': api_keys_validation.get('missing_keys', [])
            }
            logger.info(f"Application configuration loaded. Overall ready: {app_config['ready']}.")
        except Exception as e:
            logger.error(f"Critical error loading application config: {e}", exc_info=True)
            app_config = {
                'status': 'error', 'error': str(e), 'ready': False,
                'has_serp_api': False, 'has_imgur': False, 'has_deepseek': False,
                'has_grok': False, 'has_chatgpt': False, 'supported_formats': [],
                'version': 'N/A', 'missing_keys': ['Configuration load failed']
            }
    return app_config

# ===== IMAGE SEARCH FUNCTIONS =====
@eel.expose
def search_image_web(image_data_base64: str, filename: str, social_media_only: bool = False) -> Dict[str, Any]:
    """Handles image search requests from the web UI."""
    temp_image_path: Optional[str] = None
    try:
        if not image_data_base64 or not filename:
            raise ValueError("Image data or filename is missing for search operation.")
        is_social_only_search = safe_bool(social_media_only, False)
        logger.info(f"Initiating image search for '{filename}'. Social media only: {is_social_only_search}.")
        image_bytes = validate_base64_data(image_data_base64, expected_prefix="data:image")
        upload_validation = validate_image_upload(image_bytes)
        if not upload_validation.get('valid', False):
            raise ValueError(upload_validation.get('error', 'Uploaded image is invalid.'))
        temp_image_path = create_temp_file(image_bytes, filename)
        cfg = get_app_config()
        if not cfg.get('has_serp_api') or not cfg.get('has_imgur'):
             raise EnvironmentError("SERP API key or Imgur Client ID is not configured. Cannot perform image search.")
        search_results_list = search_image_sync(
            image_path=temp_image_path, serp_api_key=SERP_API_KEY,
            imgur_client_id=IMGUR_CLIENT_ID, social_media_only=is_social_only_search,
            timeout=120
        )
        found_count = len(search_results_list) if search_results_list else 0
        logger.info(f"Image search for '{filename}' completed. Found {found_count} results.")
        return {'success': True, 'results': search_results_list or [], 'total': found_count, 'social_only': is_social_only_search}
    except ValueError as ve:
        logger.error(f"Image search validation error for '{filename}': {ve}", exc_info=False)
        return {'success': False, 'error': str(ve), 'results': [], 'total': 0}
    except EnvironmentError as ee:
        logger.error(f"Image search environment/configuration error for '{filename}': {ee}", exc_info=False)
        return {'success': False, 'error': str(ee), 'results': [], 'total': 0}
    except Exception as e:
        logger.error(f"Unexpected error during image search for '{filename}': {e}", exc_info=True)
        return {'success': False, 'error': f"An unexpected error occurred: {e}", 'results': [], 'total': 0}
    finally:
        if temp_image_path: cleanup_temp_file(temp_image_path)

@eel.expose
def search_image_async_web(image_data_base64: str, filename: str, social_media_only: bool = False) -> str:
    """Starts an asynchronous image search operation."""
    search_id = f"search_{int(time.time())}_{threading.get_ident()}"
    def run_search_in_thread():
        def progress_update(percent, message):
            if eel: eel.searchProgress(search_id, percent, message)()
        try:
            progress_update(0, f"Starting async search for {filename}...")
            result_dict = search_image_web(image_data_base64, filename, social_media_only)
            if result_dict['success']:
                progress_update(100, f"Search for {filename} complete. Found {result_dict['total']} results.")
                eel.searchComplete(search_id, result_dict['results'])()
            else:
                progress_update(100, f"Search for {filename} failed: {result_dict['error']}")
                eel.searchError(search_id, "Search Operation Failed", result_dict['error'])()
        except Exception as e:
            logger.error(f"Async image search thread for {filename} (ID: {search_id}) failed: {e}", exc_info=True)
            try:
                progress_update(100, f"System error during search for {filename}.")
                eel.searchError(search_id, "System Error in Search Thread", str(e))()
            except Exception as eel_err: logger.error(f"Failed to send error to frontend for search ID {search_id}: {eel_err}")
        finally: active_searches.pop(search_id, None)
    search_thread = threading.Thread(target=run_search_in_thread, daemon=True)
    active_searches[search_id] = search_thread
    search_thread.start()
    logger.info(f"Asynchronous image search started for '{filename}' with ID: {search_id}")
    return search_id

# ===== DOCUMENT PROCESSING FUNCTIONS =====
def validate_processing_settings(settings_from_frontend: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and normalizes document processing settings received from the frontend."""
    normalized_settings: Dict[str, Any] = {}
    try:
        ai_models_selection = settings_from_frontend.get('ai_models', {})
        if not isinstance(ai_models_selection, dict): ai_models_selection = {}
        normalized_settings['deepseek_key'] = DEEPSEEK_API_KEY if safe_bool(ai_models_selection.get('deepseek')) and DEEPSEEK_API_KEY else None
        normalized_settings['grok_key'] = GROK_API_KEY if safe_bool(ai_models_selection.get('grok')) and GROK_API_KEY else None
        normalized_settings['chatgpt_key'] = CHATGPT_API_KEY if safe_bool(ai_models_selection.get('chatgpt')) and CHATGPT_API_KEY else None
        normalized_settings['summary_level'] = safe_int(settings_from_frontend.get('detail_level', 50), default=50, min_val=10, max_val=90)
        lang_code = settings_from_frontend.get('language')
        normalized_settings['target_language_code'] = lang_code if lang_code and isinstance(lang_code, str) and lang_code.strip() else None
        
        processing_mode = settings_from_frontend.get('processing_mode', 'individual')
        normalized_settings['is_synthesis_task'] = (processing_mode == 'batch')

        return normalized_settings
    except Exception as e:
        logger.error(f"Error validating processing settings: {e}", exc_info=True)
        return {
            'deepseek_key': None, 'grok_key': None, 'chatgpt_key': None,
            'summary_level': 50, 'target_language_code': None, 'is_synthesis_task': False
        }

@eel.expose
def get_document_text_on_upload(file_data_base64: str, filename: str) -> Dict[str, Any]:
    """Extracts text from an uploaded document for immediate UI preview."""
    temp_doc_path: Optional[str] = None
    try:
        if not file_data_base64 or not filename:
            raise ValueError("File data or filename is missing for text preview.")
        logger.info(f"Requesting immediate text extraction for preview: '{filename}'.")
        doc_bytes = validate_base64_data(file_data_base64)
        MAX_PREVIEW_SIZE_BYTES = 50 * 1024 * 1024
        if len(doc_bytes) > MAX_PREVIEW_SIZE_BYTES:
            raise ValueError(f"File '{filename}' is too large for immediate preview (max {MAX_PREVIEW_SIZE_BYTES // (1024*1024)}MB).")
        temp_doc_path = create_temp_file(doc_bytes, filename, suffix=os.path.splitext(filename)[1] or '.tmp')
        extracted_text_content = extract_text(temp_doc_path)
        char_count = len(extracted_text_content) if extracted_text_content else 0
        logger.info(f"Successfully extracted {char_count} characters for preview from '{filename}'.")
        return {'success': True, 'text_content': extracted_text_content or "", 'filename': filename}
    except ValueError as ve:
        logger.warning(f"Validation error during text preview for '{filename}': {ve}")
        return {'success': False, 'error': str(ve), 'text_content': ""}
    except IOError as ioe:
        logger.error(f"File I/O error during text preview for '{filename}': {ioe}", exc_info=True)
        return {'success': False, 'error': f"File system error: {ioe}", 'text_content': ""}
    except Exception as e:
        logger.error(f"Unexpected error extracting text preview for '{filename}': {e}", exc_info=True)
        return {'success': False, 'error': f"Extraction error: {e}", 'text_content': ""}
    finally:
        if temp_doc_path: cleanup_temp_file(temp_doc_path)

@eel.expose
def process_document_web(doc_file_path: str, settings_from_frontend: Dict[str, Any]) -> Dict[str, Any]:
    """Processes a single document specified by its path using given settings."""
    file_basename = os.path.basename(doc_file_path) if doc_file_path else "Unknown Document"
    try:
        if not validate_file_path(doc_file_path):
            raise ValueError(f"Invalid or inaccessible document path provided: {doc_file_path}")
        current_settings = settings_from_frontend if isinstance(settings_from_frontend, dict) else {}
        logger.info(f"Starting single document processing for '{file_basename}' (Path: {doc_file_path}).")
        normalized_processing_settings = validate_processing_settings(current_settings)

        is_single_doc_synthesis = normalized_processing_settings.pop('is_synthesis_task', False)

        if not any(normalized_processing_settings.get(key) for key in ['deepseek_key', 'grok_key', 'chatgpt_key']):
            logger.warning(f"No AI models configured/enabled for '{file_basename}'. Only text analysis if extraction succeeds.")

        processing_result = process_document(
            file_path=doc_file_path,
            is_synthesis_task=is_single_doc_synthesis,
            **normalized_processing_settings
        )

        if not processing_result or not isinstance(processing_result, dict):
            raise SystemError(f"Core document processing for '{file_basename}' returned an invalid result.")
        logger.info(f"Single document processing completed for '{file_basename}'.")

        ai_results_output = []
        for model_key, model_name in [('deepseek', 'DeepSeek'), ('grok', 'Grok'), ('chatgpt', 'ChatGPT')]:
            content = processing_result.get(model_key)
            if content is not None and not (isinstance(content, str) and content.startswith("<Not executed")):
                is_error_msg = isinstance(content, str) and ("Error:" in content or "failed" in content.lower())
                ai_results_output.append({'model': model_name, 'content': content, 'is_error': is_error_msg})
        
        overall_error_message = processing_result.get('error')
        has_any_errors = bool(overall_error_message) or any(res.get('is_error') for res in ai_results_output if isinstance(res, dict))
        return {
            'success': not bool(overall_error_message),
            'original_text': processing_result.get('original_text', ''),
            'ai_results': ai_results_output,
            'analysis': processing_result.get('analysis', {}),
            'error': overall_error_message,
            'has_errors': has_any_errors
        }
    except ValueError as ve:
        logger.error(f"Input validation error processing document '{file_basename}': {ve}", exc_info=False)
        return {'success': False, 'error': str(ve), 'original_text': '', 'ai_results': [], 'analysis': {}, 'has_errors': True}
    except SystemError as se:
        logger.error(f"System error processing document '{file_basename}': {se}", exc_info=True)
        return {'success': False, 'error': str(se), 'original_text': '', 'ai_results': [], 'analysis': {}, 'has_errors': True}
    except Exception as e:
        logger.error(f"Unexpected error processing document '{file_basename}': {e}", exc_info=True)
        return {'success': False, 'error': f"An unexpected error occurred: {e}", 'original_text': '', 'ai_results': [], 'analysis': {}, 'has_errors': True}

@eel.expose
def process_documents_batch_web(temp_file_paths: List[str], settings_from_frontend: Dict[str, Any]) -> Dict[str, Any]:
    """Processes a batch of documents (specified by their temporary file paths) for synthesis."""
    num_docs = len(temp_file_paths)
    logger.info(f"Received request to process a batch of {num_docs} documents.")
    if not temp_file_paths:
        return {'success': False, 'error': "No file paths provided for batch processing.", 'ai_results': [], 'processed_files': [], 'failed_files': []}
    try:
        valid_paths_for_batch = []
        failed_file_validations = []
        for i, path in enumerate(temp_file_paths):
            if validate_file_path(path):
                valid_paths_for_batch.append(path)
            else:
                filename = os.path.basename(path) if path else f"UnknownFile_{i}"
                failed_file_validations.append((filename, "Invalid or inaccessible path"))
                logger.warning(f"Invalid path in batch: '{path}' for file '{filename}'. Skipping.")

        if not valid_paths_for_batch:
            return {'success': False, 'error': "None of the provided file paths were valid for batch processing.", 'ai_results': [], 'processed_files': [], 'failed_files': failed_file_validations}

        current_settings = settings_from_frontend if isinstance(settings_from_frontend, dict) else {}
        normalized_settings = validate_processing_settings(current_settings)

        batch_result = process_batch_synthesis(
            file_paths=valid_paths_for_batch,
            deepseek_key=normalized_settings.get('deepseek_key'),
            grok_key=normalized_settings.get('grok_key'),
            chatgpt_key=normalized_settings.get('chatgpt_key'),
            summary_level=normalized_settings.get('summary_level', 50),
            target_language_code=normalized_settings.get('target_language_code')
        )

        ai_batch_syntheses = []
        for model_key_suffix, model_name in [('_synthesis', 'DeepSeek'), ('_synthesis', 'Grok'), ('_synthesis', 'ChatGPT')]:
            full_model_key = model_name.lower().replace(" ", "") + model_key_suffix
            content = batch_result.get(full_model_key)
            if content is not None and not (isinstance(content, str) and content.startswith("<Not executed")):
                is_error_msg = isinstance(content, str) and "Error:" in content
                ai_batch_syntheses.append({'model': model_name, 'content': content, 'is_error': is_error_msg})

        overall_batch_error = batch_result.get('overall_error')
        has_any_batch_errors = bool(overall_batch_error) or any(res.get('is_error') for res in ai_batch_syntheses if isinstance(res, dict))

        logger.info(f"Batch synthesis processing completed for {len(valid_paths_for_batch)} valid documents.")
        return {
            'success': not bool(overall_batch_error),
            'ai_results': ai_batch_syntheses,
            'processed_files': batch_result.get('processed_files', []),
            'failed_files': batch_result.get('failed_files', []) + failed_file_validations,
            'concatenated_text_char_count': batch_result.get('concatenated_text_char_count', 0),
            'error': overall_batch_error,
            'has_errors': has_any_batch_errors
        }
    except Exception as e:
        logger.error(f"Unexpected error during batch document processing: {e}", exc_info=True)
        return {
            'success': False, 'error': f"An unexpected error occurred during batch processing: {e}",
            'ai_results': [], 'processed_files': [],
            'failed_files': [(os.path.basename(p) if p else "Unknown", "Batch system error") for p in temp_file_paths],
            'has_errors': True
        }

@eel.expose
def process_document_async_web(file_input: Union[str, Dict[str, Any], List[Dict[str, Any]]], settings: Dict[str, Any]) -> str:
    """Starts an asynchronous document processing task (single file, batch files, or direct text)."""
    process_id = f"doc_process_{int(time.time())}_{threading.get_ident()}"

    def run_doc_processing_in_thread():
        temp_file_for_this_task: Optional[str] = None
        temp_files_for_batch: List[str] = []
        log_filename = "document_processing_task"
        result_dict: Dict[str, Any] = {}

        try:
            normalized_settings = validate_processing_settings(settings)
            is_batch_mode_from_settings = normalized_settings.get('is_synthesis_task', False)

            is_batch_operation = isinstance(file_input, list)

            if is_batch_operation:
                if not all(isinstance(item, dict) and 'file_data' in item and 'filename' in item for item in file_input):
                    raise ValueError("For batch mode, 'file_input' must be a list of {file_data, filename} dictionaries.")
                
                log_filename = f"{len(file_input)} documents in batch"
                eel.processingProgress(process_id, 5, f"Preparing batch: {log_filename}...")()
                
                failed_uploads = []
                for i, item_dict in enumerate(file_input):
                    eel.processingProgress(process_id, 10 + int(i/len(file_input)*20), f"Uploading {item_dict.get('filename','file ' + str(i+1))}...")()
                    upload_resp = upload_file_web(item_dict['file_data'], item_dict['filename'])
                    if upload_resp['success'] and upload_resp['temp_path']:
                        temp_files_for_batch.append(upload_resp['temp_path'])
                    else:
                        failed_uploads.append(item_dict.get('filename','file ' + str(i+1)))
                
                if failed_uploads:
                    raise IOError(f"Upload failed for some files in batch: {', '.join(failed_uploads)}")
                if not temp_files_for_batch:
                    raise ValueError("No files successfully uploaded for batch processing.")
                
                eel.processingProgress(process_id, 30, f"Starting backend batch processing for {log_filename}...")()
                result_dict = process_documents_batch_web(temp_files_for_batch, normalized_settings)

            elif isinstance(file_input, dict) and 'direct_text_content' in file_input:
                log_filename = file_input.get('text_input_name', 'Direct Text Input')
                text_content_to_process = file_input['direct_text_content']
                eel.processingProgress(process_id, 5, f"Preparing to process direct text: {log_filename}...")()
                if not text_content_to_process or not text_content_to_process.strip():
                    raise ValueError("Direct text input cannot be empty.")
                
                eel.processingProgress(process_id, 30, f"Starting backend processing for {log_filename}...")()
                
                # Call process_document from document_api
                raw_result = process_document(
                    input_text_to_process=text_content_to_process,
                    file_path=None,
                    is_synthesis_task=normalized_settings.get('is_synthesis_task', False),
                    deepseek_key=normalized_settings.get('deepseek_key'),
                    grok_key=normalized_settings.get('grok_key'),
                    chatgpt_key=normalized_settings.get('chatgpt_key'),
                    summary_level=normalized_settings.get('summary_level', 50),
                    target_language_code=normalized_settings.get('target_language_code')
                )
                
                # Transform to expected frontend format
                ai_results_output = []
                for model_key, model_name in [('deepseek', 'DeepSeek'), ('grok', 'Grok'), ('chatgpt', 'ChatGPT')]:
                    content = raw_result.get(model_key)
                    if content is not None and not (isinstance(content, str) and content.startswith("<Not executed")):
                        is_error_msg = isinstance(content, str) and ("Error:" in content or "failed" in content.lower())
                        ai_results_output.append({'model': model_name, 'content': content, 'is_error': is_error_msg})
                
                overall_error_message = raw_result.get('error')
                has_any_errors = bool(overall_error_message) or any(res.get('is_error') for res in ai_results_output if isinstance(res, dict))
                
                result_dict = {
                    'success': not bool(overall_error_message),
                    'original_text': raw_result.get('original_text', ''),
                    'ai_results': ai_results_output,
                    'analysis': raw_result.get('analysis', {}),
                    'error': overall_error_message,
                    'has_errors': has_any_errors
                }

            elif isinstance(file_input, str):
                log_filename = os.path.basename(file_input)
                eel.processingProgress(process_id, 5, f"Preparing to process file: {log_filename}...")()
                result_dict = process_document_web(file_input, settings)

            elif isinstance(file_input, dict) and 'file_data' in file_input and 'filename' in file_input:
                log_filename = file_input['filename']
                eel.processingProgress(process_id, 10, f"Uploading {log_filename} for processing...")()
                upload_resp = upload_file_web(file_input['file_data'], log_filename)
                if not upload_resp['success'] or not upload_resp['temp_path']:
                    raise IOError(f"Upload failed for async processing of '{log_filename}': {upload_resp.get('error', 'Unknown upload error')}")
                temp_file_for_this_task = upload_resp['temp_path']
                eel.processingProgress(process_id, 30, f"Starting backend processing for {log_filename}...")()
                result_dict = process_document_web(temp_file_for_this_task, settings)
                
            else:
                raise ValueError(
                    "Invalid 'file_input' type. Expected path string, {file_data, filename} dict for single file, "
                    "a list of such dicts for batch, or {direct_text_content, text_input_name} for direct text."
                )

            eel.processingProgress(process_id, 100, f"Finalizing results for {log_filename}...")()
            eel.processingComplete(process_id, result_dict)()
        except Exception as e:
            logger.error(f"Async document processing for '{log_filename}' (ID: {process_id}) failed in thread: {e}", exc_info=True)
            try:
                eel.processingError(process_id, f"Error processing '{log_filename}': {str(e)}")()
            except Exception as eel_err:
                 logger.error(f"Failed to send error to frontend for doc process ID {process_id}: {eel_err}")
        finally:
            if temp_file_for_this_task:
                cleanup_temp_file(temp_file_for_this_task)
            if temp_files_for_batch:
                logger.info(f"Async batch task {process_id} finished. Cleaning up {len(temp_files_for_batch)} temp files from batch upload.")
                for path_to_clean in temp_files_for_batch:
                    cleanup_temp_file(path_to_clean)
            
            active_processing_tasks.pop(process_id, None)

    doc_process_thread = threading.Thread(target=run_doc_processing_in_thread, daemon=True)
    active_processing_tasks[process_id] = doc_process_thread
    doc_process_thread.start()
    logger.info(f"Asynchronous document processing task started with ID: {process_id} (Input type: {type(file_input)})")
    return process_id

# ===== FILE HANDLING & PREVIEWS (Web Exposed) =====
@eel.expose
def upload_file_web(file_data_base64: str, filename: str) -> Dict[str, Any]:
    """Handles file uploads, saves to temp, returns path for further backend use."""
    temp_upload_path: Optional[str] = None
    try:
        if not file_data_base64 or not filename:
            raise ValueError("File data or filename is missing for upload.")
        if len(filename) > 255:
            raise ValueError("Filename is too long (max 255 characters).")
        logger.info(f"Processing upload request for file: '{filename}'.")
        file_bytes = validate_base64_data(file_data_base64)
        MAX_UPLOAD_SIZE_MB = 50
        if len(file_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File '{filename}' exceeds upload size limit of {MAX_UPLOAD_SIZE_MB}MB.")
        temp_upload_path = create_temp_file(file_bytes, filename, suffix=os.path.splitext(filename)[1] or '.bin')
        logger.info(f"File '{filename}' uploaded to temp path: '{temp_upload_path}' (Size: {len(file_bytes)} bytes).")
        return {
            'success': True, 'temp_path': temp_upload_path,
            'size': len(file_bytes), 'filename': os.path.basename(temp_upload_path)
        }
    except ValueError as ve:
        logger.warning(f"Upload validation error for '{filename}': {ve}")
        return {'success': False, 'error': str(ve), 'temp_path': None}
    except IOError as ioe:
        logger.error(f"File I/O error during upload of '{filename}': {ioe}", exc_info=True)
        return {'success': False, 'error': f"File system error during upload: {ioe}", 'temp_path': None}
    except Exception as e:
        if temp_upload_path and os.path.exists(temp_upload_path): cleanup_temp_file(temp_upload_path)
        logger.error(f"Unexpected error during upload of '{filename}': {e}", exc_info=True)
        return {'success': False, 'error': f"An unexpected error occurred during upload: {e}", 'temp_path': None}

@eel.expose
def get_document_preview(doc_file_path: str, max_chars: int = 1000) -> Dict[str, Any]:
    """Generates a short text preview for a document at the given path."""
    file_basename = os.path.basename(doc_file_path) if doc_file_path else "Unknown Document"
    try:
        if not validate_file_path(doc_file_path):
            raise ValueError(f"Invalid or inaccessible path for document preview: {doc_file_path}")
        max_preview_chars = safe_int(max_chars, default=1000, min_val=100, max_val=5000)
        preview_content = extract_text_preview(doc_file_path, max_preview_chars)
        logger.info(f"Generated preview for '{file_basename}' (max {max_preview_chars} chars).")
        return {'success': True, 'preview': preview_content or f"No preview could be extracted from '{file_basename}'."}
    except ValueError as ve:
        logger.warning(f"Preview generation validation error for '{file_basename}': {ve}")
        return {'success': False, 'error': str(ve), 'preview': ''}
    except Exception as e:
        logger.error(f"Unexpected error generating preview for '{file_basename}': {e}", exc_info=True)
        return {'success': False, 'error': f"Error generating preview: {e}", 'preview': ''}

# ===== SYSTEM STATUS & UTILITIES (Web Exposed) =====
@eel.expose
def get_system_status() -> Dict[str, Any]:
    """Provides a snapshot of the system's status to the frontend."""
    try:
        temp_dir = 'temp'; temp_files_on_disk = 0
        if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
            temp_files_on_disk = len([f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))])
        live_searches = sum(1 for t in active_searches.values() if t.is_alive())
        live_processing = sum(1 for t in active_processing_tasks.values() if t.is_alive())
        return {
            'success': True, 'active_searches': live_searches, 'active_processing_tasks': live_processing,
            'temp_files_on_disk': temp_files_on_disk, 'python_version': sys.version.split()[0],
            'working_directory': os.getcwd(),
            'eel_server_port': eel.SERVER_PORT if hasattr(eel, 'SERVER_PORT') else 'N/A',
            'backend_loaded': True
        }
    except Exception as e:
        logger.error(f"Failed to retrieve system status: {e}", exc_info=True)
        return {
            'success': False, 'error': str(e), 'active_searches': 0,
            'active_processing_tasks': 0, 'temp_files_on_disk': 0, 'backend_loaded': False
        }

@eel.expose
def perform_cleanup_temp_files() -> Dict[str, Any]:
    """Cleans up all files within the 'temp' directory."""
    temp_dir_path = 'temp'; cleaned_files_count = 0; errors_list = []
    try:
        logger.info(f"Initiating cleanup of temporary files in '{temp_dir_path}'.")
        if os.path.exists(temp_dir_path) and os.path.isdir(temp_dir_path):
            for item_name in os.listdir(temp_dir_path):
                item_path = os.path.join(temp_dir_path, item_name)
                if os.path.isfile(item_path):
                    try: os.unlink(item_path); cleaned_files_count += 1
                    except Exception as e_file:
                        err_detail = f"Could not delete temp file '{item_path}': {e_file}"
                        logger.warning(err_detail); errors_list.append(err_detail)
            message = f"Cleanup complete. {cleaned_files_count} files deleted."
            if errors_list: message += f" Encountered {len(errors_list)} errors."
            logger.info(message)
            return {'success': True, 'cleaned_count': cleaned_files_count, 'errors': errors_list}
        else:
            logger.info(f"Temporary directory '{temp_dir_path}' not found. No files to clean.")
            return {'success': True, 'cleaned_count': 0, 'errors': [], 'message': "Temp directory not found."}
    except Exception as e_main:
        logger.error(f"Major error during temporary files cleanup: {e_main}", exc_info=True)
        return {'success': False, 'error': str(e_main), 'cleaned_count': 0, 'errors': [str(e_main)]}

@eel.expose
def save_user_settings(settings_map: Dict[str, Any]) -> Dict[str, Any]:
    """Saves user preferences to a JSON file."""
    settings_file = 'user_settings.json'
    try:
        if not isinstance(settings_map, dict):
            raise ValueError("Invalid settings format: input must be a dictionary.")
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings_map, f, indent=4, ensure_ascii=False)
        logger.info(f"User settings successfully saved to '{settings_file}'.")
        return {'success': True, 'message': 'Settings saved.'}
    except ValueError as ve:
        logger.error(f"Failed to save settings due to invalid data: {ve}", exc_info=False)
        return {'success': False, 'error': str(ve)}
    except Exception as e:
        logger.error(f"Failed to save user settings to '{settings_file}': {e}", exc_info=True)
        return {'success': False, 'error': f"Could not save settings: {e}"}

@eel.expose
def load_user_settings() -> Dict[str, Any]:
    """Loads user preferences from a JSON file."""
    settings_file = 'user_settings.json'
    try:
        if os.path.isfile(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f: loaded_data = json.load(f)
            if not isinstance(loaded_data, dict):
                raise TypeError("Settings file content is not a valid JSON object (dictionary).")
            logger.info(f"User settings loaded from '{settings_file}'.")
            return {'success': True, 'settings': loaded_data}
        else:
            logger.info(f"User settings file '{settings_file}' not found. Returning empty defaults.")
            return {'success': True, 'settings': {}}
    except (json.JSONDecodeError, TypeError) as json_err:
        logger.error(f"Error decoding settings file '{settings_file}': {json_err}", exc_info=True)
        return {'success': False, 'error': f"Settings file is corrupted or invalid: {json_err}", 'settings': {}}
    except Exception as e:
        logger.error(f"Failed to load user settings from '{settings_file}': {e}", exc_info=True)
        return {'success': False, 'error': f"Could not load settings: {e}", 'settings': {}}

# ===== APPLICATION STARTUP LOGIC =====
def start_app():
    """Initializes and starts the Eel-based web application."""
    try:
        ensure_dir_exists('temp'); ensure_dir_exists('logs')
        initial_app_config = get_app_config()
        logger.info(f"🚀 Enhanced Toolkit v{initial_app_config.get('version', 'N/A')} starting with Python {sys.version.split()[0]}...")
        logger.info(f"Initial configuration status: {initial_app_config.get('status', 'unknown')}. API readiness: {initial_app_config.get('ready', False)}.")
        if not initial_app_config.get('ready', False):
            missing_api_keys = initial_app_config.get('missing_keys', [])
            if missing_api_keys: logger.warning(f"⚠️ Application may operate in a limited mode. Essential API keys missing: {', '.join(missing_api_keys)}")
            else: logger.warning("⚠️ Application readiness check failed, but no specific missing keys reported. Check config.")
        if not try_start_browser():
            logger.critical("❌ Application launch failed after all attempts. Please review logs.")
            input("Press Enter to exit..."); sys.exit(1)
    except Exception as e:
        logger.critical(f"❌ Fatal error during application startup: {e}", exc_info=True)
        input("Press Enter to exit..."); sys.exit(1)

def try_start_browser() -> bool:
    """Attempts to start Eel with various browser modes, falling back to server-only."""
    eel_start_options = {
        'size': (WINDOW_WIDTH, WINDOW_HEIGHT), 'position': (50, 50),
        'disable_cache': True, 'port': 0, 'host': 'localhost'
    }
    browser_preferences = [
        {'mode': 'chrome', 'name': 'Google Chrome/Chromium'}, {'mode': 'edge', 'name': 'Microsoft Edge'},
        {'mode': 'brave', 'name': 'Brave Browser'},
    ]
    for browser_pref in browser_preferences:
        try:
            logger.info(f"🌐 Attempting to launch with {browser_pref['name']}...")
            eel.start('index.html', mode=browser_pref['mode'], **eel_start_options)
            logger.info(f"✅ Successfully launched with {browser_pref['name']}. Application is running.")
            return True
        except Exception as e:
            logger.debug(f"❌ Failed to launch with {browser_pref['name']} (mode: {browser_pref['mode']}): {e}. Trying next...")
            time.sleep(0.2)
    logger.info("🌐 Attempting to launch with system default HTML handler (Eel 'default' mode)...")
    try:
        eel_default_options = eel_start_options.copy(); eel_default_options.pop('position', None)
        eel.start('index.html', mode='default', **eel_default_options)
        server_port = eel.SERVER_PORT if hasattr(eel, 'SERVER_PORT') else eel_start_options['port']
        logger.info(f"✅ Launched with system default handler. Access at http://{eel_start_options['host']}:{server_port}")
        input("🚀 Enhanced Toolkit is running! Press Enter to stop server and exit...")
        return True
    except Exception as e: logger.error(f"❌ Failed to launch with system default HTML handler: {e}")
    logger.warning("⚠️ Could not automatically open the application in a browser window.")
    server_port_fallback = eel_start_options['port'] if eel_start_options['port'] != 0 else 8000
    logger.info(f"💡 Starting in server-only mode. Please manually open your browser to: http://{eel_start_options['host']}:{server_port_fallback}")
    try:
        eel_server_only_options = {k: v for k, v in eel_start_options.items() if k not in ['size', 'position']}
        eel_server_only_options['port'] = server_port_fallback
        eel.start('index.html', mode=None, block=True, **eel_server_only_options)
        logger.info("✅ Server has been stopped.")
        return True
    except Exception as e:
        logger.critical(f"❌ Failed to start in server-only mode: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    if '--debug-early' in sys.argv:
        logger.setLevel("DEBUG")
        logger.info("🐛 Early debug mode enabled via command line.")
    start_app()