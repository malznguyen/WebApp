import eel
import sys
import os
import json
import base64
import tempfile
import threading
import time
import traceback 
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# --- Backend Path Setup ---
# Ensures the 'BE' directory is in the Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'BE')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# --- Initialize Logger First ---
# It's crucial to have the logger ready before any other modules use it.
from BE.utils.logger import setup_logger
logger = setup_logger()

# --- Module Availability Flags ---
# These flags help the app gracefully handle missing optional features.
VISION_MODULE_AVAILABLE = False

# --- Core Backend Imports ---
# This structured import helps diagnose issues more easily.
try:
    from BE.utils.helpers import ensure_dir_exists
    from BE.config.settings import (
        SERP_API_KEY, IMGUR_CLIENT_ID,
        CHATGPT_API_KEY, WINDOW_WIDTH, WINDOW_HEIGHT
    )
    from BE.config import constants
    from BE.core.api_client import validate_api_keys
    from BE.core.search_thread import search_image_sync
    from BE.core.document_api import (
        process_document,
        extract_text_preview,
        extract_text,
        process_batch_synthesis,
        process_documents_batch_web,
        process_url_document
    )
    from BE.core.image_processing import validate_image_upload
    from BE.core.image_metadata import ImageMetadataExtractor
    logger.info("✅ Core modules imported successfully.")
except ImportError as e:
    logger.critical(f"❌ Failed to import a core backend module: {e}", exc_info=True)
    # This is a fatal error, so we should exit.
    sys.exit(f"Core module import failed: {e}")

# --- Vision Module Import (Optional) ---
# This is kept separate because it's a new, non-essential feature.
try:
    from BE.core.vision_api import (
        OpenAIVisionClient,
        describe_image_with_openai_vision,
        detect_ai_image_with_openai_vision,
    )
    VISION_MODULE_AVAILABLE = True
    logger.info("✅ Vision module imported successfully.")
except ImportError as e:
    logger.warning(f"⚠️ Vision module not available. Image description will be disabled. Error: {e}")
    # Define placeholder functions if the module is missing
    def describe_image_with_openai_vision(*args, **kwargs):
        return {'success': False, 'error': 'Vision module not installed on the server.'}

    def detect_ai_image_with_openai_vision(*args, **kwargs):
        return {'success': False, 'error': 'Vision module not installed on the server.'}
    VISION_MODULE_AVAILABLE = False


# --- Globals & Initialization ---
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
        logger.debug(f"safe_int conversion failed for value: {value}, using default: {default}")
        return default

def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert a value to a boolean."""
    if isinstance(value, bool): return value
    if isinstance(value, str): return value.lower() in ('true', '1', 'yes', 'on')
    try: return bool(value)
    except: 
        logger.debug(f"safe_bool conversion failed for value: {value}, using default: {default}")
        return default

def validate_base64_data(data: str, expected_prefix: Optional[str] = None) -> bytes:
    """Validates and decodes base64 encoded data with size checks."""
    if not data or not isinstance(data, str):
        raise ValueError("Invalid or empty base64 data provided.")
    try:
        actual_data = data
        if data.startswith('data:'):
            if ',' not in data:
                raise ValueError("Invalid base64 data URL format: missing comma separator.")
            header, actual_data = data.split(',', 1)
            if expected_prefix and not header.startswith(expected_prefix):
                raise ValueError(
                    f"Invalid data URL prefix. Expected '{expected_prefix}', got '{header}'.")
        decoded_bytes = base64.b64decode(actual_data)
        if not decoded_bytes:
            raise ValueError("Decoded base64 data is empty.")
        from BE.config.constants import MAX_BASE64_DECODE_SIZE
        if len(decoded_bytes) > MAX_BASE64_DECODE_SIZE:
            size_mb = MAX_BASE64_DECODE_SIZE / (1024 * 1024)
            raise ValueError(f"Decoded data exceeds limit of {size_mb:.0f}MB.")
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
        logger.debug(f"Created temp file: {temp_file_path}")
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

# ===== EEL EXPOSED FUNCTIONS =====

@eel.expose
def get_app_config():
    """Enhanced version with vision capabilities check."""
    global app_config
    if app_config is None:
        try:
            api_keys_validation = validate_api_keys()
            
            app_config = {
                'status': 'ready',
                'has_serp_api': api_keys_validation.get('has_serp_api', False),
                'has_imgur': api_keys_validation.get('has_imgur', False),
                'has_chatgpt': bool(CHATGPT_API_KEY and CHATGPT_API_KEY.strip()),
                'has_vision': VISION_MODULE_AVAILABLE and bool(CHATGPT_API_KEY and CHATGPT_API_KEY.strip()),
                'vision_available': VISION_MODULE_AVAILABLE,
                'supported_formats': ['.pdf', '.docx', '.txt', '.md'],
                'version': '2.3.0-url',
                'ready': api_keys_validation.get('ready', False),
                'missing_keys': api_keys_validation.get('missing_keys', [])
            }
            
            if not app_config['has_vision']:
                if not VISION_MODULE_AVAILABLE:
                    app_config['missing_keys'].append('Vision dependencies (e.g., openai>=1.10.0)')
                elif not (CHATGPT_API_KEY and CHATGPT_API_KEY.strip()):
                    if 'CHATGPT_API_KEY' not in app_config['missing_keys']:
                        app_config['missing_keys'].append('CHATGPT_API_KEY (for Vision)')
            
            logger.info(f"Application configuration loaded. Vision available: {app_config['has_vision']}")
            
        except Exception as e:
            logger.error(f"Critical error loading application config: {e}", exc_info=True)
            app_config = {
                'status': 'error', 'error': str(e), 'ready': False,
                'has_serp_api': False, 'has_imgur': False, 'has_chatgpt': False, 'has_vision': False,
                'vision_available': False, 'supported_formats': [],
                'version': 'N/A', 'missing_keys': ['Configuration load failed']
            }
    
    return app_config

@eel.expose
def get_vision_capabilities() -> Dict[str, Any]:
    """Get current vision API capabilities and configuration status."""
    try:
        capabilities = {
            'available': VISION_MODULE_AVAILABLE,
            'api_configured': bool(CHATGPT_API_KEY and CHATGPT_API_KEY.strip()),
            'supported_languages': ['vietnamese', 'english'],
            'detail_levels': ['brief', 'detailed', 'extensive'],
        }
        return capabilities
    except Exception as e:
        logger.error(f"Error checking vision capabilities: {e}", exc_info=True)
        return {
            'available': False,
            'api_configured': False,
            'error': f"Capability check failed: {str(e)}"
        }

@eel.expose
def process_image_from_url(url: str) -> Dict[str, Any]:
    """Downloads an image from a URL, validates it, and prepares it for the app."""
    logger.info(f"Processing image from URL: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'image' not in content_type:
            return {'success': False, 'error': f"URL did not point to a direct image. Content type was '{content_type}'."}

        max_bytes = constants.MAX_IMAGE_SIZE_MB * 2 * 1024 * 1024
        content_length = int(response.headers.get('content-length', 0))
        if content_length and content_length > max_bytes:
            return {'success': False, 'error': 'Image exceeds allowed size limit.'}

        image_bytes = response.content
        if len(image_bytes) > max_bytes:
            return {'success': False, 'error': 'Image exceeds allowed size limit after download.'}
        validation = validate_image_upload(image_bytes)
        if not validation['valid']:
             return {'success': False, 'error': f"Invalid image from URL: {validation['error']}"}

        filename = os.path.basename(url.split('?')[0]) or "downloaded_image.jpg"
        
        # Prepare data for the frontend, similar to a file upload
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        return {
            'success': True,
            'name': filename,
            'size': len(image_bytes),
            'data': f"data:{content_type};base64,{base64_data}"
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching image from URL {url}: {e}", exc_info=True)
        return {'success': False, 'error': f"Network error: Could not fetch image from URL."}
    except Exception as e:
        logger.error(f"Unexpected error processing image URL {url}: {e}", exc_info=True)
        return {'success': False, 'error': f"An unexpected error occurred: {e}"}

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

def _run_async_search_task(task_id: str, *args, **kwargs):
    """Wrapper specifically for search tasks."""
    try:
        eel.searchProgress(task_id, 20, "Uploading and preparing image...")()
        result_dict = search_image_web(*args, **kwargs)
        eel.searchProgress(task_id, 100, "Search complete.")()
        if result_dict.get('success'):
            eel.searchComplete(task_id, result_dict['results'])()
        else:
            eel.searchError(task_id, "Search Operation Failed", result_dict.get('error', 'Unknown Error'))()
    except Exception as e:
        logger.error(f"Async search task thread for ID {task_id} failed: {e}", exc_info=True)
        eel.searchError(task_id, "System Error in Search Thread", str(e))()
    finally:
        active_searches.pop(task_id, None)

def _run_async_vision_task(task_id: str, *args, **kwargs):
    """Wrapper specifically for vision tasks."""
    try:
        eel.visionProgress(task_id, 15, "Validating and preparing image...")()
        eel.visionProgress(task_id, 40, "Sending image to AI for analysis...")()
        result_dict = describe_image_web(*args, **kwargs)
        eel.visionProgress(task_id, 90, "Processing AI response...")()
        if result_dict.get('success'):
            eel.visionComplete(task_id, result_dict)()
        else:
            eel.visionError(task_id, "Vision Analysis Failed", result_dict.get('error', 'Unknown Error'))()
    except Exception as e:
        logger.error(f"Async vision task thread for ID {task_id} failed: {e}", exc_info=True)
        eel.visionError(task_id, "System Error in Vision Thread", str(e))()
    finally:
        active_processing_tasks.pop(task_id, None)

def _run_async_detection_task(task_id: str, *args, **kwargs):
    """Wrapper for AI detection tasks."""
    try:
        eel.detectionProgress(task_id, 15, "Validating and preparing image...")()
        eel.detectionProgress(task_id, 40, "Sending image to AI for analysis...")()
        result_dict = detect_ai_image_web(*args, **kwargs)
        eel.detectionProgress(task_id, 90, "Processing AI response...")()
        if result_dict.get('success'):
            eel.detectionComplete(task_id, result_dict)()
        else:
            eel.detectionError(task_id, "AI Detection Failed", result_dict.get('error', 'Unknown Error'))()
    except Exception as e:
        logger.error(f"Async detection task thread for ID {task_id} failed: {e}", exc_info=True)
        eel.detectionError(task_id, "System Error in Detection Thread", str(e))()
    finally:
        active_processing_tasks.pop(task_id, None)

@eel.expose
def search_image_async_web(image_data_base64: str, filename: str, social_media_only: bool = False) -> str:
    search_id = f"search_{int(time.time())}"
    thread = threading.Thread(target=_run_async_search_task, args=(search_id, image_data_base64, filename, social_media_only))
    active_searches[search_id] = thread
    thread.start()
    logger.debug(f"Started async search thread with ID: {search_id}")
    return search_id

@eel.expose
def describe_image_web(image_data_base64: str, filename: str, language: str = "vietnamese", detail_level: str = "detailed", custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """Web API endpoint for image description using OpenAI Vision."""
    if not VISION_MODULE_AVAILABLE:
        return {'success': False, 'error': 'Vision module not available on the server.', 'description': None}
    
    try:
        image_bytes = validate_base64_data(image_data_base64, "data:image")
        result = describe_image_with_openai_vision(image_bytes, filename, language, detail_level, custom_prompt)
        return result
    except Exception as e:
        logger.error(f"Error in describe_image_web for '{filename}': {e}", exc_info=True)
        return {'success': False, 'error': f"System error: {e}", 'description': None}

@eel.expose  
def describe_image_async_web(image_data_base64: str, filename: str, language: str = "vietnamese", detail_level: str = "detailed", custom_prompt: Optional[str] = None) -> str:
    """Asynchronous version for vision tasks."""
    task_id = f"vision_{int(time.time())}"
    thread = threading.Thread(target=_run_async_vision_task, args=(task_id, image_data_base64, filename, language, detail_level, custom_prompt))
    active_processing_tasks[task_id] = thread
    thread.start()
    logger.debug(f"Started async vision thread with ID: {task_id}")
    return task_id

@eel.expose
def detect_ai_image_web(image_data_base64: str, filename: str) -> Dict[str, Any]:
    """Web API endpoint for AI-generated image detection."""
    if not VISION_MODULE_AVAILABLE:
        return {'success': False, 'error': 'Vision module not available on the server.', 'detection': None}

    try:
        image_bytes = validate_base64_data(image_data_base64, "data:image")
        result = detect_ai_image_with_openai_vision(image_bytes, filename)
        return result
    except Exception as e:
        logger.error(f"Error in detect_ai_image_web for '{filename}': {e}", exc_info=True)
        return {'success': False, 'error': f"System error: {e}", 'detection': None}

@eel.expose
def detect_ai_image_async_web(image_data_base64: str, filename: str) -> str:
    """Asynchronous version of AI detection."""
    task_id = f"aidetect_{int(time.time())}"
    thread = threading.Thread(target=_run_async_detection_task, args=(task_id, image_data_base64, filename))
    active_processing_tasks[task_id] = thread
    thread.start()
    logger.debug(f"Started async detection thread with ID: {task_id}")
    return task_id

@eel.expose
def analyze_image_metadata(image_data_base64: str, filename: str, include_sensitive: bool = False) -> Dict[str, Any]:
    """Extract image metadata and fingerprints from base64 image data."""
    try:
        image_bytes = validate_base64_data(image_data_base64, "data:image")
        extractor = ImageMetadataExtractor(privacy_safe=not include_sensitive)
        return extractor.extract_metadata(image_bytes, filename)
    except Exception as e:
        logger.error(f"Metadata analysis failed for '{filename}': {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def validate_processing_settings(settings_from_frontend: Dict[str, Any]) -> Dict[str, Any]:
    """Validates and normalizes document processing settings received from the frontend."""
    normalized_settings: Dict[str, Any] = {}
    try:
        use_ai = safe_bool(settings_from_frontend.get('use_ai', False))
        normalized_settings['chatgpt_key'] = CHATGPT_API_KEY if use_ai else None
        
        normalized_settings['summary_mode'] = settings_from_frontend.get('summary_mode', 'full')
        normalized_settings['summary_level'] = safe_int(settings_from_frontend.get('detail_level', 50), default=50, min_val=10, max_val=90)
        normalized_settings['word_count_limit'] = safe_int(settings_from_frontend.get('word_count_limit', 500), default=500, min_val=50, max_val=5000)
        lang_code = settings_from_frontend.get('language')
        normalized_settings['target_language_code'] = lang_code if lang_code and isinstance(lang_code, str) and lang_code.strip() else None
        
        processing_mode = settings_from_frontend.get('processing_mode', 'individual')
        normalized_settings['is_synthesis_task'] = (processing_mode == 'batch')

        logger.debug(f"Normalized processing settings: {normalized_settings}")
        return normalized_settings
    except Exception as e:
        logger.error(f"Error validating processing settings: {e}", exc_info=True)
        return {
            'chatgpt_key': None,
            'summary_mode': 'full',
            'summary_level': 50,
            'word_count_limit': 500,
            'target_language_code': None,
            'is_synthesis_task': False,
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

        processing_result = process_document(**normalized_processing_settings, file_path=doc_file_path)
        
        # This formatting logic should be consistent everywhere
        ai_result = None
        content = processing_result.get('chatgpt')
        if content is not None and not (isinstance(content, str) and content.startswith("<Not executed")):
            is_error_msg = isinstance(content, str) and ("Error:" in content or "failed" in content.lower())
            ai_result = {'model': 'ChatGPT', 'content': content, 'is_error': is_error_msg}
        
        overall_error_message = processing_result.get('error')
        has_any_errors = bool(overall_error_message) or (ai_result is not None and ai_result.get('is_error'))
        
        return {
            'success': not bool(overall_error_message),
            'original_text': processing_result.get('original_text', ''),
            'ai_result': ai_result,
            'analysis': processing_result.get('analysis', {}),
            'error': overall_error_message,
            'has_errors': has_any_errors
        }
    except Exception as e:
        logger.error(f"Unexpected error in process_document_web for '{file_basename}': {e}", exc_info=True)
        return {'success': False, 'error': f"An unexpected error occurred: {e}", 'original_text': '', 'ai_result': None, 'analysis': {}, 'has_errors': True}

@eel.expose
def process_document_async_web(file_input: Union[str, Dict[str, Any], List[Dict[str, Any]]], settings: Dict[str, Any]) -> str:
    """Correctly starts and handles asynchronous document processing."""
    process_id = f"doc_process_{int(time.time())}_{threading.get_ident()}"

    def run_doc_processing_in_thread():
        temp_file_for_this_task: Optional[str] = None
        temp_files_for_batch: List[str] = []
        log_filename = "document_processing_task"
        final_result_for_frontend: Dict[str, Any] = {}

        try:
            eel.processingProgress(process_id, 5, "Initializing processing...")()
            normalized_settings = validate_processing_settings(settings)
            
            raw_processing_result: Dict[str, Any] = {}

            if isinstance(file_input, list): # BATCH PROCESSING
                log_filename = f"{len(file_input)} documents in batch"
                eel.processingProgress(process_id, 10, f"Uploading {log_filename}...")()
                
                # Upload all files and collect temp paths
                for i, item_dict in enumerate(file_input):
                    eel.processingProgress(process_id, 10 + int(i / len(file_input) * 15), f"Uploading {item_dict.get('filename', 'file ' + str(i+1))}...")()
                    upload_resp = upload_file_web(item_dict['file_data'], item_dict['filename'])
                    if not upload_resp['success']: 
                        raise IOError(f"Upload failed for {item_dict['filename']}: {upload_resp.get('error', 'Unknown error')}")
                    temp_files_for_batch.append(upload_resp['temp_path'])
                
                eel.processingProgress(process_id, 30, "Starting batch synthesis...")()
                
                # 🚨 NOW THIS WILL WORK!
                raw_processing_result = process_documents_batch_web(temp_files_for_batch, normalized_settings)
                logger.info(f"Batch processing result structure: {list(raw_processing_result.keys())}")

                # Format batch result for frontend
                ai_result = None
                content = raw_processing_result.get('chatgpt_synthesis')
                if content and content != "<Not executed: No API key>":
                    is_error_msg = isinstance(content, str) and ("Error:" in content)
                    ai_result = {'model': 'ChatGPT', 'content': content, 'is_error': is_error_msg}
                
                final_result_for_frontend = {
                    'success': raw_processing_result.get('success', False),
                    'ai_result': ai_result,
                    'processed_files': raw_processing_result.get('processed_files', []),
                    'failed_files': raw_processing_result.get('failed_files', []),
                    'concatenated_text_char_count': raw_processing_result.get('concatenated_text_char_count', 0),
                    'overall_error': raw_processing_result.get('overall_error'),
                    'has_errors': bool(raw_processing_result.get('overall_error')) or (ai_result is not None and ai_result.get('is_error'))
                }

            else: # SINGLE ITEM PROCESSING (File or Text or URL)
                if isinstance(file_input, dict) and 'direct_text_content' in file_input:
                    log_filename = file_input.get('text_input_name', 'Direct Text Input')
                    eel.processingProgress(process_id, 20, f"Processing direct text: {log_filename}...")()
                    raw_processing_result = process_document(input_text_to_process=file_input['direct_text_content'], **normalized_settings)
                elif isinstance(file_input, dict) and 'url' in file_input:
                    log_filename = file_input['url']
                    eel.processingProgress(process_id, 20, f"Processing URL: {log_filename}...")()
                    raw_processing_result = process_url_document(
                        file_input['url'],
                        chatgpt_key=normalized_settings.get('chatgpt_key'),
                        summary_mode=normalized_settings.get('summary_mode', 'full'),
                        summary_level=normalized_settings.get('summary_level', 50),
                        word_count_limit=normalized_settings.get('word_count_limit', 500),
                        target_language_code=normalized_settings.get('target_language_code')
                    )
                else:
                    file_data = file_input['file_data']
                    log_filename = file_input['filename']
                    eel.processingProgress(process_id, 15, f"Uploading {log_filename}...")()
                    upload_resp = upload_file_web(file_data, log_filename)
                    if not upload_resp['success']: 
                        raise IOError(f"Upload failed for {log_filename}: {upload_resp.get('error', 'Unknown error')}")
                    temp_file_for_this_task = upload_resp['temp_path']
                    eel.processingProgress(process_id, 30, f"Processing file: {log_filename}...")()
                    raw_processing_result = process_document(file_path=temp_file_for_this_task, **normalized_settings)
                
                # Format single item result for frontend
                ai_result = None
                content = raw_processing_result.get('chatgpt')
                if content is not None and not (isinstance(content, str) and content.startswith("<Not executed")):
                    is_error_msg = isinstance(content, str) and ("Error:" in content or "failed" in content.lower())
                    ai_result = {'model': 'ChatGPT', 'content': content, 'is_error': is_error_msg}
                
                overall_error_message = raw_processing_result.get('error')
                has_any_errors = bool(overall_error_message) or (ai_result is not None and ai_result.get('is_error'))
                
                final_result_for_frontend = {
                    'success': not bool(overall_error_message),
                    'original_text': raw_processing_result.get('original_text', ''),
                    'ai_result': ai_result,
                    'analysis': raw_processing_result.get('analysis', {}),
                    'error': overall_error_message,
                    'has_errors': has_any_errors
                }

            eel.processingProgress(process_id, 100, f"Finalizing results for {log_filename}...")()
            eel.processingComplete(process_id, final_result_for_frontend)()
            logger.info(f"Processing complete for '{log_filename}' (ID: {process_id})")
            
        except Exception as e:
            logger.error(f"Async document processing for '{log_filename}' (ID: {process_id}) failed in thread: {e}", exc_info=True)
            eel.processingError(process_id, f"Error processing '{log_filename}': {str(e)}")()
        finally:
            # Cleanup temp files
            if temp_file_for_this_task: cleanup_temp_file(temp_file_for_this_task)
            for path_to_clean in temp_files_for_batch: cleanup_temp_file(path_to_clean)
            active_processing_tasks.pop(process_id, None)
            logger.debug(f"Cleaned up resources for process ID: {process_id}")

    doc_process_thread = threading.Thread(target=run_doc_processing_in_thread, daemon=True)
    active_processing_tasks[process_id] = doc_process_thread
    doc_process_thread.start()
    logger.info(f"Started document processing thread with ID: {process_id}")
    return process_id

@eel.expose
def upload_file_web(file_data_base64: str, filename: str) -> Dict[str, Any]:
    """Handles file uploads, saves to temp, returns path for further backend use."""
    temp_upload_path: Optional[str] = None
    try:
        if not file_data_base64 or not filename:
            raise ValueError("File data or filename is missing for upload.")
        file_bytes = validate_base64_data(file_data_base64)
        temp_upload_path = create_temp_file(file_bytes, filename, suffix=os.path.splitext(filename)[1] or '.bin')
        logger.info(f"File uploaded successfully: {filename} -> {temp_upload_path}")
        return {
            'success': True, 'temp_path': temp_upload_path,
            'size': len(file_bytes), 'filename': os.path.basename(temp_upload_path)
        }
    except Exception as e:
        if temp_upload_path: cleanup_temp_file(temp_upload_path)
        logger.error(f"Error during upload of '{filename}': {e}", exc_info=True)
        return {'success': False, 'error': f"Upload error: {e}", 'temp_path': None}

@eel.expose
def perform_cleanup_temp_files() -> Dict[str, Any]:
    """Cleans up all files within the 'temp' directory."""
    temp_dir_path = 'temp'
    cleaned_files_count = 0
    errors_list = []
    
    try:
        logger.info(f"Initiating cleanup of temporary files in '{temp_dir_path}'.")
        if os.path.isdir(temp_dir_path):
            for item_name in os.listdir(temp_dir_path):
                item_path = os.path.join(temp_dir_path, item_name)
                if os.path.isfile(item_path):
                    try:
                        os.unlink(item_path)
                        cleaned_files_count += 1
                    except Exception as e_file:
                        err_detail = f"Could not delete temp file '{item_path}': {e_file}"
                        logger.warning(err_detail)
                        errors_list.append(err_detail)
            message = f"Cleanup complete. {cleaned_files_count} files deleted."
            if errors_list: message += f" Encountered {len(errors_list)} errors."
            logger.info(message)
            return {'success': True, 'cleaned_count': cleaned_files_count, 'errors': errors_list}
        else:
            logger.warning(f"Temp directory '{temp_dir_path}' not found.")
            return {'success': True, 'cleaned_count': 0, 'errors': [], 'message': "Temp directory not found."}
    except Exception as e_main:
        logger.error(f"Major error during temporary files cleanup: {e_main}", exc_info=True)
        return {'success': False, 'error': str(e_main), 'cleaned_count': 0, 'errors': [str(e_main)]}

# ===== APPLICATION STARTUP LOGIC =====
def start_app():
    """Initializes and starts the Eel-based web application."""
    try:
        ensure_dir_exists('temp')
        ensure_dir_exists('logs')
        
        initial_config = get_app_config()
        logger.info(f"🚀 Enhanced Toolkit v{initial_config.get('version', 'N/A')} starting...")
        logger.info(f"Configuration status: {initial_config}")
        
        if not initial_config.get('ready', False) or not initial_config.get('has_serp_api'):
            logger.warning(f"⚠️ App may operate in limited mode. Missing keys: {initial_config.get('missing_keys', [])}")

        eel_start_options = {
            'size': (WINDOW_WIDTH, WINDOW_HEIGHT), 
            'position': (50, 50),
            'disable_cache': True, 
            'port': 0,  # Auto-select port
            'host': 'localhost'
        }
        
        logger.info("Starting Eel web server...")
        # This starts the web server and opens the browser
        eel.start('index.html', mode='chrome', **eel_start_options)

    except (IOError, OSError) as e:
        logger.critical(f"❌ Failed to start. A browser (like Chrome) is required. Error: {e}", exc_info=True)
        print("\n--- BROWSER NOT FOUND ---\nCould not find Google Chrome. Please install it or try starting the app with a different browser mode.\n")
    except Exception as e:
        logger.critical(f"❌ Fatal error during application startup: {e}", exc_info=True)
        print(f"\n--- FATAL ERROR ---\nAn unexpected error occurred: {e}\nCheck the latest log file in the 'logs' directory for details.")

# 🚨 PARANOID CHECK: Ensure all required functions exist
def _verify_imports():
    """Verify all critical imports are available (run this during dev/testing)"""
    required_functions = [
        ('process_document', process_document),
        ('process_batch_synthesis', process_batch_synthesis),
        ('process_documents_batch_web', process_documents_batch_web),
        ('process_url_document', process_url_document),
        ('extract_text', extract_text),
        ('validate_image_upload', validate_image_upload),
        ('search_image_sync', search_image_sync)
    ]
    
    for func_name, func_obj in required_functions:
        if not callable(func_obj):
            logger.critical(f"❌ CRITICAL: {func_name} is not callable!")
            raise ImportError(f"{func_name} is not properly imported or defined!")
    
    logger.info("✅ All critical imports verified successfully.")

# Run verification in debug mode
if __name__ == '__main__' and os.environ.get('DEBUG', '').lower() == 'true':
    _verify_imports()

if __name__ == '__main__':
    start_app()