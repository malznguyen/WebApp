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
from typing import Dict, Any, Optional, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'BE')
sys.path.insert(0, backend_path)

try:
    from core.search_thread import SearchThread, search_image_sync
    from core.document_api import process_document, process_document_async, extract_text_preview
    from core.image_processing import process_web_upload, validate_image_upload
    from core.api_client import validate_api_keys
    from config.settings import *
    from utils.logger import setup_logger
    from utils.helpers import ensure_dir_exists
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure backend modules are properly set up!")
    sys.exit(1)

# Initialize logger
logger = setup_logger()

# Initialize eel with correct path
eel.init('FE')

# Global state
active_searches = {}
processing_queue = {}
app_config = None

# ===== UTILITY FUNCTIONS =====

def safe_int(value: Any, default: int = 0, min_val: int = None, max_val: int = None) -> int:
    try:
        result = int(value)
        if min_val is not None:
            result = max(min_val, result)
        if max_val is not None:
            result = min(max_val, result)
        return result
    except (ValueError, TypeError):
        return default

def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert value to boolean"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    try:
        return bool(value)
    except:
        return default

def validate_base64_image(data: str) -> bytes:
    """Validate and decode base64 image data"""
    if not data:
        raise ValueError("Empty image data")
    
    try:
        if data.startswith('data:image'):
            if ',' not in data:
                raise ValueError("Invalid base64 format")
            header, data = data.split(',', 1)
        
        image_bytes = base64.b64decode(data)
        if len(image_bytes) == 0:
            raise ValueError("Decoded image is empty")
        
        return image_bytes
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")

def create_temp_file(data: bytes, filename: str, suffix: str = None) -> str:
    """Create temporary file with proper cleanup handling"""
    try:
        temp_dir = ensure_dir_exists('temp')
        
        if suffix is None:
            suffix = os.path.splitext(filename)[1] or '.tmp'
        
        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=temp_dir,
            prefix='upload_'
        )
        
        temp_file.write(data)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        raise IOError(f"Failed to create temporary file: {str(e)}")

def cleanup_temp_file(file_path: str) -> None:
    """Safely cleanup temporary file"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temp file: {file_path}")
    except Exception as e:
        logger.warning(f"Could not delete temp file {file_path}: {e}")

def validate_file_path(file_path: str) -> bool:
    """Validate file path exists and is readable"""
    try:
        return (file_path and 
                os.path.exists(file_path) and 
                os.path.isfile(file_path) and 
                os.access(file_path, os.R_OK))
    except:
        return False

# ===== INITIALIZATION =====

@eel.expose
def get_app_config():
    """Get application configuration for web UI"""
    global app_config
    
    if app_config is None:
        try:
            validation = validate_api_keys()
            
            app_config = {
                'status': 'ready',
                'has_serp_api': validation.get('has_serp_api', False),
                'has_imgur': validation.get('has_imgur', False),
                'has_deepseek': bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY.strip()),
                'has_grok': bool(GROK_API_KEY and GROK_API_KEY.strip()),
                'has_chatgpt': bool(CHATGPT_API_KEY and CHATGPT_API_KEY.strip()),
                'supported_formats': ['.pdf', '.docx', '.txt', '.md'],
                'version': '2.0',
                'ready': validation.get('ready', False),
                'missing_keys': validation.get('missing_keys', [])
            }
            
            logger.info(f"App config loaded - Ready: {app_config['ready']}")
            
        except Exception as e:
            logger.error(f"Error loading app config: {e}")
            app_config = {
                'status': 'error',
                'error': str(e),
                'ready': False
            }
    
    return app_config

# ===== IMAGE SEARCH FUNCTIONS =====

@eel.expose
def search_image_web(image_data: str, filename: str, social_only: bool = False) -> Dict[str, Any]:
    """Handle image search from web UI with comprehensive error handling"""
    temp_file_path = None
    
    try:
        # Validate inputs
        if not image_data or not filename:
            raise ValueError("Missing image data or filename")
        
        if not isinstance(social_only, bool):
            social_only = safe_bool(social_only)
        
        logger.info(f"Starting image search - File: {filename}, Social only: {social_only}")
        
        # Validate and decode image
        image_bytes = validate_base64_image(image_data)
        
        # Validate image format and size
        validation = validate_image_upload(image_bytes)
        if not validation.get('valid', False):
            raise ValueError(validation.get('error', 'Invalid image'))
        
        # Create temporary file
        temp_file_path = create_temp_file(image_bytes, filename)
        logger.info(f"Image saved to: {temp_file_path}")
        
        # Validate API keys
        if not SERP_API_KEY or not IMGUR_CLIENT_ID:
            raise ValueError("Missing required API keys (SERP_API_KEY or IMGUR_CLIENT_ID)")
        
        # Perform search
        results = search_image_sync(
            image_path=temp_file_path,
            serp_api_key=SERP_API_KEY,
            imgur_client_id=IMGUR_CLIENT_ID,
            social_media_only=social_only,
            timeout=120
        )
        
        logger.info(f"Search completed - Found {len(results)} results")
        
        return {
            'success': True,
            'results': results or [],
            'total': len(results) if results else 0,
            'social_only': social_only
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Image search failed: {error_msg}", exc_info=True)
        return {
            'success': False,
            'error': error_msg,
            'results': [],
            'total': 0
        }
    
    finally:
        if temp_file_path:
            cleanup_temp_file(temp_file_path)

@eel.expose
def search_image_async_web(image_data: str, filename: str, social_only: bool = False) -> str:
    """Start async image search with progress updates"""
    search_id = f"search_{int(time.time())}_{threading.get_ident()}"
    
    def run_search():
        try:
            result = search_image_web(image_data, filename, social_only)
            if result['success']:
                eel.searchComplete(search_id, result['results'])
            else:
                eel.searchError(search_id, "Search Failed", result['error'])
        except Exception as e:
            logger.error(f"Async search failed: {e}", exc_info=True)
            eel.searchError(search_id, "System Error", str(e))
        finally:
            active_searches.pop(search_id, None)
    
    thread = threading.Thread(target=run_search, daemon=True)
    thread.start()
    
    active_searches[search_id] = thread
    return search_id

# ===== DOCUMENT PROCESSING FUNCTIONS =====

def validate_processing_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize processing settings"""
    try:
        normalized = {}
        
        # Handle AI models
        ai_models = settings.get('ai_models', {})
        if not isinstance(ai_models, dict):
            ai_models = {}
        
        # Validate API keys and models
        normalized['deepseek_key'] = DEEPSEEK_API_KEY if safe_bool(ai_models.get('deepseek')) else None
        normalized['grok_key'] = GROK_API_KEY if safe_bool(ai_models.get('grok')) else None
        normalized['chatgpt_key'] = CHATGPT_API_KEY if safe_bool(ai_models.get('chatgpt')) else None
        
        # Validate summary level
        normalized['summary_level'] = safe_int(
            settings.get('detail_level', 50), 
            default=50, 
            min_val=10, 
            max_val=90
        )
        
        # Handle language
        language = settings.get('language')
        normalized['target_language_code'] = language if language and language.strip() else None
        
        # Handle processing mode
        processing_mode = settings.get('processing_mode', 'individual')
        normalized['is_synthesis_task'] = processing_mode == 'batch'
        
        return normalized
        
    except Exception as e:
        logger.error(f"Settings validation failed: {e}")
        return {
            'deepseek_key': None,
            'grok_key': None, 
            'chatgpt_key': None,
            'summary_level': 50,
            'target_language_code': None,
            'is_synthesis_task': False
        }

@eel.expose
def process_document_web(file_path: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Process document from web UI with comprehensive error handling"""
    try:
        # Validate inputs
        if not file_path:
            raise ValueError("Missing file path")
        
        if not validate_file_path(file_path):
            raise ValueError(f"File not found or not readable: {file_path}")
        
        if not isinstance(settings, dict):
            settings = {}
        
        logger.info(f"Starting document processing - File: {file_path}")
        
        # Validate and normalize settings
        processing_settings = validate_processing_settings(settings)
        
        # Check if at least one AI model is enabled
        has_ai_model = any([
            processing_settings['deepseek_key'],
            processing_settings['grok_key'],
            processing_settings['chatgpt_key']
        ])
        
        if not has_ai_model:
            logger.warning("No AI models enabled, processing will only include text analysis")
        
        # Process document
        result = process_document(
            file_path=file_path,
            **processing_settings
        )
        
        if not result:
            raise ValueError("Document processing returned empty result")
        
        logger.info(f"Document processing completed for: {file_path}")
        
        # Format AI results
        ai_results = []
        for model_name, model_key in [('DeepSeek', 'deepseek'), ('Grok', 'grok'), ('ChatGPT', 'chatgpt')]:
            content = result.get(model_key)
            if content and content != "<Not executed>":
                ai_results.append({
                    'model': model_name,
                    'content': content
                })
        
        return {
            'success': True,
            'original_text': result.get('original_text', ''),
            'ai_results': ai_results,
            'analysis': result.get('analysis', {}),
            'error': result.get('error'),
            'has_errors': bool(result.get('error'))
        }
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Document processing failed: {error_msg}", exc_info=True)
        return {
            'success': False,
            'error': error_msg,
            'original_text': '',
            'ai_results': [],
            'analysis': {},
            'has_errors': True
        }

@eel.expose
def process_document_async_web(file_path: str, settings: Dict[str, Any]) -> str:
    """Start async document processing with progress updates"""
    process_id = f"process_{int(time.time())}_{threading.get_ident()}"
    
    def run_processing():
        try:
            result = process_document_web(file_path, settings)
            eel.processingComplete(process_id, result)
        except Exception as e:
            logger.error(f"Async processing failed: {e}", exc_info=True)
            eel.processingError(process_id, str(e))
        finally:
            processing_queue.pop(process_id, None)
    
    thread = threading.Thread(target=run_processing, daemon=True)
    thread.start()
    
    processing_queue[process_id] = thread
    return process_id

@eel.expose
def get_document_preview(file_path: str, max_chars: int = 1000) -> Dict[str, Any]:
    """Get document text preview for web UI"""
    try:
        if not validate_file_path(file_path):
            raise ValueError("File not found or not readable")
        
        max_chars = safe_int(max_chars, default=1000, min_val=100, max_val=5000)
        
        preview = extract_text_preview(file_path, max_chars)
        return {
            'success': True,
            'preview': preview or "Could not extract preview"
        }
    except Exception as e:
        logger.error(f"Preview extraction failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'preview': ''
        }

# ===== FILE HANDLING =====

@eel.expose
def upload_file_web(file_data: str, filename: str) -> Dict[str, Any]:
    """Handle file upload from web UI with comprehensive validation"""
    try:
        if not file_data or not filename:
            raise ValueError("Missing file data or filename")
        
        # Validate filename
        if len(filename) > 255:
            raise ValueError("Filename too long")
        
        # Decode file data
        try:
            if file_data.startswith('data:'):
                if ',' not in file_data:
                    raise ValueError("Invalid file data format")
                header, data = file_data.split(',', 1)
                file_bytes = base64.b64decode(data)
            else:
                file_bytes = base64.b64decode(file_data)
        except Exception as e:
            raise ValueError(f"Invalid file data: {str(e)}")
        
        if len(file_bytes) == 0:
            raise ValueError("File is empty")
        
        # Check file size (50MB limit)
        if len(file_bytes) > 50 * 1024 * 1024:
            raise ValueError("File too large (max 50MB)")
        
        # Create temporary file
        temp_path = create_temp_file(file_bytes, filename)
        
        logger.info(f"File uploaded: {temp_path} ({len(file_bytes)} bytes)")
        
        return {
            'success': True,
            'temp_path': temp_path,
            'size': len(file_bytes),
            'filename': os.path.basename(temp_path)
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return {
            'success': False,
            'error': str(e),
            'temp_path': None
        }

# ===== UTILITY FUNCTIONS =====

@eel.expose
def get_system_status() -> Dict[str, Any]:
    """Get system status for web UI"""
    try:
        temp_count = 0
        if os.path.exists('temp'):
            temp_count = len([f for f in os.listdir('temp') if os.path.isfile(os.path.join('temp', f))])
        
        return {
            'active_searches': len(active_searches),
            'processing_queue': len(processing_queue),
            'temp_files': temp_count,
            'python_version': sys.version.split()[0],
            'working_directory': os.getcwd(),
            'backend_loaded': True
        }
    except Exception as e:
        return {
            'error': str(e),
            'active_searches': 0,
            'processing_queue': 0,
            'temp_files': 0,
            'backend_loaded': False
        }

@eel.expose
def cleanup_temp_files() -> Dict[str, Any]:
    """Clean up temporary files"""
    try:
        temp_dir = 'temp'
        cleaned = 0
        
        if os.path.exists(temp_dir):
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        os.unlink(file_path)
                        cleaned += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleaned} temporary files")
        return {'success': True, 'cleaned': cleaned}
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {'success': False, 'error': str(e), 'cleaned': 0}

@eel.expose
def save_user_settings(settings: Dict[str, Any]) -> Dict[str, bool]:
    """Save user preferences"""
    try:
        if not isinstance(settings, dict):
            raise ValueError("Settings must be a dictionary")
        
        settings_file = 'user_settings.json'
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        
        return {'success': True}
    except Exception as e:
        logger.error(f"Settings save failed: {e}")
        return {'success': False}

@eel.expose
def load_user_settings() -> Dict[str, Any]:
    """Load user preferences"""
    try:
        settings_file = 'user_settings.json'
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            return {'success': True, 'settings': settings}
        return {'success': True, 'settings': {}}
    except Exception as e:
        logger.error(f"Settings load failed: {e}")
        return {'success': False, 'settings': {}}

# ===== APPLICATION STARTUP =====

def start_app():
    """Start the web application"""
    try:
        # Ensure required directories exist
        ensure_dir_exists('temp')
        ensure_dir_exists('logs')
        
        # Load initial config
        config = get_app_config()
        logger.info(f"🚀 Enhanced Toolkit v2.0 starting...")
        logger.info(f"Config status: {config.get('status')}")
        
        if not config.get('ready'):
            logger.warning(f"⚠️  Missing API keys: {config.get('missing_keys', [])}")
            logger.warning("App will run in demo mode")
        
        # Start with error handling
        try_start_browser()
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}", exc_info=True)
        input("Press Enter to exit...")
        sys.exit(1)

def try_start_browser():
    """Try multiple browsers in order of preference"""
    browsers_to_try = [
        ('chrome', 'Google Chrome'),
        ('edge', 'Microsoft Edge'), 
        ('brave', 'Brave Browser'),
        ('firefox', 'Mozilla Firefox'),
        ('safari', 'Safari'),
        ('default', 'Default Browser')
    ]
    
    for browser_mode, browser_name in browsers_to_try:
        try:
            logger.info(f"🌐 Trying {browser_name}...")
            
            if browser_mode == 'default':
                eel.start(
                    'index.html',
                    size=(1200, 800),
                    disable_cache=True,
                    port=8000,
                    mode=None,
                    host='localhost',
                    block=False
                )
                import webbrowser
                webbrowser.open('http://localhost:8000')
                logger.info(f"✅ Opened in {browser_name} at http://localhost:8000")
                input("🚀 Enhanced Toolkit is running! Press Enter to stop...")
                return True
            else:
                eel.start(
                    'index.html',
                    size=(1200, 800),
                    position=(100, 100),
                    disable_cache=True,
                    port=8000,
                    mode=browser_mode
                )
                logger.info(f"✅ Successfully started with {browser_name}")
                return True
                
        except Exception as e:
            logger.debug(f"❌ {browser_name} failed: {e}")
            continue
    
    # Fallback server mode
    logger.error("❌ Could not start any browser!")
    logger.info("💡 Manual option: Open http://localhost:8000 in your browser")
    
    try:
        def start_server():
            eel.start('index.html', port=8000, mode=None, block=True)
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        logger.info("🌐 Server started at http://localhost:8000")
        logger.info("📖 Please open this URL in your browser manually")
        input("Press Enter to stop server...")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--debug':
            eel.start('index.html', debug=True)
        elif sys.argv[1] == '--demo':
            logger.info("🎭 Starting in demo mode")
            start_app()
        else:
            print("Usage: python main.py [--debug|--demo]")
    else:
        start_app()