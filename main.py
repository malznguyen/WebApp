#!/usr/bin/env python3
"""
Enhanced Toolkit v2.0 - Web UI Bridge
Main entry point connecting Python backend to web frontend via Eel
"""

import eel
import sys
import os
import json
import base64
import tempfile
import threading
import time
from pathlib import Path

# Add backend to path - adjust based on your structure
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'BE')  # Since you have BE folder
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
    print("Make sure you've completed Phase 1 setup!")
    sys.exit(1)

# Initialize logger
logger = setup_logger()

# Initialize eel
eel.init('web-frontend')

# Global state
active_searches = {}
processing_queue = {}
app_config = None

# ===== INITIALIZATION =====

@eel.expose
def get_app_config():
    """Get application configuration for web UI"""
    global app_config
    
    if app_config is None:
        try:
            # Validate API keys
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
            
            logger.info(f"App config loaded - API Keys ready: {app_config['ready']}")
            
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
def search_image_web(image_data, filename, social_only=False):
    """Handle image search from web UI"""
    try:
        logger.info(f"Starting web image search - File: {filename}, Social only: {social_only}")
        
        # Validate image data
        if not image_data or not filename:
            raise ValueError("Missing image data or filename")
        
        # Process base64 image data
        if image_data.startswith('data:image'):
            header, data = image_data.split(',', 1)
            image_bytes = base64.b64decode(data)
        else:
            image_bytes = base64.b64decode(image_data)
        
        # Validate image
        validation = validate_image_upload(image_bytes)
        if not validation['valid']:
            raise ValueError(validation['error'])
        
        # Save to temporary file
        temp_dir = ensure_dir_exists('temp')
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=os.path.splitext(filename)[1],
            dir=temp_dir
        )
        temp_file.write(image_bytes)
        temp_file.close()
        
        logger.info(f"Image saved to: {temp_file.name}")
        
        # Perform search using existing core
        try:
            results = search_image_sync(
                image_path=temp_file.name,
                serp_api_key=SERP_API_KEY,
                imgur_client_id=IMGUR_CLIENT_ID,
                social_media_only=social_only,
                timeout=120
            )
            
            logger.info(f"Search completed - Found {len(results)} results")
            
            return {
                'success': True,
                'results': results,
                'total': len(results),
                'social_only': social_only
            }
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_file.name)
                logger.debug(f"Cleaned up temp file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Could not delete temp file: {e}")
        
    except Exception as e:
        logger.error(f"Image search failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

@eel.expose
def search_image_async_web(image_data, filename, social_only=False):
    """Start async image search with progress updates"""
    search_id = f"search_{int(time.time())}"
    
    def progress_callback(message):
        eel.updateSearchProgress(search_id, message)
    
    def complete_callback(results):
        eel.searchComplete(search_id, results)
    
    def error_callback(error_title, error_msg):
        eel.searchError(search_id, error_title, error_msg)
    
    # Start search in background thread
    def run_search():
        try:
            result = search_image_web(image_data, filename, social_only)
            if result['success']:
                complete_callback(result['results'])
            else:
                error_callback("Search Failed", result['error'])
        except Exception as e:
            error_callback("System Error", str(e))
    
    thread = threading.Thread(target=run_search, daemon=True)
    thread.start()
    
    active_searches[search_id] = thread
    return search_id

# ===== DOCUMENT PROCESSING FUNCTIONS =====

@eel.expose
def process_document_web(file_path, settings):
    """Process document from web UI"""
    try:
        logger.info(f"Starting document processing - File: {file_path}")
        
        # Validate settings
        if not isinstance(settings, dict):
            settings = {}
        
        # Extract settings
        processing_settings = {
            'deepseek_key': DEEPSEEK_API_KEY if settings.get('ai_models', {}).get('deepseek', False) else None,
            'grok_key': GROK_API_KEY if settings.get('ai_models', {}).get('grok', False) else None,
            'chatgpt_key': CHATGPT_API_KEY if settings.get('ai_models', {}).get('chatgpt', False) else None,
            'summary_level': settings.get('detail_level', 50),
            'target_language': settings.get('language', None),
            'is_synthesis': settings.get('processing_mode') == 'batch'
        }
        
        # Process document using existing core
        result = process_document(
            file_path=file_path,
            **processing_settings
        )
        
        logger.info(f"Document processing completed for: {file_path}")
        
        # Format result for web UI
        return {
            'success': True,
            'original_text': result.get('original_text', ''),
            'ai_results': [
                {'model': 'DeepSeek', 'content': result.get('deepseek', '')} if result.get('deepseek') else None,
                {'model': 'Grok', 'content': result.get('grok', '')} if result.get('grok') else None,
                {'model': 'ChatGPT', 'content': result.get('chatgpt', '')} if result.get('chatgpt') else None,
            ],
            'analysis': result.get('analysis', {}),
            'error': result.get('error')
        }
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

@eel.expose
def process_document_async_web(file_path, settings):
    """Start async document processing with progress updates"""
    process_id = f"process_{int(time.time())}"
    
    def progress_callback(message):
        eel.updateProcessingProgress(process_id, message)
    
    # Start processing in background
    def run_processing():
        try:
            result = process_document_async(
                file_path=file_path,
                settings=settings,
                progress_callback=progress_callback
            )
            eel.processingComplete(process_id, result)
        except Exception as e:
            eel.processingError(process_id, str(e))
    
    thread = threading.Thread(target=run_processing, daemon=True)
    thread.start()
    
    processing_queue[process_id] = thread
    return process_id

@eel.expose
def get_document_preview(file_path, max_chars=1000):
    """Get document text preview for web UI"""
    try:
        preview = extract_text_preview(file_path, max_chars)
        return {
            'success': True,
            'preview': preview
        }
    except Exception as e:
        logger.error(f"Preview extraction failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# ===== FILE HANDLING =====

@eel.expose
def upload_file_web(file_data, filename):
    """Handle file upload from web UI"""
    try:
        # Process base64 file data
        if file_data.startswith('data:'):
            header, data = file_data.split(',', 1)
            file_bytes = base64.b64decode(data)
        else:
            file_bytes = base64.b64decode(file_data)
        
        # Save to temp directory
        temp_dir = ensure_dir_exists('temp')
        temp_path = os.path.join(temp_dir, filename)
        
        # Add timestamp if file exists
        if os.path.exists(temp_path):
            name, ext = os.path.splitext(filename)
            timestamp = int(time.time())
            temp_path = os.path.join(temp_dir, f"{name}_{timestamp}{ext}")
        
        with open(temp_path, 'wb') as f:
            f.write(file_bytes)
        
        logger.info(f"File uploaded: {temp_path}")
        
        return {
            'success': True,
            'temp_path': temp_path,
            'size': len(file_bytes)
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# ===== UTILITY FUNCTIONS =====

@eel.expose
def get_system_status():
    """Get system status for web UI"""
    return {
        'active_searches': len(active_searches),
        'processing_queue': len(processing_queue),
        'temp_files': len(os.listdir('temp')) if os.path.exists('temp') else 0,
        'python_version': sys.version,
        'working_directory': os.getcwd()
    }

@eel.expose
def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        temp_dir = 'temp'
        if os.path.exists(temp_dir):
            count = 0
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    count += 1
            logger.info(f"Cleaned up {count} temporary files")
            return {'success': True, 'cleaned': count}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {'success': False, 'error': str(e)}

@eel.expose
def save_user_settings(settings):
    """Save user preferences"""
    try:
        settings_file = 'user_settings.json'
        with open(settings_file, 'w') as f:
            json.dump(settings, f, indent=2)
        return {'success': True}
    except Exception as e:
        logger.error(f"Settings save failed: {e}")
        return {'success': False, 'error': str(e)}

@eel.expose
def load_user_settings():
    """Load user preferences"""
    try:
        settings_file = 'user_settings.json'
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            return {'success': True, 'settings': settings}
        return {'success': True, 'settings': {}}
    except Exception as e:
        logger.error(f"Settings load failed: {e}")
        return {'success': False, 'error': str(e)}

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
                # Special handling for default browser
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
                # Try specific browser
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
    
    # If all browsers fail
    logger.error("❌ Could not start any browser!")
    logger.info("💡 Manual option: Open http://localhost:8000 in your browser")
    
    # Start server without browser
    try:
        import threading
        import time
        
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
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}", exc_info=True)
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == '__main__':
    # Check for command line arguments
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