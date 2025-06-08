import threading
import time
import logging
from typing import Callable, Optional, List, Dict, Any
from core.api_client import ImgurClient, SerpApiClient
from config.constants import SOCIAL_MEDIA_FILTER_DEFAULT

logger = logging.getLogger('ImageSearchApp')


class SearchThread:
    """Web-compatible search thread using standard threading instead of PyQt"""
    
    def __init__(self, image_path, serp_api_key, imgur_client_id, social_media_only=SOCIAL_MEDIA_FILTER_DEFAULT, parent=None):
        self.image_path = image_path
        self.imgur_client = ImgurClient(imgur_client_id)
        self.serp_api_client = SerpApiClient(serp_api_key)
        self.social_media_only = social_media_only
        self.search_active = True
        self.thread = None
        
        # Callback functions replace PyQt signals
        self.on_progress: Optional[Callable[[str], None]] = None
        self.on_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None
        self.on_finished: Optional[Callable[[], None]] = None

    def start(self):
        """Start search in background thread"""
        if self.thread and self.thread.is_alive():
            logger.warning("Search thread already running")
            return
            
        self.search_active = True
        self.thread = threading.Thread(target=self._run_search, name="SearchThread")
        self.thread.daemon = True
        self.thread.start()
        logger.debug("Search thread started")

    def stop(self):
        """Stop search thread"""
        logger.info("Requesting search thread stop...")
        self.search_active = False

    def is_running(self):
        """Check if search thread is running"""
        return self.thread and self.thread.is_alive()

    def wait(self, timeout=None):
        """Wait for thread to finish"""
        if self.thread:
            self.thread.join(timeout)
            return not self.thread.is_alive()
        return True

    def _run_search(self):
        """Main search execution - converted from PyQt run() method"""
        try:
            logger.info(f"Search thread started for image: {self.image_path}, social media filter: {self.social_media_only}")

            if not self.search_active:
                logger.info("Thread cancelled before processing")
                self._emit_finished()
                return

            self._emit_progress("Uploading image to Imgur...")
            image_url = self.imgur_client.upload_image(self.image_path)

            if not image_url:
                logger.error("Failed to upload image to Imgur, stopping search.")
                self._emit_error("Image Upload Error", "Failed to upload image to Imgur. Please check Client ID and network connection.")
                self._emit_finished()
                return

            logger.info(f"Image uploaded to Imgur: {image_url}")

            if not self.search_active:
                logger.info("Thread cancelled after Imgur upload")
                self._emit_finished()
                return

            self._emit_progress("Querying SerpAPI for image search...")
            logger.info("Starting SerpAPI search request")
            
            search_results, error = self.serp_api_client.search_by_image(
                image_url,
                social_media_only=self.social_media_only
            )

            if not self.search_active:
                logger.info("Thread cancelled during SerpAPI search")
                self._emit_finished()
                return

            if error:
                logger.error(f"SerpAPI error occurred: {error}")
                if not search_results:
                    self._emit_error("SerpAPI Error", f"Failed to get data: {error}")
                    self._emit_finished()
                    return
                else:
                    logger.warning(f"Error occurred ({error}), but showing {len(search_results)} results found.")

            self._emit_progress(f"Search completed. Found {len(search_results)} results.")
            self._emit_complete(search_results or [])

        except Exception as e:
            logger.error(f"Unexpected error in SearchThread: {str(e)}", exc_info=True)
            if self.search_active:
                self._emit_error("System Error", f"An unexpected error occurred during search: {e}")
        finally:
            self._emit_finished()

    def _emit_progress(self, message):
        """Emit progress update"""
        if self.on_progress and self.search_active:
            try:
                self.on_progress(message)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def _emit_complete(self, results):
        """Emit search completion with results"""
        if self.on_complete and self.search_active:
            try:
                self.on_complete(results)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")

    def _emit_error(self, error_title, error_message):
        """Emit error notification"""
        if self.on_error and self.search_active:
            try:
                self.on_error(error_title, error_message)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def _emit_finished(self):
        """Emit thread finished notification"""
        if self.on_finished:
            try:
                self.on_finished()
            except Exception as e:
                logger.error(f"Error in finished callback: {e}")

    def run_sync(self, timeout=120):
        """Synchronous version for web API - blocks until completion"""
        results = None
        error_info = None
        completed = threading.Event()
        
        def on_complete_sync(search_results):
            nonlocal results
            results = search_results
            completed.set()
        
        def on_error_sync(error_title, error_msg):
            nonlocal error_info
            error_info = (error_title, error_msg)
            completed.set()
        
        def on_finished_sync():
            completed.set()
        
        # Set up callbacks
        self.on_complete = on_complete_sync
        self.on_error = on_error_sync
        self.on_finished = on_finished_sync
        
        # Start search
        self.start()
        
        # Wait for completion
        if not completed.wait(timeout=timeout):
            self.stop()
            raise TimeoutError(f"Search timed out after {timeout} seconds")
        
        if error_info:
            error_title, error_msg = error_info
            raise Exception(f"{error_title}: {error_msg}")
        
        return results or []

    def run_async_with_callbacks(self, progress_callback=None, complete_callback=None, error_callback=None):
        """Async version with callback setup for web UI"""
        if progress_callback:
            self.on_progress = progress_callback
        if complete_callback:
            self.on_complete = complete_callback
        if error_callback:
            self.on_error = error_callback
        
        self.start()
        return self


class SearchManager:
    """Manages multiple search operations for web UI"""
    
    def __init__(self):
        self.active_searches = {}
        self.search_counter = 0
        self.lock = threading.Lock()

    def start_search(self, image_path, serp_api_key, imgur_client_id, social_media_only=False):
        """Start a new search and return search ID"""
        with self.lock:
            search_id = f"search_{self.search_counter}"
            self.search_counter += 1
            
            search_thread = SearchThread(
                image_path=image_path,
                serp_api_key=serp_api_key,
                imgur_client_id=imgur_client_id,
                social_media_only=social_media_only
            )
            
            # Auto-cleanup when finished
            original_on_finished = search_thread.on_finished
            def cleanup_finished():
                with self.lock:
                    if search_id in self.active_searches:
                        del self.active_searches[search_id]
                if original_on_finished:
                    original_on_finished()
            
            search_thread.on_finished = cleanup_finished
            self.active_searches[search_id] = search_thread
            
            return search_id, search_thread

    def get_search(self, search_id):
        """Get search thread by ID"""
        with self.lock:
            return self.active_searches.get(search_id)

    def stop_search(self, search_id):
        """Stop specific search"""
        with self.lock:
            search_thread = self.active_searches.get(search_id)
            if search_thread:
                search_thread.stop()
                return True
            return False

    def stop_all_searches(self):
        """Stop all active searches"""
        with self.lock:
            for search_thread in self.active_searches.values():
                search_thread.stop()
            
            # Wait for all to finish
            for search_thread in self.active_searches.values():
                search_thread.wait(timeout=2)
            
            self.active_searches.clear()

    def get_active_count(self):
        """Get number of active searches"""
        with self.lock:
            return len([s for s in self.active_searches.values() if s.is_running()])


# Global search manager instance
_search_manager = SearchManager()


def create_search_thread(image_path, serp_api_key, imgur_client_id, social_media_only=False):
    """Factory function to create configured search thread"""
    return SearchThread(
        image_path=image_path,
        serp_api_key=serp_api_key,
        imgur_client_id=imgur_client_id,
        social_media_only=social_media_only
    )


def search_image_sync(image_path, serp_api_key, imgur_client_id, social_media_only=False, timeout=120):
    """Synchronous search function for simple use cases"""
    search_thread = create_search_thread(image_path, serp_api_key, imgur_client_id, social_media_only)
    return search_thread.run_sync(timeout=timeout)


def search_image_async(image_path, serp_api_key, imgur_client_id, social_media_only=False, 
                      progress_callback=None, complete_callback=None, error_callback=None):
    """Asynchronous search function with callbacks"""
    search_thread = create_search_thread(image_path, serp_api_key, imgur_client_id, social_media_only)
    return search_thread.run_async_with_callbacks(progress_callback, complete_callback, error_callback)


def search_from_bytes(image_bytes, filename, serp_api_key, imgur_client_id, social_media_only=False):
    """Search using image bytes (for web uploads)"""
    import tempfile
    import os
    
    # Save bytes to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name
    
    try:
        return search_image_sync(temp_path, serp_api_key, imgur_client_id, social_media_only)
    finally:
        # Cleanup temp file
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Could not delete temp file {temp_path}: {e}")


def get_search_manager():
    """Get global search manager instance"""
    return _search_manager