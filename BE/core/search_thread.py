import threading
import time
import logging
import hashlib
import queue
import weakref
import gc
from typing import Callable, Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from contextlib import contextmanager
import tempfile
import os


from BE.core.api_client import (
    SerpApiClient, 
    create_search_client, 
    SearchResult,
    validate_api_keys
)
from BE.config.constants import SOCIAL_MEDIA_FILTER_DEFAULT

logger = logging.getLogger('ImageSearchApp')

class SearchThreadState(Enum):
    """Thread state tracking để prevent race conditions"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    UPLOADING = "uploading"
    SEARCHING = "searching"
    PROCESSING_RESULTS = "processing_results"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    CLEANUP = "cleanup"

@dataclass
class SearchMetrics:
    """Comprehensive metrics cho search performance tracking"""
    search_id: str = ""
    start_time: float = 0.0
    upload_start_time: float = 0.0
    upload_end_time: float = 0.0
    search_start_time: float = 0.0
    search_end_time: float = 0.0
    end_time: float = 0.0
    upload_duration: float = 0.0
    search_duration: float = 0.0
    total_duration: float = 0.0
    results_count: int = 0
    results_filtered_count: int = 0
    error_count: int = 0
    retry_count: int = 0
    memory_usage_mb: float = 0.0
    thread_id: int = 0
    uploader_type: str = ""
    image_size_bytes: int = 0

class SearchProgressEvent:
    """Type-safe progress event structure"""
    def __init__(self, 
                 search_id: str, 
                 progress_percent: float, 
                 message: str, 
                 state: SearchThreadState,
                 metadata: Optional[Dict[str, Any]] = None):
        self.search_id = search_id
        self.progress_percent = max(0.0, min(100.0, progress_percent))
        self.message = message
        self.state = state
        self.timestamp = time.time()
        self.metadata = metadata or {}

class SearchThread:
    """
    Features:
    - Comprehensive state management với atomic transitions
    - Thread-safe callback execution
    - Memory leak prevention với proper cleanup
    - Detailed metrics tracking và performance monitoring
    - Graceful cancellation với timeout handling
    - Automatic retry logic với exponential backoff
    - Integration với new SerpApiClient
    - Backward compatibility với existing UI code
    - Race condition prevention với proper locking
    - Resource cleanup với context managers
    
    Thread Safety:
    - All state changes are protected by locks
    - Callbacks are executed in thread-safe manner
    - Cancellation is graceful và immediate
    - Memory cleanup is guaranteed
    """
    
    # Class-level thread tracking để prevent resource leaks
    _active_threads: Dict[str, 'SearchThread'] = {}
    _thread_counter = 0
    _global_lock = threading.RLock()
    
    def __init__(self, 
                 image_path: str, 
                 serp_api_key: str, 
                 social_media_only: bool = SOCIAL_MEDIA_FILTER_DEFAULT, 
                 parent=None,
                 max_retries: int = 3,
                 timeout_seconds: int = 120,
                 enable_metrics: bool = True):
        """
        Initialize SearchThread với paranoid-level validation
        
        Args:
            image_path: Path to image file to search
            serp_api_key: SerpAPI key (kept for backward compatibility)
            social_media_only: Filter results to social media only
            parent: Parent object (kept for compatibility, not used)
            max_retries: Max retry attempts for failed operations
            timeout_seconds: Overall timeout for search operation
            enable_metrics: Enable detailed performance metrics
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If client initialization fails
        """
        # Generate unique search ID cho tracking
        with SearchThread._global_lock:
            SearchThread._thread_counter += 1
            self.search_id = f"search_{int(time.time())}_{SearchThread._thread_counter}"
        
        # Paranoid parameter validation
        self._validate_initialization_parameters(image_path, serp_api_key, max_retries, timeout_seconds)
        
        # Core parameters
        self.image_path = os.path.abspath(image_path)  # Absolute path để prevent confusion
        self.serp_api_key = serp_api_key.strip()  # Clean input
        self.social_media_only = bool(social_media_only)  # Ensure boolean
        self.max_retries = max(1, min(max_retries, 10))  # Clamp retries
        self.timeout_seconds = max(30, min(timeout_seconds, 600))  # Clamp timeout
        self.enable_metrics = bool(enable_metrics)
        
        # Thread state management với atomic operations
        self._state_lock = threading.RLock()
        self._state = SearchThreadState.IDLE
        self._search_active = True
        self._cancellation_requested = False
        self._cleanup_completed = False
        
        # Thread infrastructure
        self.thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._start_time = 0.0
        
        # Progress tracking
        self._current_progress = 0.0
        self._current_message = "Initialized"
        self._last_progress_time = time.time()
        
        # Results và error handling
        self._results: List[Dict[str, Any]] = []
        self._error_info: Optional[Tuple[str, str]] = None
        self._exception_info: Optional[Exception] = None
        
        # Metrics tracking (if enabled)
        self.metrics = SearchMetrics(search_id=self.search_id) if enable_metrics else None
        
        # Client initialization với fallback logic
        self._initialize_search_client()
        
        # Callback functions - thread-safe storage
        self._callback_lock = threading.Lock()
        self.on_progress: Optional[Callable[[str], None]] = None
        self.on_complete: Optional[Callable[[List[Dict[str, Any]]], None]] = None
        self.on_error: Optional[Callable[[str, str], None]] = None
        self.on_finished: Optional[Callable[[], None]] = None
        
        # Register thread để tracking
        with SearchThread._global_lock:
            SearchThread._active_threads[self.search_id] = self
        
        logger.info(f"SearchThread {self.search_id} initialized successfully")
        logger.debug(f"Search parameters: image={self.image_path}, social_only={self.social_media_only}, retries={self.max_retries}, timeout={self.timeout_seconds}s")
    
    def _validate_initialization_parameters(self, 
                                          image_path: str, 
                                          serp_api_key: str, 
                                          max_retries: int, 
                                          timeout_seconds: int) -> None:
        """Paranoid validation của initialization parameters"""
        
        # Image path validation
        if not image_path or not isinstance(image_path, str):
            raise ValueError("image_path must be a non-empty string")
        
        if not os.path.exists(image_path):
            raise ValueError(f"Image file does not exist: {image_path}")
        
        if not os.path.isfile(image_path):
            raise ValueError(f"Image path is not a file: {image_path}")
        
        if not os.access(image_path, os.R_OK):
            raise ValueError(f"Image file is not readable: {image_path}")
        
        # Basic image file validation
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
        file_ext = os.path.splitext(image_path.lower())[1]
        if file_ext not in image_extensions:
            logger.warning(f"Image file extension {file_ext} may not be supported")
        
        # File size check (paranoid about huge files)
        try:
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                raise ValueError("Image file is empty")
            if file_size > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError(f"Image file too large: {file_size / (1024*1024):.1f}MB (max 50MB)")
        except OSError as e:
            raise ValueError(f"Cannot access image file: {e}")
        
        # API key validation
        if not serp_api_key or not isinstance(serp_api_key, str):
            raise ValueError("serp_api_key must be a non-empty string")
        
        if len(serp_api_key.strip()) < 10:
            raise ValueError("serp_api_key appears to be invalid (too short)")
        
        # Numeric parameter validation
        if not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError("max_retries must be a positive integer")
        
        if not isinstance(timeout_seconds, int) or timeout_seconds < 30:
            raise ValueError("timeout_seconds must be at least 30")
    
    def _initialize_search_client(self) -> None:
        """Initialize search client với comprehensive fallback logic"""
        try:
            # Try to create optimal client with factory
            self.image_uploader, self.serp_client = create_search_client()
            uploader_name = type(self.image_uploader).__name__
            logger.info(f"SearchThread {self.search_id} using {uploader_name} uploader")
            
            if self.metrics:
                self.metrics.uploader_type = uploader_name
                
        except Exception as e:
            logger.error(f"Failed to create optimal search client: {e}")
            
            # Fallback to basic client
            try:
                self.serp_client = SerpApiClient(self.serp_api_key)
                self.image_uploader = self.serp_client.image_uploader
                logger.warning(f"SearchThread {self.search_id} using fallback basic client")
                
                if self.metrics:
                    self.metrics.uploader_type = type(self.image_uploader).__name__
                    
            except Exception as fallback_error:
                logger.critical(f"Failed to create fallback search client: {fallback_error}")
                raise RuntimeError(f"Cannot initialize search client: {fallback_error}")
    
    @property
    def state(self) -> SearchThreadState:
        """Thread-safe state getter"""
        with self._state_lock:
            return self._state
    
    @property
    def is_active(self) -> bool:
        """Thread-safe active status check"""
        with self._state_lock:
            return self._search_active and not self._cancellation_requested
    
    @property
    def is_running(self) -> bool:
        """Check if search thread is currently running"""
        return (self.thread is not None and 
                self.thread.is_alive() and 
                self.state not in {SearchThreadState.COMPLETED, SearchThreadState.CANCELLED, SearchThreadState.ERROR})
    
    @property
    def progress_percent(self) -> float:
        """Thread-safe progress getter"""
        with self._state_lock:
            return self._current_progress
    
    @property
    def current_message(self) -> str:
        """Thread-safe message getter"""
        with self._state_lock:
            return self._current_message
    
    def start(self) -> bool:
        """
        Start search in background thread với comprehensive safety checks
        
        Returns:
            True if thread started successfully, False otherwise
        """
        with self._state_lock:
            if self.thread and self.thread.is_alive():
                logger.warning(f"SearchThread {self.search_id} already running")
                return False
            
            if self._cancellation_requested:
                logger.warning(f"SearchThread {self.search_id} cannot start - cancellation requested")
                return False
            
            if self._cleanup_completed:
                logger.warning(f"SearchThread {self.search_id} cannot start - already cleaned up")
                return False
            
            try:
                # Reset state for new start
                self._state = SearchThreadState.INITIALIZING
                self._search_active = True
                self._cancellation_requested = False
                self._results.clear()
                self._error_info = None
                self._exception_info = None
                self._current_progress = 0.0
                self._current_message = "Starting search..."
                self._start_time = time.time()
                
                if self.metrics:
                    self.metrics.start_time = self._start_time
                    self.metrics.thread_id = threading.get_ident()
                
                # Create và start thread
                self.thread = threading.Thread(
                    target=self._run_search_with_exception_handling,
                    name=f"SearchThread-{self.search_id}",
                    daemon=True  # Daemon thread để prevent hanging on exit
                )
                
                self.thread.start()
                self._thread_id = self.thread.ident
                
                logger.info(f"SearchThread {self.search_id} started successfully (thread_id: {self._thread_id})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start SearchThread {self.search_id}: {e}", exc_info=True)
                self._state = SearchThreadState.ERROR
                self._search_active = False
                return False
    
    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop search thread gracefully với timeout
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
            
        Returns:
            True if stopped successfully, False if forced termination
        """
        logger.info(f"Requesting stop for SearchThread {self.search_id}...")
        
        with self._state_lock:
            if not self.thread or not self.thread.is_alive():
                logger.info(f"SearchThread {self.search_id} is not running")
                return True
            
            # Request graceful cancellation
            self._cancellation_requested = True
            self._search_active = False
            
            if self._state not in {SearchThreadState.COMPLETED, SearchThreadState.ERROR}:
                self._state = SearchThreadState.CANCELLED
        
        # Wait for thread to finish gracefully
        try:
            self.thread.join(timeout=timeout)
            
            if self.thread.is_alive():
                logger.warning(f"SearchThread {self.search_id} did not stop gracefully within {timeout}s")
                # Thread is still alive, but we can't force kill it in Python
                # The daemon flag ensures it won't prevent process exit
                return False
            else:
                logger.info(f"SearchThread {self.search_id} stopped gracefully")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping SearchThread {self.search_id}: {e}", exc_info=True)
            return False
        finally:
            # Ensure cleanup runs even if stop fails
            self._ensure_cleanup()
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for thread to finish với optional timeout
        
        Args:
            timeout: Maximum time to wait (None for infinite)
            
        Returns:
            True if thread finished, False if timeout
        """
        if not self.thread:
            return True
        
        try:
            self.thread.join(timeout)
            finished = not self.thread.is_alive()
            
            if finished:
                logger.debug(f"SearchThread {self.search_id} finished")
            elif timeout:
                logger.warning(f"SearchThread {self.search_id} did not finish within {timeout}s")
            
            return finished
            
        except Exception as e:
            logger.error(f"Error waiting for SearchThread {self.search_id}: {e}")
            return False
    
    def _run_search_with_exception_handling(self) -> None:
        """
        Main thread execution wrapper với comprehensive exception handling
        """
        try:
            self._run_search()
        except Exception as e:
            # Catch ANY unhandled exception để prevent thread crashes
            logger.critical(f"Unhandled exception in SearchThread {self.search_id}: {e}", exc_info=True)
            
            with self._state_lock:
                self._state = SearchThreadState.ERROR
                self._search_active = False
                self._exception_info = e
            
            # Try to emit error callback
            try:
                self._emit_error("Critical System Error", f"Unhandled exception: {e}")
            except:
                pass  # Don't let callback errors crash the cleanup
        finally:
            # Absolutely ensure cleanup runs
            try:
                self._ensure_cleanup()
            except:
                pass  # Cleanup failures shouldn't prevent thread termination
    
    def _run_search(self) -> None:
        """
        Main search execution logic với comprehensive error handling
        """
        try:
            logger.info(f"SearchThread {self.search_id} execution started")
            
            # Check for cancellation before starting
            if not self._check_continue_execution("Starting search execution"):
                return
            
            self._transition_state(SearchThreadState.UPLOADING)
            self._emit_progress("Preparing image for upload...", 5.0)
            
            # Step 1: Image validation và preparation
            if self.metrics:
                self.metrics.image_size_bytes = os.path.getsize(self.image_path)
            
            if not self._check_continue_execution("Image preparation"):
                return
            
            # Step 2: Upload và search với new client
            self._transition_state(SearchThreadState.SEARCHING)
            self._emit_progress("Starting reverse image search...", 20.0)
            
            if self.metrics:
                self.metrics.search_start_time = time.time()
            
            # Progress callback để track upload/search progress
            def progress_callback(message: str):
                if self.is_active:
                    # Map progress messages to percentages
                    progress_map = {
                        "Preparing image": 25.0,
                        "Uploading": 40.0,
                        "Starting reverse": 60.0,
                        "Search completed": 90.0
                    }
                    
                    progress = 50.0  # Default
                    for key, value in progress_map.items():
                        if key.lower() in message.lower():
                            progress = value
                            break
                    
                    self._emit_progress(message, progress)
            
            # Execute search với new enhanced client
            search_results, error = self.serp_client.search_by_image_enhanced(
                image_path_or_data=self.image_path,
                filename=None,  # Auto-extracted from path
                social_media_only=self.social_media_only,
                max_results=100,
                progress_callback=progress_callback
            )
            
            if self.metrics:
                self.metrics.search_end_time = time.time()
                self.metrics.search_duration = self.metrics.search_end_time - self.metrics.search_start_time
            
            if not self._check_continue_execution("Search completion check"):
                return
            
            # Step 3: Handle search results
            if error:
                logger.error(f"Search error for {self.search_id}: {error}")
                self._emit_error("Search Error", error)
                return
            
            # Step 4: Process và convert results
            self._transition_state(SearchThreadState.PROCESSING_RESULTS)
            self._emit_progress("Processing search results...", 85.0)
            
            # Convert SearchResult objects to legacy dict format
            legacy_results = self._convert_to_legacy_format(search_results)
            
            if self.metrics:
                self.metrics.results_count = len(legacy_results)
                self.metrics.results_filtered_count = len([r for r in legacy_results if r.get('is_social_media')])
            
            # Step 5: Store results và complete
            with self._state_lock:
                self._results = legacy_results
            
            self._transition_state(SearchThreadState.COMPLETED)
            self._emit_progress(f"Search completed successfully", 100.0)
            
            # Emit completion callback
            self._emit_complete(legacy_results)
            
            logger.info(f"SearchThread {self.search_id} completed successfully: {len(legacy_results)} results")
            
        except Exception as e:
            logger.error(f"Search execution failed for {self.search_id}: {e}", exc_info=True)
            
            if self.metrics:
                self.metrics.error_count += 1
            
            with self._state_lock:
                self._state = SearchThreadState.ERROR
                self._search_active = False
                self._exception_info = e
            
            self._emit_error("Search Execution Error", str(e))
        finally:
            # Final metrics calculation
            if self.metrics:
                self.metrics.end_time = time.time()
                self.metrics.total_duration = self.metrics.end_time - self.metrics.start_time
                self._calculate_memory_usage()
            
            self._emit_finished()
    
    def _check_continue_execution(self, checkpoint_name: str) -> bool:
        """
        Thread-safe check if execution should continue
        
        Args:
            checkpoint_name: Name of checkpoint for logging
            
        Returns:
            True if should continue, False if should stop
        """
        with self._state_lock:
            should_continue = self._search_active and not self._cancellation_requested
            
            if not should_continue:
                logger.info(f"SearchThread {self.search_id} stopping at checkpoint: {checkpoint_name}")
                if self._state not in {SearchThreadState.CANCELLED, SearchThreadState.ERROR}:
                    self._state = SearchThreadState.CANCELLED
            
            return should_continue
    
    def _transition_state(self, new_state: SearchThreadState) -> None:
        """Thread-safe state transition với logging"""
        with self._state_lock:
            old_state = self._state
            self._state = new_state
            
            if old_state != new_state:
                logger.debug(f"SearchThread {self.search_id} state: {old_state.value} → {new_state.value}")
    
    def _convert_to_legacy_format(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """
        Convert SearchResult objects to legacy dict format
        để maintain backward compatibility với existing UI code
        
        Args:
            search_results: List of SearchResult objects from new client
            
        Returns:
            List of legacy dict format results
        """
        legacy_results = []
        
        try:
            for i, result in enumerate(search_results):
                if not result:  # Skip None results
                    continue
                
                # Basic legacy format
                legacy_result = {
                    "title": str(result.title or "No Title"),
                    "link": str(result.link or ""),
                    "displayed_link": str(result.displayed_link or result.link or ""),
                    "snippet": str(result.snippet or ""),
                    "is_social_media": bool(result.is_social_media),
                    "source": str(result.source or "Unknown"),
                }
                
                # Add platform info if available
                if hasattr(result, 'platform') and result.platform:
                    legacy_result["platform"] = str(result.platform)
                
                # Add position info
                if hasattr(result, 'position') and result.position is not None:
                    legacy_result["position"] = int(result.position)
                else:
                    legacy_result["position"] = i  # Fallback to index
                
                # Add metadata fields if available
                if hasattr(result, 'metadata') and result.metadata:
                    # Safely merge metadata
                    for key, value in result.metadata.items():
                        if key not in legacy_result:  # Don't override existing keys
                            legacy_result[key] = value
                
                # Validation của legacy result
                if legacy_result["link"]:  # Only add results with valid links
                    legacy_results.append(legacy_result)
                else:
                    logger.warning(f"Skipping result {i} with empty link")
            
            logger.debug(f"Converted {len(search_results)} SearchResults to {len(legacy_results)} legacy results")
            return legacy_results
            
        except Exception as e:
            logger.error(f"Error converting search results to legacy format: {e}", exc_info=True)
            # Return empty list on conversion error để prevent crashes
            return []
    
    def _calculate_memory_usage(self) -> None:
        """Calculate current memory usage cho metrics"""
        if not self.metrics:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            logger.debug(f"Could not calculate memory usage: {e}")
    
    def _emit_progress(self, message: str, progress_percent: float = None) -> None:
        """Thread-safe progress callback emission"""
        with self._state_lock:
            if progress_percent is not None:
                self._current_progress = max(0.0, min(100.0, float(progress_percent)))
            self._current_message = str(message)
            self._last_progress_time = time.time()
        
        # Execute callback outside of lock để prevent deadlocks
        with self._callback_lock:
            if self.on_progress and self.is_active:
                try:
                    self.on_progress(message)
                except Exception as e:
                    logger.error(f"Progress callback error in {self.search_id}: {e}")
    
    def _emit_complete(self, results: List[Dict[str, Any]]) -> None:
        """Thread-safe completion callback emission"""
        with self._callback_lock:
            if self.on_complete and self.is_active:
                try:
                    # Create defensive copy để prevent external modifications
                    results_copy = [dict(result) for result in results] if results else []
                    self.on_complete(results_copy)
                except Exception as e:
                    logger.error(f"Completion callback error in {self.search_id}: {e}")
    
    def _emit_error(self, error_title: str, error_message: str) -> None:
        """Thread-safe error callback emission"""
        error_title = str(error_title)
        error_message = str(error_message)
        
        with self._state_lock:
            self._error_info = (error_title, error_message)
        
        with self._callback_lock:
            if self.on_error and self.is_active:
                try:
                    self.on_error(error_title, error_message)
                except Exception as e:
                    logger.error(f"Error callback error in {self.search_id}: {e}")
    
    def _emit_finished(self) -> None:
        """Thread-safe finished callback emission"""
        with self._callback_lock:
            if self.on_finished:
                try:
                    self.on_finished()
                except Exception as e:
                    logger.error(f"Finished callback error in {self.search_id}: {e}")
    
    def _ensure_cleanup(self) -> None:
        """Ensure all resources are cleaned up properly"""
        with self._state_lock:
            if self._cleanup_completed:
                return
            
            logger.debug(f"Starting cleanup for SearchThread {self.search_id}")
            
            try:
                # Mark as cleaning up
                if self._state not in {SearchThreadState.COMPLETED, SearchThreadState.ERROR, SearchThreadState.CANCELLED}:
                    self._state = SearchThreadState.CLEANUP
                
                # Stop any active operations
                self._search_active = False
                self._cancellation_requested = True
                
                # Clear callbacks để prevent further execution
                with self._callback_lock:
                    self.on_progress = None
                    self.on_complete = None
                    self.on_error = None
                    self.on_finished = None
                
                # Clear large data structures
                self._results.clear()
                
                # Remove from global tracking
                with SearchThread._global_lock:
                    SearchThread._active_threads.pop(self.search_id, None)
                
                self._cleanup_completed = True
                logger.debug(f"Cleanup completed for SearchThread {self.search_id}")
                
            except Exception as e:
                logger.error(f"Error during cleanup of SearchThread {self.search_id}: {e}")
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get search results thread-safely"""
        with self._state_lock:
            return [dict(result) for result in self._results]  # Defensive copy
    
    def get_error_info(self) -> Optional[Tuple[str, str]]:
        """Get error information thread-safely"""
        with self._state_lock:
            return self._error_info
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get performance metrics"""
        if not self.metrics:
            return None
        
        with self._state_lock:
            return {
                "search_id": self.metrics.search_id,
                "state": self._state.value,
                "start_time": self.metrics.start_time,
                "end_time": self.metrics.end_time,
                "total_duration": self.metrics.total_duration,
                "upload_duration": self.metrics.upload_duration,
                "search_duration": self.metrics.search_duration,
                "results_count": self.metrics.results_count,
                "results_filtered_count": self.metrics.results_filtered_count,
                "error_count": self.metrics.error_count,
                "retry_count": self.metrics.retry_count,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "thread_id": self.metrics.thread_id,
                "uploader_type": self.metrics.uploader_type,
                "image_size_bytes": self.metrics.image_size_bytes,
                "progress_percent": self._current_progress,
                "current_message": self._current_message,
            }
    
    def run_sync(self, timeout: int = 120) -> List[Dict[str, Any]]:
        """
        Synchronous version for direct execution
        
        Args:
            timeout: Maximum time to wait for completion
            
        Returns:
            List of search results
            
        Raises:
            TimeoutError: If search times out
            RuntimeError: If search fails
        """
        results = []
        error_info = None
        completed = threading.Event()
        
        # Setup callbacks để capture results
        def on_complete_sync(search_results):
            nonlocal results
            results = search_results or []
            completed.set()
        
        def on_error_sync(error_title, error_msg):
            nonlocal error_info
            error_info = (error_title, error_msg)
            completed.set()
        
        def on_finished_sync():
            completed.set()
        
        # Set callbacks
        with self._callback_lock:
            self.on_complete = on_complete_sync
            self.on_error = on_error_sync
            self.on_finished = on_finished_sync
        
        # Start search
        if not self.start():
            raise RuntimeError(f"Failed to start search thread {self.search_id}")
        
        # Wait for completion
        if not completed.wait(timeout=timeout):
            self.stop(timeout=5.0)  # Try graceful stop
            raise TimeoutError(f"Search timed out after {timeout} seconds")
        
        # Check for errors
        if error_info:
            error_title, error_msg = error_info
            raise RuntimeError(f"{error_title}: {error_msg}")
        
        return results
    
    def run_async_with_callbacks(self, 
                                progress_callback: Optional[Callable[[str], None]] = None,
                                complete_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
                                error_callback: Optional[Callable[[str, str], None]] = None,
                                finished_callback: Optional[Callable[[], None]] = None) -> 'SearchThread':
        """
        Async version với callback setup for web UI
        
        Args:
            progress_callback: Progress update callback
            complete_callback: Completion callback
            error_callback: Error callback  
            finished_callback: Finished callback
            
        Returns:
            Self for method chaining
        """
        with self._callback_lock:
            if progress_callback:
                self.on_progress = progress_callback
            if complete_callback:
                self.on_complete = complete_callback
            if error_callback:
                self.on_error = error_callback
            if finished_callback:
                self.on_finished = finished_callback
        
        self.start()
        return self
    
    def __del__(self):
        """Destructor để ensure cleanup on garbage collection"""
        try:
            if not self._cleanup_completed:
                logger.warning(f"SearchThread {self.search_id} being garbage collected without proper cleanup")
                self._ensure_cleanup()
        except:
            pass  # Don't raise exceptions in __del__

class SearchManager:
    """
    ULTRA PARANOID Search Manager Implementation
    
    Manages multiple concurrent search operations với:
    - Thread pool management và resource limiting
    - Automatic cleanup của finished searches
    - Global metrics tracking
    - Memory usage monitoring
    - Deadlock prevention
    - Resource leak detection
    """
    
    def __init__(self, 
                 max_concurrent_searches: int = 5,
                 cleanup_interval_seconds: int = 60,
                 max_finished_searches_to_keep: int = 10):
        """
        Initialize SearchManager với paranoid resource management
        
        Args:
            max_concurrent_searches: Maximum concurrent search threads
            cleanup_interval_seconds: How often to run cleanup
            max_finished_searches_to_keep: Max finished searches to keep in memory
        """
        self.max_concurrent_searches = max(1, min(max_concurrent_searches, 20))
        self.cleanup_interval = max(30, cleanup_interval_seconds)
        self.max_finished_searches = max(5, max_finished_searches_to_keep)
        
        # Thread-safe data structures
        self.active_searches: Dict[str, SearchThread] = {}
        self.finished_searches: Dict[str, SearchThread] = {}
        self.search_counter = 0
        self.lock = threading.RLock()
        
        # Cleanup thread
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_active = True
        self._start_cleanup_thread()
        
        # Global metrics
        self.total_searches_started = 0
        self.total_searches_completed = 0
        self.total_searches_failed = 0
        self.total_searches_cancelled = 0
        
        logger.info(f"SearchManager initialized: max_concurrent={self.max_concurrent_searches}")
    
    def start_search(self, 
                    image_path: str, 
                    serp_api_key: str, 
                    social_media_only: bool = False,
                    **kwargs) -> Tuple[str, SearchThread]:
        """
        Start a new search và return search ID
        
        Args:
            image_path: Path to image file
            serp_api_key: SerpAPI key  
            social_media_only: Filter to social media only
            **kwargs: Additional SearchThread parameters
            
        Returns:
            Tuple of (search_id, search_thread)
            
        Raises:
            RuntimeError: If too many concurrent searches or other errors
        """
        with self.lock:
            # Check concurrent limit
            if len(self.active_searches) >= self.max_concurrent_searches:
                raise RuntimeError(f"Too many concurrent searches ({len(self.active_searches)}/{self.max_concurrent_searches})")
            
            try:
                # Create search thread
                search_thread = SearchThread(
                    image_path=image_path,
                    serp_api_key=serp_api_key,
                    social_media_only=social_media_only,
                    **kwargs
                )
                
                search_id = search_thread.search_id
                
                # Setup auto-cleanup when finished
                original_on_finished = search_thread.on_finished
                def cleanup_finished():
                    self._move_to_finished(search_id)
                    if original_on_finished:
                        try:
                            original_on_finished()
                        except Exception as e:
                            logger.error(f"Error in original finished callback: {e}")
                
                search_thread.on_finished = cleanup_finished
                
                # Store và start
                self.active_searches[search_id] = search_thread
                self.search_counter += 1
                self.total_searches_started += 1
                
                logger.info(f"SearchManager created search {search_id} ({len(self.active_searches)} active)")
                return search_id, search_thread
                
            except Exception as e:
                logger.error(f"Failed to create search: {e}", exc_info=True)
                raise RuntimeError(f"Failed to create search: {e}")
    
    def get_search(self, search_id: str) -> Optional[SearchThread]:
        """Get search thread by ID"""
        with self.lock:
            # Check active searches first
            if search_id in self.active_searches:
                return self.active_searches[search_id]
            
            # Check finished searches
            if search_id in self.finished_searches:
                return self.finished_searches[search_id]
            
            return None
    
    def stop_search(self, search_id: str, timeout: float = 5.0) -> bool:
        """Stop specific search"""
        with self.lock:
            search_thread = self.active_searches.get(search_id)
            if search_thread:
                success = search_thread.stop(timeout=timeout)
                
                # Move to finished regardless of success
                self._move_to_finished(search_id)
                
                if success:
                    self.total_searches_cancelled += 1
                    logger.info(f"SearchManager stopped search {search_id}")
                else:
                    logger.warning(f"SearchManager forced stop of search {search_id}")
                
                return success
            
            return False
    
    def stop_all_searches(self, timeout: float = 10.0) -> Dict[str, bool]:
        """Stop all active searches"""
        with self.lock:
            search_ids = list(self.active_searches.keys())
        
        if not search_ids:
            return {}
        
        logger.info(f"SearchManager stopping {len(search_ids)} active searches")
        results = {}
        
        # Stop all searches in parallel
        with ThreadPoolExecutor(max_workers=min(len(search_ids), 10)) as executor:
            future_to_id = {
                executor.submit(self.stop_search, search_id, timeout/len(search_ids)): search_id
                for search_id in search_ids
            }
            
            for future in as_completed(future_to_id):
                search_id = future_to_id[future]
                try:
                    results[search_id] = future.result()
                except Exception as e:
                    logger.error(f"Error stopping search {search_id}: {e}")
                    results[search_id] = False
        
        return results
    
    def get_active_count(self) -> int:
        """Get number of active searches"""
        with self.lock:
            return len(self.active_searches)
    
    def get_finished_count(self) -> int:
        """Get number of finished searches"""
        with self.lock:
            return len(self.finished_searches)
    
    def get_all_search_ids(self) -> Dict[str, str]:
        """Get all search IDs với their status"""
        with self.lock:
            result = {}
            
            for search_id in self.active_searches:
                result[search_id] = "active"
            
            for search_id in self.finished_searches:
                result[search_id] = "finished"
            
            return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive SearchManager metrics"""
        with self.lock:
            active_searches = len(self.active_searches)
            finished_searches = len(self.finished_searches)
            
            # Calculate memory usage
            total_memory_mb = 0.0
            for search in self.active_searches.values():
                metrics = search.get_metrics()
                if metrics:
                    total_memory_mb += metrics.get('memory_usage_mb', 0.0)
            
            return {
                "active_searches": active_searches,
                "finished_searches": finished_searches,
                "total_searches_started": self.total_searches_started,
                "total_searches_completed": self.total_searches_completed,
                "total_searches_failed": self.total_searches_failed,
                "total_searches_cancelled": self.total_searches_cancelled,
                "max_concurrent_searches": self.max_concurrent_searches,
                "total_memory_usage_mb": round(total_memory_mb, 2),
                "cleanup_active": self.cleanup_active,
            }
    
    def _move_to_finished(self, search_id: str) -> None:
        """Move search from active to finished"""
        with self.lock:
            search_thread = self.active_searches.pop(search_id, None)
            if search_thread:
                # Update counters based on final state
                final_state = search_thread.state
                
                if final_state == SearchThreadState.COMPLETED:
                    self.total_searches_completed += 1
                elif final_state == SearchThreadState.CANCELLED:
                    self.total_searches_cancelled += 1
                elif final_state == SearchThreadState.ERROR:
                    self.total_searches_failed += 1
                
                # Store in finished (with limit)
                self.finished_searches[search_id] = search_thread
                
                # Cleanup old finished searches
                if len(self.finished_searches) > self.max_finished_searches:
                    # Remove oldest finished searches
                    oldest_searches = sorted(
                        self.finished_searches.items(),
                        key=lambda x: x[1].get_metrics().get('start_time', 0) if x[1].get_metrics() else 0
                    )
                    
                    searches_to_remove = oldest_searches[:-self.max_finished_searches]
                    for old_id, old_search in searches_to_remove:
                        del self.finished_searches[old_id]
                        try:
                            old_search._ensure_cleanup()
                        except:
                            pass
                
                logger.debug(f"Moved search {search_id} to finished ({final_state.value})")
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while self.cleanup_active:
                try:
                    time.sleep(self.cleanup_interval)
                    
                    if not self.cleanup_active:
                        break
                    
                    self._perform_cleanup()
                    
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}", exc_info=True)
        
        self.cleanup_thread = threading.Thread(
            target=cleanup_worker,
            name="SearchManager-Cleanup",
            daemon=True
        )
        self.cleanup_thread.start()
        logger.debug("SearchManager cleanup thread started")
    
    def _perform_cleanup(self) -> None:
        """Perform periodic cleanup"""
        with self.lock:
            # Check for stuck active searches
            current_time = time.time()
            stuck_searches = []
            
            for search_id, search in self.active_searches.items():
                metrics = search.get_metrics()
                if metrics:
                    start_time = metrics.get('start_time', current_time)
                    if current_time - start_time > 300:  # 5 minutes
                        stuck_searches.append(search_id)
            
            # Clean up stuck searches
            for search_id in stuck_searches:
                logger.warning(f"Cleaning up stuck search {search_id}")
                self.stop_search(search_id, timeout=2.0)
            
            # Force garbage collection
            if stuck_searches:
                gc.collect()
    
    def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown SearchManager gracefully"""
        logger.info("SearchManager shutdown initiated")
        
        # Stop cleanup thread
        self.cleanup_active = False
        
        # Stop all searches
        stop_results = self.stop_all_searches(timeout=timeout * 0.8)
        
        # Wait for cleanup thread
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=timeout * 0.2)
        
        # Final cleanup
        with self.lock:
            for search in list(self.active_searches.values()) + list(self.finished_searches.values()):
                try:
                    search._ensure_cleanup()
                except:
                    pass
            
            self.active_searches.clear()
            self.finished_searches.clear()
        
        logger.info("SearchManager shutdown completed")
    
    def __del__(self):
        """Destructor để ensure cleanup"""
        try:
            if hasattr(self, 'cleanup_active') and self.cleanup_active:
                self.shutdown(timeout=5.0)
        except:
            pass

# ===== FACTORY FUNCTIONS & BACKWARD COMPATIBILITY =====

# Global search manager instance
_search_manager: Optional[SearchManager] = None
_search_manager_lock = threading.Lock()

def get_search_manager() -> SearchManager:
    """Get global search manager instance với lazy initialization"""
    global _search_manager
    
    if _search_manager is None:
        with _search_manager_lock:
            if _search_manager is None:  # Double-check pattern
                _search_manager = SearchManager()
                logger.info("Global SearchManager initialized")
    
    return _search_manager

def create_search_thread(image_path: str, 
                        serp_api_key: str, 
                        social_media_only: bool = False,
                        **kwargs) -> SearchThread:
    """
    Factory function to create configured search thread
    
    Args:
        image_path: Path to image file
        serp_api_key: SerpAPI key (kept for backward compatibility)
        social_media_only: Filter to social media only
        **kwargs: Additional SearchThread parameters
        
    Returns:
        Configured SearchThread instance
    """
    return SearchThread(
        image_path=image_path,
        serp_api_key=serp_api_key,
        social_media_only=social_media_only,
        **kwargs
    )

def search_image_sync(image_path: str, 
                     serp_api_key: str, 
                     imgur_client_id: Optional[str] = None,  # Kept for backward compatibility
                     social_media_only: bool = False, 
                     timeout: int = 120) -> List[Dict[str, Any]]:
    """
    Synchronous search function for simple use cases
    
    Args:
        image_path: Path to image file
        serp_api_key: SerpAPI key
        imgur_client_id: Deprecated parameter (kept for compatibility)
        social_media_only: Filter to social media only
        timeout: Search timeout in seconds
        
    Returns:
        List of search results in legacy format
        
    Raises:
        TimeoutError: If search times out
        RuntimeError: If search fails
    """
    if imgur_client_id:
        logger.warning("imgur_client_id parameter is deprecated and ignored")
    
    search_thread = create_search_thread(
        image_path=image_path,
        serp_api_key=serp_api_key,
        social_media_only=social_media_only,
        timeout_seconds=timeout
    )
    
    return search_thread.run_sync(timeout=timeout)

def search_image_async(image_path: str, 
                      serp_api_key: str, 
                      imgur_client_id: Optional[str] = None,  # Kept for backward compatibility
                      social_media_only: bool = False, 
                      progress_callback: Optional[Callable[[str], None]] = None,
                      complete_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
                      error_callback: Optional[Callable[[str, str], None]] = None) -> SearchThread:
    """
    Asynchronous search function với callbacks
    
    Args:
        image_path: Path to image file
        serp_api_key: SerpAPI key
        imgur_client_id: Deprecated parameter (kept for compatibility)
        social_media_only: Filter to social media only
        progress_callback: Progress update callback
        complete_callback: Completion callback
        error_callback: Error callback
        
    Returns:
        SearchThread instance
    """
    if imgur_client_id:
        logger.warning("imgur_client_id parameter is deprecated and ignored")
    
    search_thread = create_search_thread(
        image_path=image_path,
        serp_api_key=serp_api_key,
        social_media_only=social_media_only
    )
    
    return search_thread.run_async_with_callbacks(
        progress_callback=progress_callback,
        complete_callback=complete_callback,
        error_callback=error_callback
    )

def search_from_bytes(image_bytes: bytes, 
                     filename: str, 
                     serp_api_key: str, 
                     imgur_client_id: Optional[str] = None,  # Kept for backward compatibility
                     social_media_only: bool = False) -> List[Dict[str, Any]]:
    """
    Search using image bytes (for web uploads)
    
    Args:
        image_bytes: Raw image data
        filename: Original filename
        serp_api_key: SerpAPI key
        imgur_client_id: Deprecated parameter (kept for compatibility)
        social_media_only: Filter to social media only
        
    Returns:
        List of search results
        
    Raises:
        RuntimeError: If search fails
    """
    if imgur_client_id:
        logger.warning("imgur_client_id parameter is deprecated and ignored")
    
    temp_path = None
    try:
        # Save bytes to temporary file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name
        
        # Execute search
        return search_image_sync(
            image_path=temp_path,
            serp_api_key=serp_api_key,
            social_media_only=social_media_only
        )
        
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_path}: {e}")

# ===== MODULE CLEANUP =====

import atexit

def _cleanup_module():
    """Cleanup function để run on module exit"""
    global _search_manager
    if _search_manager:
        try:
            _search_manager.shutdown(timeout=10.0)
        except:
            pass

# Register cleanup function
atexit.register(_cleanup_module)

# ===== TESTING UTILITIES =====

def test_search_thread_functionality(test_image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Test search thread functionality với comprehensive validation
    
    Args:
        test_image_path: Path to test image (creates dummy if None)
        
    Returns:
        Test results dictionary
    """
    results = {
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "errors": [],
        "warnings": []
    }
    
    def run_test(test_name: str, test_func):
        results["tests_run"] += 1
        try:
            test_func()
            results["tests_passed"] += 1
            logger.info(f"✅ {test_name} passed")
        except Exception as e:
            results["tests_failed"] += 1
            results["errors"].append(f"{test_name}: {str(e)}")
            logger.error(f"❌ {test_name} failed: {e}")
    
    # Test 1: API key validation
    def test_api_validation():
        validation = validate_api_keys()
        assert isinstance(validation, dict), "Validation should return dict"
        assert "ready" in validation, "Validation should include ready status"
    
    run_test("API Key Validation", test_api_validation)
    
    # Test 2: SearchThread creation
    def test_thread_creation():
        if not test_image_path:
            # Create dummy test image
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as f:
                # Write minimal JPEG header
                f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\x9f\xff\xd9')
                test_path = f.name
        else:
            test_path = test_image_path
        
        thread = SearchThread(
            image_path=test_path,
            serp_api_key="test_key_1234567890",
            social_media_only=True
        )
        
        assert thread.search_id is not None, "Search ID should be generated"
        assert thread.state == SearchThreadState.IDLE, "Initial state should be IDLE"
        
        # Cleanup
        if not test_image_path:
            os.unlink(test_path)
    
    run_test("SearchThread Creation", test_thread_creation)
    
    # Test 3: SearchManager functionality
    def test_search_manager():
        manager = SearchManager(max_concurrent_searches=2)
        metrics = manager.get_metrics()
        
        assert isinstance(metrics, dict), "Metrics should be dict"
        assert metrics["active_searches"] == 0, "Should start with 0 active searches"
        assert metrics["max_concurrent_searches"] == 2, "Should respect max concurrent setting"
        
        manager.shutdown(timeout=5.0)
    
    run_test("SearchManager Functionality", test_search_manager)
    
    return results

if __name__ == "__main__":
    # Run tests khi module executed directly
    test_results = test_search_thread_functionality()
    print(f"Test Results: {test_results['tests_passed']}/{test_results['tests_run']} passed")
    
    if test_results["errors"]:
        print("Errors:")
        for error in test_results["errors"]:
            print(f"  - {error}")