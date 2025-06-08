# BE/core/api_client.py - ULTRA PARANOID EDITION
import requests
import json
import os
import datetime
import time
import logging
import hashlib
import threading
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from urllib.parse import urlparse, parse_qs
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from collections import defaultdict

# Import our image uploaders
from BE.core.imgbb_client import ImgBBClient
from BE.config.constants import API_TIMEOUT_SEC, SOCIAL_MEDIA_DOMAINS

logger = logging.getLogger('ImageSearchApp')

class SearchResultSource(Enum):
    """Enum for tracking search result sources"""
    IMAGE_RESULTS = "image_results"
    ORGANIC_RESULTS = "organic_results" 
    INLINE_IMAGES = "inline_images"
    IMAGE_CONTENT = "image_content"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    UNKNOWN = "unknown"

@dataclass
class SerpApiMetrics:
    """Tracking metrics cho SerpAPI usage"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_results_found: int = 0
    average_response_time: float = 0.0
    rate_limit_hits: int = 0
    last_request_time: float = 0.0
    quota_remaining: Optional[int] = None
    errors_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

@dataclass 
class SearchResult:
    """Standardized search result structure"""
    title: str
    link: str
    displayed_link: str
    snippet: str
    is_social_media: bool
    source: str
    platform: Optional[str] = None
    thumbnail: Optional[str] = None
    position: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class ImgurClient:
    """Legacy Imgur client cho backward compatibility"""
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.api_url = "https://api.imgur.com/3/image"
        logger.warning("Using deprecated Imgur client - consider switching to ImgBB")
    
    def upload_image(self, image_path: str) -> Optional[str]:
        """Legacy method - chỉ để backward compatibility"""
        try:
            from BE.core.image_processing import optimize_image
            binary_data = optimize_image(image_path)
            return self.upload_image_from_bytes(binary_data, os.path.basename(image_path))
        except Exception as e:
            logger.error(f"Imgur upload from path failed: {e}")
            return None
    
    def upload_image_from_bytes(self, image_bytes: bytes, filename: str = "upload.jpg") -> Optional[str]:
        """Legacy Imgur upload method"""
        try:
            headers = {'Authorization': f'Client-ID {self.client_id}'}
            response = requests.post(
                self.api_url,
                headers=headers,
                files={'image': (filename, image_bytes, 'image/jpeg')},
                timeout=API_TIMEOUT_SEC
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data['data']['link']
            
            logger.error(f"Imgur upload failed: {response.status_code} - {response.text}")
            return None
            
        except Exception as e:
            logger.error(f"Imgur upload error: {e}")
            return None

def create_image_uploader(force_type: Optional[str] = None):
    """
    Factory function để tạo image uploader với intelligent fallback
    
    Args:
        force_type: Force specific uploader type ('imgbb', 'imgur', None for auto)
        
    Returns:
        Image uploader instance
        
    Raises:
        ValueError: If no valid uploader can be created
    """
    from BE.config.settings import IMGBB_API_KEY, IMGUR_CLIENT_ID
    
    # Force specific type nếu requested
    if force_type == 'imgbb':
        if not IMGBB_API_KEY or not IMGBB_API_KEY.strip():
            raise ValueError("IMGBB_API_KEY is required but not configured")
        return ImgBBClient(IMGBB_API_KEY)
    
    elif force_type == 'imgur':
        if not IMGUR_CLIENT_ID or not IMGUR_CLIENT_ID.strip():
            raise ValueError("IMGUR_CLIENT_ID is required but not configured") 
        return ImgurClient(IMGUR_CLIENT_ID)
    
    # Auto-selection với priority order
    uploaders_to_try = []
    
    # ImgBB first (preferred - more reliable)
    if IMGBB_API_KEY and IMGBB_API_KEY.strip():
        uploaders_to_try.append(('ImgBB', lambda: ImgBBClient(IMGBB_API_KEY)))
    
    # Imgur second (fallback)
    if IMGUR_CLIENT_ID and IMGUR_CLIENT_ID.strip():
        uploaders_to_try.append(('Imgur', lambda: ImgurClient(IMGUR_CLIENT_ID)))
    
    # Try each uploader
    last_error = None
    for uploader_name, uploader_factory in uploaders_to_try:
        try:
            uploader = uploader_factory()
            logger.info(f"Successfully initialized {uploader_name} as image uploader")
            return uploader
        except Exception as e:
            last_error = e
            logger.warning(f"Failed to initialize {uploader_name}: {e}")
    
    # No uploader available
    error_msg = f"No image uploader available. Last error: {last_error}"
    error_msg += "\nPlease configure IMGBB_API_KEY or IMGUR_CLIENT_ID in your environment."
    raise ValueError(error_msg)

class SerpApiClient:
    """
    ULTRA PARANOID SerpAPI Client với comprehensive error handling,
    rate limiting, metrics tracking, và fallback mechanisms.
    
    Features:
    - Multiple image uploader support (ImgBB primary, Imgur fallback)
    - Comprehensive error handling và retry logic
    - Rate limiting và quota tracking
    - Detailed metrics và performance monitoring
    - Social media filtering với platform detection
    - Result deduplication và validation
    - Request/response logging cho debugging
    - Async/sync operation modes
    """
    
    def __init__(self, 
                 api_key: str, 
                 image_uploader=None, 
                 max_retries: int = 3,
                 rate_limit_per_minute: int = 50,
                 enable_caching: bool = True):
        """
        Initialize SerpAPI client với paranoid configuration
        
        Args:
            api_key: SerpAPI key
            image_uploader: Custom image uploader (auto-created if None)
            max_retries: Max retry attempts for failed requests
            rate_limit_per_minute: Conservative rate limit
            enable_caching: Enable request caching for repeated searches
        """
        # Validation với existential dread level
        if not api_key or not isinstance(api_key, str) or len(api_key.strip()) < 10:
            raise ValueError("SerpAPI key is invalid. Expected non-empty string with length > 10")
        
        self.api_key = api_key.strip()
        self.api_url = "https://serpapi.com/search"
        self.max_retries = max(1, min(max_retries, 10))  # Clamp between 1-10
        self.timeout = API_TIMEOUT_SEC * 2  # Extra conservative timeout
        self.rate_limit_per_minute = max(1, min(rate_limit_per_minute, 100))
        self.enable_caching = enable_caching
        
        # Image uploader với fallback logic
        try:
            self.image_uploader = image_uploader or create_image_uploader()
            logger.info(f"SerpApiClient using {type(self.image_uploader).__name__} for image uploads")
        except Exception as e:
            logger.error(f"Failed to initialize image uploader: {e}")
            raise ValueError(f"Cannot initialize image uploader: {e}")
        
        # Metrics tracking (because I'm paranoid about performance)
        self.metrics = SerpApiMetrics()
        self._metrics_lock = threading.Lock()
        
        # Rate limiting infrastructure
        self._request_times = []
        self._rate_limit_lock = threading.Lock()
        self._last_request_time = 0
        
        # Caching (paranoid about repeated requests)
        self._cache = {} if enable_caching else None
        self._cache_lock = threading.Lock() if enable_caching else None
        self._cache_ttl = 3600  # 1 hour cache TTL
        
        # Social media detection setup
        self.social_media_domains = set(SOCIAL_MEDIA_DOMAINS)
        self.platform_keywords = {
            'facebook': ['facebook.com', 'fb.com', 'fb.me'],
            'instagram': ['instagram.com', 'instagr.am'],
            'twitter': ['twitter.com', 'x.com', 't.co'],
            'linkedin': ['linkedin.com', 'lnkd.in'],
            'tiktok': ['tiktok.com', 'tiktok.app'],
            'youtube': ['youtube.com', 'youtu.be'],
            'pinterest': ['pinterest.com', 'pin.it'],
            'reddit': ['reddit.com', 'redd.it'],
            'tumblr': ['tumblr.com'],
            'snapchat': ['snapchat.com'],
        }
        
        logger.info(f"SerpApiClient initialized successfully with {len(self.social_media_domains)} social media domains")
    
    def search_by_image(self, 
                       image_url: str, 
                       social_media_only: bool = True,
                       max_results: int = 100,
                       enable_deduplication: bool = True) -> Tuple[List[SearchResult], Optional[str]]:
        """
        Core image search method với comprehensive error handling
        
        Args:
            image_url: Public URL of image to search
            social_media_only: Filter results to social media only
            max_results: Maximum results to return (clamped to SerpAPI limits)
            enable_deduplication: Remove duplicate results
            
        Returns:
            Tuple of (results_list, error_message)
            - results_list: List of SearchResult objects
            - error_message: None if successful, error string if failed
        """
        request_start_time = time.time()
        search_id = hashlib.md5(f"{image_url}_{social_media_only}_{max_results}".encode()).hexdigest()[:8]
        
        try:
            # Validation với paranoid level
            self._validate_search_parameters(image_url, max_results)
            
            # Check cache first (nếu enabled)
            if self.enable_caching:
                cached_result = self._get_cached_result(search_id)
                if cached_result:
                    logger.info(f"Returning cached result for search {search_id}")
                    return cached_result, None
            
            # Rate limiting check
            self._enforce_rate_limits()
            
            # Prepare search parameters
            search_params = self._prepare_search_params(image_url, max_results)
            
            # Execute search với retry logic
            raw_response = self._execute_search_with_retries(search_params, search_id)
            
            if not raw_response:
                error_msg = "Failed to get valid response from SerpAPI after all retries"
                self._update_metrics(request_start_time, success=False, error_type="no_response")
                return [], error_msg
            
            # Process và validate results
            search_results = self._process_search_response(
                raw_response, 
                social_media_only, 
                enable_deduplication
            )
            
            # Cache result (nếu enabled)
            if self.enable_caching and search_results:
                self._cache_result(search_id, search_results)
            
            # Update metrics
            self._update_metrics(
                request_start_time, 
                success=True, 
                results_count=len(search_results)
            )
            
            logger.info(f"Search {search_id} completed successfully: {len(search_results)} results")
            return search_results, None
            
        except Exception as e:
            error_msg = f"Search failed for {search_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self._update_metrics(request_start_time, success=False, error_type=type(e).__name__)
            return [], error_msg
    
    def search_by_image_enhanced(self, 
                                image_path_or_data: Union[str, bytes], 
                                filename: Optional[str] = None,
                                social_media_only: bool = True,
                                max_results: int = 100,
                                progress_callback: Optional[Callable[[str], None]] = None) -> Tuple[List[SearchResult], Optional[str]]:
        """
        Enhanced search method hỗ trợ cả file path và bytes với upload
        
        Args:
            image_path_or_data: File path string hoặc image bytes
            filename: Filename hint (auto-generated if None)
            social_media_only: Filter to social media only
            max_results: Max results to return
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple of (results_list, error_message)
        """
        upload_start_time = time.time()
        
        try:
            # Progress callback helper
            def update_progress(message: str):
                if progress_callback:
                    try:
                        progress_callback(message)
                    except Exception as e:
                        logger.warning(f"Progress callback failed: {e}")
            
            update_progress("Preparing image for upload...")
            
            # Determine filename
            if isinstance(image_path_or_data, (str, os.PathLike)):
                # File path case
                image_path = str(image_path_or_data)
                if not filename:
                    filename = os.path.basename(image_path)
                
                update_progress(f"Uploading image: {filename}...")
                image_url = self.image_uploader.upload_image(image_path)
                
            elif isinstance(image_path_or_data, bytes):
                # Bytes case
                if not filename:
                    timestamp = int(time.time())
                    filename = f"uploaded_image_{timestamp}.jpg"
                
                update_progress(f"Uploading image data: {filename}...")
                image_url = self.image_uploader.upload_image_from_bytes(image_path_or_data, filename)
                
            else:
                raise ValueError("image_path_or_data must be file path string or bytes")
            
            # Validate upload success
            if not image_url:
                error_msg = f"Failed to upload {filename} to image hosting service"
                logger.error(error_msg)
                return [], error_msg
            
            upload_time = time.time() - upload_start_time
            logger.info(f"Image upload successful in {upload_time:.2f}s: {image_url}")
            
            update_progress("Starting reverse image search...")
            
            # Proceed với normal search
            search_results, search_error = self.search_by_image(
                image_url, 
                social_media_only, 
                max_results, 
                enable_deduplication=True
            )
            
            if search_error:
                return [], search_error
            
            update_progress(f"Search completed: {len(search_results)} results found")
            return search_results, None
            
        except Exception as e:
            error_msg = f"Enhanced search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return [], error_msg
    
    def _validate_search_parameters(self, image_url: str, max_results: int) -> None:
        """Paranoid validation của search parameters"""
        if not image_url or not isinstance(image_url, str):
            raise ValueError("image_url must be a non-empty string")
        
        # Basic URL validation
        try:
            parsed = urlparse(image_url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError("image_url must be a valid URL with scheme and domain")
        except Exception as e:
            raise ValueError(f"Invalid image_url format: {e}")
        
        # Max results validation
        if not isinstance(max_results, int) or max_results < 1:
            raise ValueError("max_results must be a positive integer")
        
        if max_results > 200:  # SerpAPI practical limit
            logger.warning(f"max_results {max_results} clamped to 200 (SerpAPI limit)")
            max_results = 200
    
    def _enforce_rate_limits(self) -> None:
        """Paranoid rate limiting enforcement"""
        current_time = time.time()
        
        with self._rate_limit_lock:
            # Clean old requests (older than 1 minute)
            cutoff_time = current_time - 60
            self._request_times = [t for t in self._request_times if t > cutoff_time]
            
            # Check if we're at rate limit
            if len(self._request_times) >= self.rate_limit_per_minute:
                oldest_request = min(self._request_times)
                wait_time = 60 - (current_time - oldest_request)
                
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, sleeping {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    self.metrics.rate_limit_hits += 1
            
            # Minimum interval between requests (be nice to SerpAPI)
            time_since_last = current_time - self._last_request_time
            if time_since_last < 1.0:  # Min 1 second between requests
                sleep_time = 1.0 - time_since_last
                time.sleep(sleep_time)
            
            # Record this request
            self._request_times.append(time.time())
            self._last_request_time = time.time()
    
    def _prepare_search_params(self, image_url: str, max_results: int) -> Dict[str, Any]:
        """Prepare optimal search parameters cho SerpAPI"""
        params = {
            "api_key": self.api_key,
            "engine": "google_reverse_image",
            "image_url": image_url,
            "num": min(max_results, 100),  # SerpAPI limit per request
            "ijn": 0,  # Start from first page
            "hl": "en",  # Language
            "gl": "us",  # Geographic location
            "safe": "off",  # Include all results
        }
        
        logger.debug(f"Search params prepared: {params}")
        return params
    
    def _execute_search_with_retries(self, 
                                   search_params: Dict[str, Any], 
                                   search_id: str) -> Optional[Dict[str, Any]]:
        """Execute search với comprehensive retry logic"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"SerpAPI request attempt {attempt + 1}/{self.max_retries} for {search_id}")
                
                response = requests.get(
                    self.api_url,
                    params=search_params,
                    timeout=self.timeout,
                    headers={
                        'User-Agent': 'Enhanced-Toolkit-v2.0-SerpAPI-Client',
                        'Accept': 'application/json',
                    }
                )
                
                # Handle response status codes
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Check for SerpAPI error responses
                        if "error" in data:
                            error_msg = data["error"]
                            logger.error(f"SerpAPI returned error: {error_msg}")
                            
                            # Don't retry on certain errors
                            if any(err_type in error_msg.lower() for err_type in 
                                  ['invalid api key', 'insufficient credits', 'quota exceeded']):
                                return None
                            
                            raise requests.exceptions.RequestException(f"SerpAPI error: {error_msg}")
                        
                        # Update quota info if available
                        if "search_metadata" in data:
                            metadata = data["search_metadata"]
                            if "google_url" in metadata:
                                logger.debug(f"SerpAPI search URL: {metadata['google_url']}")
                        
                        # Log response để debugging
                        self._log_api_response(data, search_id)
                        
                        return data
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse SerpAPI JSON response: {e}")
                        logger.debug(f"Raw response: {response.text[:500]}")
                        last_error = e
                        
                elif response.status_code == 401:
                    logger.error("SerpAPI authentication failed - check API key")
                    return None
                    
                elif response.status_code == 429:
                    logger.warning("SerpAPI rate limit hit")
                    wait_time = (2 ** attempt) * 2  # Exponential backoff
                    time.sleep(wait_time)
                    last_error = requests.exceptions.RequestException("Rate limited")
                    
                elif response.status_code in [500, 502, 503, 504]:
                    logger.warning(f"SerpAPI server error: {response.status_code}")
                    wait_time = (2 ** attempt) * 1.5
                    time.sleep(wait_time)
                    last_error = requests.exceptions.RequestException(f"Server error {response.status_code}")
                    
                else:
                    logger.error(f"Unexpected SerpAPI status code: {response.status_code}")
                    logger.debug(f"Response: {response.text[:200]}")
                    last_error = requests.exceptions.RequestException(f"HTTP {response.status_code}")
                
            except requests.exceptions.Timeout:
                logger.warning(f"SerpAPI request timeout on attempt {attempt + 1}")
                last_error = requests.exceptions.Timeout("Request timeout")
                
            except requests.exceptions.ConnectionError:
                logger.warning(f"SerpAPI connection error on attempt {attempt + 1}")
                last_error = requests.exceptions.ConnectionError("Connection failed")
                
            except Exception as e:
                logger.error(f"Unexpected error in SerpAPI request: {e}", exc_info=True)
                last_error = e
            
            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"All {self.max_retries} attempts failed for {search_id}. Last error: {last_error}")
        return None
    
    def _process_search_response(self, 
                                response_data: Dict[str, Any], 
                                social_media_only: bool = True,
                                enable_deduplication: bool = True) -> List[SearchResult]:
        """Process raw SerpAPI response thành structured results"""
        search_results = []
        seen_urls = set() if enable_deduplication else None
        
        # Define result sources với priority order
        result_sources = [
            ("image_results", SearchResultSource.IMAGE_RESULTS),
            ("organic_results", SearchResultSource.ORGANIC_RESULTS), 
            ("inline_images", SearchResultSource.INLINE_IMAGES),
            ("image_content", SearchResultSource.IMAGE_CONTENT),
        ]
        
        for source_key, source_enum in result_sources:
            if source_key not in response_data:
                continue
            
            source_items = response_data[source_key]
            if not isinstance(source_items, list):
                continue
            
            logger.debug(f"Processing {len(source_items)} items from {source_key}")
            
            for position, item in enumerate(source_items):
                try:
                    processed_result = self._process_single_result(
                        item, source_enum, position, social_media_only
                    )
                    
                    if not processed_result:
                        continue
                    
                    # Deduplication check
                    if enable_deduplication:
                        if processed_result.link in seen_urls:
                            logger.debug(f"Skipping duplicate URL: {processed_result.link}")
                            continue
                        seen_urls.add(processed_result.link)
                    
                    search_results.append(processed_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to process result item: {e}")
                    continue
        
        logger.info(f"Processed {len(search_results)} valid results from SerpAPI response")
        return search_results
    
    def _process_single_result(self, 
                              item: Dict[str, Any], 
                              source: SearchResultSource,
                              position: int,
                              social_media_only: bool) -> Optional[SearchResult]:
        """Process single result item với comprehensive validation"""
        try:
            # Extract basic fields
            link = str(item.get("link", "")).strip()
            if not link or not link.startswith(('http://', 'https://')):
                return None
            
            title = str(item.get("title", "") or item.get("source", "") or "No Title").strip()
            displayed_link = str(item.get("displayed_link", "") or link).strip()
            snippet = str(item.get("snippet", "") or item.get("source", "") or "").strip()
            
            # Social media detection
            is_social_media = self._is_social_media_url(link, displayed_link)
            platform = self._detect_social_platform(link) if is_social_media else None
            
            # Apply social media filter
            if social_media_only and not is_social_media:
                return None
            
            # Extract additional metadata
            metadata = {}
            if "thumbnail" in item:
                metadata["thumbnail"] = item["thumbnail"]
            if "position" in item:
                metadata["serp_position"] = item["position"]
            if "date" in item:
                metadata["date"] = item["date"]
            
            return SearchResult(
                title=title,
                link=link,
                displayed_link=displayed_link,
                snippet=snippet,
                is_social_media=is_social_media,
                source=source.value,
                platform=platform,
                position=position,
                metadata=metadata
            )
            
        except Exception as e:
            logger.warning(f"Error processing single result: {e}")
            return None
    
    def _is_social_media_url(self, url: str, displayed_link: str = "") -> bool:
        """Comprehensive social media detection"""
        url_lower = url.lower()
        displayed_lower = displayed_link.lower()
        
        # Check against known domains
        for domain in self.social_media_domains:
            if domain in url_lower or domain in displayed_lower:
                return True
        
        # Additional pattern matching for edge cases
        social_patterns = [
            'facebook.com', 'fb.com', 'fb.me',
            'instagram.com', 'instagr.am',
            'twitter.com', 'x.com', 't.co',
            'linkedin.com', 'lnkd.in',
            'tiktok.com', 'vm.tiktok.com',
            'youtube.com', 'youtu.be',
            'pinterest.com', 'pin.it',
            'reddit.com', 'redd.it',
            'tumblr.com',
            'snapchat.com',
        ]
        
        for pattern in social_patterns:
            if pattern in url_lower or pattern in displayed_lower:
                return True
        
        return False
    
    def _detect_social_platform(self, url: str) -> Optional[str]:
        """Detect specific social media platform"""
        url_lower = url.lower()
        
        for platform, keywords in self.platform_keywords.items():
            for keyword in keywords:
                if keyword in url_lower:
                    return platform
        
        return "social_other"
    
    def _update_metrics(self, 
                       request_start_time: float, 
                       success: bool, 
                       results_count: int = 0,
                       error_type: Optional[str] = None) -> None:
        """Update comprehensive metrics tracking"""
        request_duration = time.time() - request_start_time
        
        with self._metrics_lock:
            self.metrics.total_requests += 1
            self.metrics.last_request_time = time.time()
            
            if success:
                self.metrics.successful_requests += 1
                self.metrics.total_results_found += results_count
                
                # Update rolling average response time
                if self.metrics.successful_requests == 1:
                    self.metrics.average_response_time = request_duration
                else:
                    # Exponential moving average
                    alpha = 0.1
                    self.metrics.average_response_time = (
                        alpha * request_duration + 
                        (1 - alpha) * self.metrics.average_response_time
                    )
            else:
                self.metrics.failed_requests += 1
                if error_type:
                    self.metrics.errors_by_type[error_type] += 1
    
    def _get_cached_result(self, search_id: str) -> Optional[List[SearchResult]]:
        """Get cached search result if available và still valid"""
        if not self.enable_caching or not self._cache_lock:
            return None
        
        with self._cache_lock:
            if search_id in self._cache:
                cached_data, timestamp = self._cache[search_id]
                
                # Check if cache is still valid
                if time.time() - timestamp < self._cache_ttl:
                    logger.debug(f"Cache hit for search {search_id}")
                    return cached_data
                else:
                    # Remove expired cache entry
                    del self._cache[search_id]
                    logger.debug(f"Cache expired for search {search_id}")
        
        return None
    
    def _cache_result(self, search_id: str, results: List[SearchResult]) -> None:
        """Cache search results"""
        if not self.enable_caching or not self._cache_lock:
            return
        
        with self._cache_lock:
            self._cache[search_id] = (results, time.time())
            
            # Clean up old cache entries (keep max 100 entries)
            if len(self._cache) > 100:
                # Remove oldest entries
                sorted_entries = sorted(self._cache.items(), key=lambda x: x[1][1])
                entries_to_remove = sorted_entries[:20]  # Remove oldest 20
                
                for entry_id, _ in entries_to_remove:
                    del self._cache[entry_id]
    
    def _log_api_response(self, data: Dict[str, Any], search_id: str) -> None:
        """Log API response cho debugging (paranoid logging)"""
        try:
            log_dir = "logs/api_responses"
            os.makedirs(log_dir, exist_ok=True)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{log_dir}/serpapi_response_{timestamp}_{search_id}.json"
            
            # Sanitize data để remove sensitive info
            sanitized_data = self._sanitize_response_for_logging(data)
            
            with open(log_filename, "w", encoding="utf-8") as f:
                json.dump(sanitized_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"API response logged to: {log_filename}")
            
        except Exception as e:
            logger.warning(f"Failed to log API response: {e}")
    
    def _sanitize_response_for_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive data from response before logging"""
        sanitized = data.copy()
        
        # Remove API key if present
        if "search_metadata" in sanitized:
            metadata = sanitized["search_metadata"].copy()
            if "google_url" in metadata:
                # Remove API key from URL
                url = metadata["google_url"]
                metadata["google_url"] = url.split("&api_key=")[0] if "&api_key=" in url else url
            sanitized["search_metadata"] = metadata
        
        return sanitized
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics và statistics"""
        with self._metrics_lock:
            success_rate = (
                self.metrics.successful_requests / self.metrics.total_requests * 100
                if self.metrics.total_requests > 0 else 0
            )
            
            avg_results_per_search = (
                self.metrics.total_results_found / self.metrics.successful_requests
                if self.metrics.successful_requests > 0 else 0
            )
            
            return {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate_percent": round(success_rate, 2),
                "total_results_found": self.metrics.total_results_found,
                "average_results_per_search": round(avg_results_per_search, 2),
                "average_response_time_seconds": round(self.metrics.average_response_time, 3),
                "rate_limit_hits": self.metrics.rate_limit_hits,
                "last_request_time": self.metrics.last_request_time,
                "errors_by_type": dict(self.metrics.errors_by_type),
                "cache_enabled": self.enable_caching,
                "cache_size": len(self._cache) if self._cache else 0,
                "uploader_type": type(self.image_uploader).__name__,
            }
    
    def clear_cache(self) -> None:
        """Clear all cached results"""
        if self.enable_caching and self._cache_lock:
            with self._cache_lock:
                self._cache.clear()
                logger.info("Search cache cleared")
    
    def test_connection(self) -> Dict[str, Any]:
        """Test SerpAPI connection và quota"""
        try:
            # Make minimal test request
            test_params = {
                "api_key": self.api_key,
                "engine": "google",
                "q": "test",
                "num": 1
            }
            
            response = requests.get(
                self.api_url,
                params=test_params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    return {
                        "success": False,
                        "error": data["error"],
                        "quota_remaining": None
                    }
                
                # Extract quota info if available
                quota_info = None
                if "search_metadata" in data:
                    # SerpAPI quota info varies by plan
                    quota_info = "Available"
                
                return {
                    "success": True,
                    "quota_remaining": quota_info,
                    "response_time_ms": int(response.elapsed.total_seconds() * 1000)
                }
            
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "quota_remaining": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "quota_remaining": None
            }

def validate_api_keys():
    """Enhanced API key validation với ImgBB support"""
    from BE.config.settings import SERP_API_KEY, IMGUR_CLIENT_ID, IMGBB_API_KEY
    
    validation_result = {
        "has_serp_api": bool(SERP_API_KEY and SERP_API_KEY.strip()),
        "has_imgur": bool(IMGUR_CLIENT_ID and IMGUR_CLIENT_ID.strip()),
        "has_imgbb": bool(IMGBB_API_KEY and IMGBB_API_KEY.strip()),
        "ready": False,
        "missing_keys": [],
        "preferred_uploader": None
    }
    
    # Check required keys
    if not validation_result["has_serp_api"]:
        validation_result["missing_keys"].append("SERP_API_KEY")
    
    # Check image uploaders (need at least one)
    has_uploader = validation_result["has_imgbb"] or validation_result["has_imgur"]
    if not has_uploader:
        validation_result["missing_keys"].extend(["IMGBB_API_KEY or IMGUR_CLIENT_ID"])
    
    # Determine preferred uploader
    if validation_result["has_imgbb"]:
        validation_result["preferred_uploader"] = "ImgBB"
    elif validation_result["has_imgur"]:
        validation_result["preferred_uploader"] = "Imgur"
    
    # Overall readiness
    validation_result["ready"] = (
        validation_result["has_serp_api"] and has_uploader
    )
    
    return validation_result

def create_search_client(force_uploader_type: Optional[str] = None):
    """
    Factory function để tạo fully configured search client
    
    Args:
        force_uploader_type: Force specific uploader ('imgbb', 'imgur', None for auto)
        
    Returns:
        Tuple of (image_uploader, serp_client)
        
    Raises:
        ValueError: If missing required API keys
    """
    from BE.config.settings import SERP_API_KEY
    
    validation = validate_api_keys()
    if not validation["ready"]:
        raise ValueError(f"Missing API keys: {', '.join(validation['missing_keys'])}")
    
    # Create image uploader
    image_uploader = create_image_uploader(force_uploader_type)
    
    # Create SerpAPI client
    serp_client = SerpApiClient(
        api_key=SERP_API_KEY,
        image_uploader=image_uploader,
        max_retries=3,
        rate_limit_per_minute=50,  # Conservative
        enable_caching=True
    )
    
    logger.info(f"Search client created with {validation['preferred_uploader']} uploader")
    return image_uploader, serp_client