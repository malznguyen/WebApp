import base64
import time
import logging
from typing import Optional, Dict, Any, Union
from datetime import datetime
import hashlib

logger = logging.getLogger('ImageSearchApp')

class ImgBBClient:
    """
    ImgBB API Client
    
    API Limitations (as of 2025):
    - Max file size: 32MB
    - Rate limit: ~150 requests/hour (unofficial)
    - Supported formats: JPG, PNG, GIF, BMP, WEBP, SVG
    - Free tier: Unlimited uploads (theoretically)
    """
    
    def __init__(self, api_key: str):
        if not api_key or not isinstance(api_key, str) or len(api_key.strip()) < 10:
            raise ValueError("ImgBB API key is invalid or missing. Expected non-empty string with length > 10")
        
        self.api_key = api_key.strip()
        self.api_url = "https://api.imgbb.com/1/upload"
        self.max_file_size = 32 * 1024 * 1024  # 32MB - ImgBB limit
        self.timeout = 30  # Conservative timeout
        self.max_retries = 3
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
        
        # Rate limiting tracking (paranoid level)
        self._last_request_time = 0
        self._request_count = 0
        self._rate_limit_window_start = time.time()
        self._max_requests_per_hour = 100  # Conservative estimate
        
        logger.info("ImgBB client initialized successfully")
    
    def upload_image(self, image_path: str) -> Optional[str]:
        """
        Upload image from file path to ImgBB
        
        Args:
            image_path: Path to image file
            
        Returns:
            Public URL string if successful, None if failed
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file is too large or invalid format
        """
        if not image_path or not isinstance(image_path, str):
            raise ValueError("Image path must be a non-empty string")
        
        import os
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if not os.path.isfile(image_path):
            raise ValueError(f"Path is not a file: {image_path}")
        
        try:
            with open(image_path, 'rb') as file:
                image_data = file.read()
            
            filename = os.path.basename(image_path)
            return self.upload_image_from_bytes(image_data, filename)
            
        except PermissionError as e:
            logger.error(f"Permission denied reading image file: {image_path}")
            raise PermissionError(f"Cannot read image file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading image file {image_path}: {e}", exc_info=True)
            return None
    
    def upload_image_from_bytes(self, image_data: bytes, filename: str = "upload.jpg") -> Optional[str]:
        """
        Upload image from bytes to ImgBB with paranoid error handling
        
        Args:
            image_data: Raw image bytes
            filename: Original filename for reference
            
        Returns:
            Public URL if successful, None if failed
        """
        try:
            # Validation với paranoid level
            self._validate_image_data(image_data, filename)
            self._check_rate_limits()
            
            # Prepare request với optimal settings
            upload_data = self._prepare_upload_data(image_data, filename)
            
            # Execute upload với retry logic
            for attempt in range(self.max_retries):
                try:
                    response = self._make_upload_request(upload_data, attempt)
                    
                    if response and response.get('success'):
                        url = response['data']['url']
                        self._log_successful_upload(url, filename, len(image_data))
                        return url
                    else:
                        error_msg = response.get('error', {}).get('message', 'Unknown error') if response else 'No response'
                        logger.warning(f"ImgBB upload attempt {attempt + 1} failed: {error_msg}")
                        
                        if attempt < self.max_retries - 1:
                            wait_time = (2 ** attempt) * 1.0  # Exponential backoff
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                    if attempt == self.max_retries - 1:
                        raise
            
            logger.error(f"All {self.max_retries} upload attempts failed for {filename}")
            return None
            
        except Exception as e:
            logger.error(f"ImgBB upload failed for {filename}: {e}", exc_info=True)
            return None
    
    def _validate_image_data(self, image_data: bytes, filename: str) -> None:
        """Paranoid validation của image data"""
        if not image_data:
            raise ValueError("Image data is empty")
        
        if len(image_data) > self.max_file_size:
            size_mb = len(image_data) / (1024 * 1024)
            raise ValueError(f"Image too large: {size_mb:.2f}MB > 32MB limit")
        
        if len(image_data) < 100:  # Suspiciously small
            raise ValueError("Image data too small, likely corrupted")
        
        # Check file extension (basic validation)
        import os
        file_ext = os.path.splitext(filename.lower())[1]
        if file_ext and file_ext not in self.supported_formats:
            logger.warning(f"File extension {file_ext} not in supported formats, but continuing...")
        
        # Basic image magic number validation
        if not self._is_valid_image_magic_number(image_data):
            logger.warning("Image magic number validation failed, but continuing...")
    
    def _is_valid_image_magic_number(self, data: bytes) -> bool:
        """Check image magic numbers để detect format"""
        if len(data) < 12:
            return False
        
        magic_numbers = {
            b'\xFF\xD8\xFF': 'JPEG',
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'GIF87a': 'GIF87a',
            b'GIF89a': 'GIF89a',
            b'BM': 'BMP',
            b'RIFF': 'WEBP',  # Needs more validation
        }
        
        for magic, format_name in magic_numbers.items():
            if data.startswith(magic):
                return True
        
        return False
    
    def _check_rate_limits(self) -> None:
        """Paranoid rate limiting check"""
        current_time = time.time()
        
        # Reset counter if new hour
        if current_time - self._rate_limit_window_start > 3600:
            self._request_count = 0
            self._rate_limit_window_start = current_time
        
        # Check if approaching rate limit
        if self._request_count >= self._max_requests_per_hour:
            wait_time = 3600 - (current_time - self._rate_limit_window_start)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.0f} seconds")
                time.sleep(wait_time)
                self._request_count = 0
                self._rate_limit_window_start = time.time()
        
        # Minimum interval between requests
        time_since_last = current_time - self._last_request_time
        if time_since_last < 1.0:  # Max 1 request per second
            sleep_time = 1.0 - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _prepare_upload_data(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Prepare upload payload với optimal settings"""
        # Generate unique name để avoid conflicts
        timestamp = int(time.time())
        file_hash = hashlib.md5(image_data).hexdigest()[:8]
        base_name = filename.split('.')[0] if '.' in filename else 'upload'
        unique_name = f"search_{timestamp}_{file_hash}_{base_name}"
        
        # Convert to base64 (ImgBB requirement)
        try:
            image_b64 = base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to encode image as base64: {e}")
        
        return {
            'key': self.api_key,
            'image': image_b64,
            'name': unique_name,
            'expiration': 3600,  # Auto-delete after 1 hour (perfect for search)
        }
    
    def _make_upload_request(self, upload_data: Dict[str, Any], attempt: int) -> Optional[Dict[str, Any]]:
        """Execute upload request với comprehensive error handling"""
        headers = {
            'User-Agent': 'Enhanced-Toolkit-v2.0-ImgBB-Client',
            'Accept': 'application/json',
        }
        
        try:
            logger.debug(f"Making ImgBB upload request (attempt {attempt + 1})")
            
            response = requests.post(
                self.api_url,
                data=upload_data,
                headers=headers,
                timeout=self.timeout
            )
            
            # Log response details for debugging
            logger.debug(f"ImgBB response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as e:
                    logger.error(f"Failed to parse ImgBB JSON response: {e}")
                    logger.debug(f"Raw response: {response.text[:500]}")
                    return None
            
            elif response.status_code == 400:
                logger.error(f"ImgBB bad request (400): {response.text}")
                return None  # Don't retry on 400
            
            elif response.status_code == 413:
                logger.error("ImgBB file too large (413)")
                return None  # Don't retry on 413
            
            elif response.status_code in [429, 503]:
                logger.warning(f"ImgBB rate limited or service unavailable ({response.status_code})")
                raise requests.exceptions.RequestException(f"Rate limited: {response.status_code}")
            
            else:
                logger.warning(f"ImgBB unexpected status {response.status_code}: {response.text[:200]}")
                raise requests.exceptions.RequestException(f"HTTP {response.status_code}")
        
        except requests.exceptions.Timeout:
            logger.warning(f"ImgBB request timeout on attempt {attempt + 1}")
            raise
        except requests.exceptions.ConnectionError:
            logger.warning(f"ImgBB connection error on attempt {attempt + 1}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ImgBB request: {e}", exc_info=True)
            raise
    
    def _log_successful_upload(self, url: str, filename: str, file_size: int) -> None:
        """Log successful upload với useful metrics"""
        size_mb = file_size / (1024 * 1024)
        logger.info(f"ImgBB upload SUCCESS: {filename} ({size_mb:.2f}MB) -> {url}")
        
        # Log API usage statistics
        logger.debug(f"ImgBB API usage: {self._request_count}/{self._max_requests_per_hour} requests this hour")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        current_time = time.time()
        window_remaining = 3600 - (current_time - self._rate_limit_window_start)
        
        return {
            'requests_this_hour': self._request_count,
            'max_requests_per_hour': self._max_requests_per_hour,
            'window_remaining_seconds': max(0, window_remaining),
            'last_request_time': self._last_request_time,
        }