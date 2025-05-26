import requests
import json
import os
import datetime
import logging
from core.image_processing import optimize_image
from config.constants import API_TIMEOUT_SEC, SOCIAL_MEDIA_DOMAINS

logger = logging.getLogger('ImageSearchApp')

class ImgurClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.api_url = "https://api.imgur.com/3/image"
        
    def upload_image(self, image_path):
        """Upload image to Imgur and return public URL"""
        try:
            binary_data = optimize_image(image_path)
            
            headers = {'Authorization': f'Client-ID {self.client_id}'}
            response = requests.post(
                self.api_url, 
                headers=headers,
                files={'image': binary_data},
                timeout=API_TIMEOUT_SEC
            )
            
            if response.status_code != 200:
                logger.error(f"Imgur upload error: {response.status_code}, {response.text}")
                return None
                
            data = response.json()
            self._log_response(data, "imgur")
            
            if data['success']:
                image_url = data['data']['link']
                logger.info(f"Imgur upload successful: {image_url}")
                return image_url
            else:
                error_msg = data.get('data', {}).get('error', 'Unknown error')
                logger.error(f"Imgur upload error: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading to Imgur: {str(e)}", exc_info=True)
            return None
    
    def upload_image_from_bytes(self, image_bytes, filename="upload.jpg"):
        try:
            headers = {'Authorization': f'Client-ID {self.client_id}'}
            response = requests.post(
                self.api_url, 
                headers=headers,
                files={'image': (filename, image_bytes, 'image/jpeg')},
                timeout=API_TIMEOUT_SEC
            )
            
            if response.status_code != 200:
                logger.error(f"Imgur upload error: {response.status_code}, {response.text}")
                return None
                
            data = response.json()
            self._log_response(data, "imgur")
            
            if data['success']:
                image_url = data['data']['link']
                logger.info(f"Imgur upload successful: {image_url}")
                return image_url
            else:
                error_msg = data.get('data', {}).get('error', 'Unknown error')
                logger.error(f"Imgur upload error: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Error uploading bytes to Imgur: {str(e)}", exc_info=True)
            return None
    
    def _log_response(self, data, source):
        """Log API response"""
        try:
            log_dir = "logs/api_responses"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f"{log_dir}/{source}_response_{timestamp}.json", "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {source.capitalize()} API response to logs")
        except Exception as e:
            logger.warning(f"Could not save API response log: {e}")


class SerpApiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.api_url = "https://serpapi.com/search"
        
    def search_by_image(self, image_url, social_media_only=True):
        all_results = []
        error_msg = None
        
        try:
            params = {
                "api_key": self.api_key,
                "engine": "google_reverse_image",
                "image_url": image_url,
                "num": 100,
                "ijn": 0
            }
            
            logger.info(f"Sending request to SerpAPI for up to 100 results")
            response = requests.get(
                self.api_url, 
                params=params, 
                timeout=API_TIMEOUT_SEC * 2
            )
            
            if response.status_code != 200:
                error_msg = f"SerpAPI returned error code: {response.status_code}"
                logger.error(error_msg)
                return [], error_msg
            
            data = response.json()
            self._log_response(data, "bulk_results")
            
            all_results = self._extract_results(data, social_media_only)
            
            logger.info(f"Total found {len(all_results)} results")
            if social_media_only:
                logger.info(f"Including {len(all_results)} social media results")
            
            return all_results, error_msg
            
        except requests.exceptions.Timeout:
            logger.error("Timeout when calling SerpAPI", exc_info=True)
            return [], "Request timeout. Please try again later."
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error to SerpAPI: {str(e)}", exc_info=True)
            return [], f"Connection error: {e}"
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}", exc_info=True)
            return [], "Cannot parse API response"
    
    def search_by_image_async(self, image_url, social_media_only=True, progress_callback=None):
        if progress_callback:
            progress_callback("Starting image search...")
        
        try:
            if progress_callback:
                progress_callback("Sending request to SerpAPI...")
            
            results, error = self.search_by_image(image_url, social_media_only)
            
            if progress_callback:
                if error:
                    progress_callback(f"Error: {error}")
                else:
                    progress_callback(f"Found {len(results)} results")
            
            return results, error
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            if progress_callback:
                progress_callback(error_msg)
            return [], error_msg
    
    def _extract_results(self, data, social_media_only=True):
        """Extract results from API response with social media filtering"""
        search_results = []
        
        result_sources = [
            ("image_results", "Images"),
            ("organic_results", "Organic Search"),
            ("inline_images", "Inline Images"),
            ("image_content", "Related Images")
        ]
        
        for key, source_name in result_sources:
            if key in data and isinstance(data[key], list):
                source_items = data[key]
                logger.info(f"Found {len(source_items)} results from source: {source_name}")
                
                for item in source_items:
                    link = str(item.get("link", ""))
                    if not link:
                        continue
                        
                    displayed_link = str(item.get("displayed_link", ""))
                    if not displayed_link:
                        displayed_link = link
                    
                    title = str(item.get("title", "") or item.get("source", "") or "No title")
                    snippet = str(item.get("snippet", "") or item.get("source", "") or "No description")
                    
                    if social_media_only and not self._is_social_media(link, displayed_link):
                        continue
                    
                    clean_item = {
                        "title": title,
                        "link": link,
                        "displayed_link": displayed_link,
                        "snippet": snippet,
                        "is_social_media": self._is_social_media(link, displayed_link),
                        "source": source_name
                    }
                    
                    if not any(r["link"] == link for r in search_results):
                        search_results.append(clean_item)
        
        return search_results
    
    def _is_social_media(self, url, displayed_link):
        """Check if URL or displayed_link is from social media"""
        url_lower = url.lower()
        displayed_lower = displayed_link.lower()
        
        for domain in SOCIAL_MEDIA_DOMAINS:
            if domain in url_lower or domain in displayed_lower:
                return True
                
        return False
    
    def _log_response(self, data, suffix=""):
        """Log API response"""
        try:
            log_dir = "logs/api_responses"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            api_response_path = f"{log_dir}/serpapi_response_{timestamp}_{suffix}.json"
            
            with open(api_response_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved API response to: {api_response_path}")
        except Exception as e:
            logger.warning(f"Could not save API response log: {e}")

def validate_api_keys():
    """Validate that required API keys are available"""
    from config.settings import SERP_API_KEY, IMGUR_CLIENT_ID
    
    validation_result = {
        "has_serp_api": bool(SERP_API_KEY and SERP_API_KEY.strip()),
        "has_imgur": bool(IMGUR_CLIENT_ID and IMGUR_CLIENT_ID.strip()),
        "ready": False,
        "missing_keys": []
    }
    
    if not validation_result["has_serp_api"]:
        validation_result["missing_keys"].append("SERP_API_KEY")
    
    if not validation_result["has_imgur"]:
        validation_result["missing_keys"].append("IMGUR_CLIENT_ID")
    
    validation_result["ready"] = len(validation_result["missing_keys"]) == 0
    
    return validation_result


def create_search_client():
    """Factory function to create configured search clients"""
    from config.settings import SERP_API_KEY, IMGUR_CLIENT_ID
    
    validation = validate_api_keys()
    if not validation["ready"]:
        raise ValueError(f"Missing API keys: {', '.join(validation['missing_keys'])}")
    
    imgur_client = ImgurClient(IMGUR_CLIENT_ID)
    serp_client = SerpApiClient(SERP_API_KEY)
    
    return imgur_client, serp_client