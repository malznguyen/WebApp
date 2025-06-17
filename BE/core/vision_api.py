import os
import base64
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
from PIL import Image
import io
from datetime import datetime
import json

logger = logging.getLogger('ImageSearchApp')

# === AI Image Detection Prompts ===
AI_DETECTION_SYSTEM_PROMPT = (
    "You are an expert digital forensics analyst specializing in AI-generated "
    "image detection. Your task is to analyze images for signs of artificial "
    "generation using current AI models (DALL-E, Midjourney, Stable Diffusion, etc.)."
)

AI_DETECTION_USER_PROMPT = """
ANALYSIS FRAMEWORK:
1. TECHNICAL ARTIFACTS
- Pixel-level inconsistencies
- Compression artifacts unusual for camera photos
- Lighting inconsistencies across objects
- Shadow direction conflicts
- Texture anomalies

2. ANATOMICAL INDICATORS (for human subjects)
- Facial feature asymmetries
- Hand/finger anatomy errors
- Eye reflection inconsistencies
- Skin texture unnaturalness
- Hair strand impossibilities

3. CONTEXTUAL CLUES
- Background-foreground integration issues
- Impossible physics or perspectives
- Text rendering errors or nonsensical text
- Brand logo distortions
- Architectural impossibilities

4. STYLE MARKERS
- Overly perfect or "glossy" appearance
- Telltale AI art style characteristics
- Color grading typical of AI models
- Composition patterns common in generated images

OUTPUT FORMAT (JSON):
{
  "ai_generated_probability": "percentage (0-100)",
  "confidence_level": "low/medium/high",
  "analysis_summary": "concise explanation in Vietnamese",
  "detected_indicators": [
    {
      "category": "technical/anatomical/contextual/style",
      "indicator": "specific finding",
      "severity": "minor/moderate/major",
      "explanation": "why this suggests AI generation"
    }
  ],
  "likely_generation_method": "DALL-E/Midjourney/Stable Diffusion/Real Photo/Unknown",
  "recommendations": "advice for user",
  "limitations": "analysis limitations and caveats"
}

IMPORTANT CONSIDERATIONS:
- Modern AI can be extremely sophisticated - express uncertainty when appropriate
- Real photos can have unusual characteristics too
- Heavy photo editing can mimic AI artifacts
- Provide educational value, not definitive judgments
- Be especially careful with photos of people (privacy/reputation concerns)

Nếu ảnh chứa text tiếng Việt:
- Kiểm tra dấu thanh chính xác
- Font rendering consistency
- Spelling và grammar trong signs/text
- Cultural context appropriateness

Nếu ảnh chứa Vietnamese subjects:
- Facial feature authenticity
- Cultural dress/setting accuracy
- Background context matching Vietnamese locations
"""

class OpenAIVisionClient:
    
    def __init__(self, api_key: str):
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=api_key)
        self.max_image_size = 20 * 1024 * 1024  # 20MB OpenAI limit
        self.optimal_size = 5 * 1024 * 1024     # 5MB optimal for performance
        self.supported_formats = {'JPEG', 'PNG', 'GIF', 'WEBP', 'BMP'}
        self.max_dimension = 2048  # Optimal dimension for Vision API
        
        # Rate limiting và retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 60  # seconds
        
        # Cost tracking
        self.total_tokens_used = 0
        self.total_requests = 0
        
        logger.info("OpenAI Vision client initialized successfully")
    
    def describe_image(self, 
                      image_data: bytes, 
                      filename: str = "image",
                      language: str = "vietnamese",
                      detail_level: str = "detailed",
                      custom_prompt: Optional[str] = None) -> Dict[str, Any]:

        start_time = time.time()
        
        try:
            logger.info(f"Starting vision analysis for '{filename}' (lang: {language}, detail: {detail_level})")
            
            # Step 1: Validate input
            validation_result = self._validate_image(image_data, filename)
            if not validation_result['valid']:
                return self._create_error_response(validation_result['error'], filename)
            
            # Step 2: Optimize image for API
            optimized_data, optimization_info = self._optimize_image_for_api(image_data)
            logger.debug(f"Image optimization for '{filename}': {optimization_info}")
            
            # Step 3: Prepare base64 encoding
            base64_image = base64.b64encode(optimized_data).decode('utf-8')
            
            # Step 4: Create prompt
            if custom_prompt:
                prompt = custom_prompt
                logger.info(f"Using custom prompt for '{filename}'")
            else:
                prompt = self._create_vision_prompt(language, detail_level)
            
            # Step 5: Call OpenAI API with retry logic
            api_response = self._call_vision_api_with_retry(
                base64_image=base64_image,
                prompt=prompt,
                detail_level=detail_level,
                filename=filename
            )
            
            if not api_response['success']:
                return self._create_error_response(api_response['error'], filename)
            
            # Step 6: Process response
            description = api_response['description']
            api_usage = api_response['usage']
            
            # Step 7: Extract metadata
            image_metadata = self._extract_comprehensive_metadata(image_data)
            
            processing_time = time.time() - start_time
            
            # Step 8: Create comprehensive result
            result = {
                'success': True,
                'description': description,
                'language': language,
                'detail_level': detail_level,
                'processing_time_seconds': round(processing_time, 2),
                'image_metadata': image_metadata,
                'text_metrics': {
                    'word_count': len(description.split()),
                    'char_count': len(description),
                    'sentence_count': len([s for s in description.split('.') if s.strip()]),
                    'paragraph_count': len([p for p in description.split('\n\n') if p.strip()])
                },
                'optimization_info': optimization_info,
                'api_usage': api_usage,
                'timestamp': datetime.now().isoformat(),
                'filename': filename
            }
            
            # Update tracking
            self.total_requests += 1
            self.total_tokens_used += api_usage.get('total_tokens', 0)
            
            logger.info(f"Vision analysis completed for '{filename}': "
                       f"{result['text_metrics']['word_count']} words, "
                       f"{processing_time:.2f}s, "
                       f"${api_usage.get('cost_estimate', 0):.4f}")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error in vision analysis for '{filename}': {str(e)}"
            logger.error(error_msg, exc_info=True)

            return self._create_error_response(error_msg, filename, processing_time)

    def detect_ai_generated_image(self, image_data: bytes, filename: str = "image") -> Dict[str, Any]:
        """Analyze an image for signs of AI generation."""
        start_time = time.time()
        try:
            validation_result = self._validate_image(image_data, filename)
            if not validation_result['valid']:
                return self._create_error_response(validation_result['error'], filename)

            optimized_data, optimization_info = self._optimize_image_for_api(image_data)
            base64_image = base64.b64encode(optimized_data).decode('utf-8')

            api_response = self._call_vision_api_with_retry(
                base64_image=base64_image,
                prompt=AI_DETECTION_USER_PROMPT,
                detail_level="detailed",
                filename=filename,
                system_prompt=AI_DETECTION_SYSTEM_PROMPT
            )

            if not api_response['success']:
                return self._create_error_response(api_response['error'], filename)

            raw_text = api_response['description']
            detection_data = self._extract_json_from_text(raw_text)

            image_metadata = self._extract_comprehensive_metadata(image_data)
            processing_time = time.time() - start_time

            result = {
                'success': True,
                'detection': detection_data,
                'raw_response': raw_text,
                'processing_time_seconds': round(processing_time, 2),
                'optimization_info': optimization_info,
                'image_metadata': image_metadata,
                'api_usage': api_response['usage'],
                'timestamp': datetime.now().isoformat(),
                'filename': filename
            }

            self.total_requests += 1
            self.total_tokens_used += api_response['usage'].get('total_tokens', 0)

            return result
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Unexpected error in AI detection for '{filename}': {e}", exc_info=True)
            return self._create_error_response(str(e), filename, processing_time)
    
    def _validate_image(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """Comprehensive image validation với detailed error messages"""
        try:
            # Check basic requirements
            if not image_data:
                return {'valid': False, 'error': 'Image data is empty or None'}
            
            if len(image_data) < 100:  # Suspiciously small
                return {'valid': False, 'error': 'Image data too small (likely corrupted)'}
            
            if len(image_data) > self.max_image_size:
                size_mb = len(image_data) / (1024 * 1024)
                return {'valid': False, 'error': f'Image too large: {size_mb:.1f}MB (max: 20MB)'}
            
            # Verify image integrity
            try:
                img = Image.open(io.BytesIO(image_data))
                img.verify()  # This will raise exception if corrupted
                
                # Re-open for metadata (verify() closes the image)
                img = Image.open(io.BytesIO(image_data))
                
                # Check format support
                if img.format not in self.supported_formats:
                    return {
                        'valid': False, 
                        'error': f'Unsupported format: {img.format}. Supported: {", ".join(self.supported_formats)}'
                    }
                
                # Check dimensions (minimum viable size)
                if img.size[0] < 10 or img.size[1] < 10:
                    return {'valid': False, 'error': f'Image too small: {img.size[0]}x{img.size[1]} (min: 10x10)'}
                
                # Check for extreme aspect ratios (might cause API issues)
                aspect_ratio = max(img.size) / min(img.size)
                if aspect_ratio > 20:  # Extremely wide/tall images
                    logger.warning(f"Extreme aspect ratio detected for '{filename}': {aspect_ratio:.1f}")
                
                return {
                    'valid': True, 
                    'format': img.format, 
                    'size': img.size,
                    'mode': img.mode,
                    'aspect_ratio': aspect_ratio
                }
                
            except Exception as img_error:
                return {'valid': False, 'error': f'Invalid or corrupted image: {str(img_error)}'}
                
        except Exception as e:
            return {'valid': False, 'error': f'Validation system error: {str(e)}'}
    
    def _optimize_image_for_api(self, image_data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Optimize image để maximize API performance và minimize costs
        Returns: (optimized_bytes, optimization_info)
        """
        original_size = len(image_data)
        optimization_info = {
            'original_size_mb': round(original_size / (1024 * 1024), 2),
            'optimized': False,
            'resized': False,
            'format_changed': False,
            'quality_adjusted': False
        }
        
        try:
            # If already optimal size, return as-is
            if original_size <= self.optimal_size:
                logger.debug("Image already optimal size, no optimization needed")
                optimization_info['optimized_size_mb'] = optimization_info['original_size_mb']
                return image_data, optimization_info
            
            img = Image.open(io.BytesIO(image_data))
            original_format = img.format
            original_dimensions = img.size
            
            # Step 1: Resize if dimensions too large
            if max(img.size) > self.max_dimension:
                ratio = self.max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                optimization_info['resized'] = True
                optimization_info['new_dimensions'] = new_size
                logger.debug(f"Resized image from {original_dimensions} to {new_size}")
            
            # Step 2: Convert to RGB if necessary (for JPEG optimization)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparency
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
                optimization_info['format_changed'] = True
            
            # Step 3: Save với optimized settings
            buffer = io.BytesIO()
            
            # Determine best format và quality
            if original_format in ('PNG', 'GIF') and not optimization_info['format_changed']:
                # Keep PNG for images that might need transparency
                img.save(buffer, format='PNG', optimize=True)
            else:
                # Use JPEG với progressive quality scaling
                quality = 85
                while quality >= 60:  # Don't go below 60% quality
                    buffer.seek(0)
                    buffer.truncate()
                    img.save(buffer, format='JPEG', quality=quality, optimize=True)
                    
                    if buffer.tell() <= self.optimal_size or quality <= 60:
                        break
                    quality -= 5
                
                optimization_info['quality_adjusted'] = True
                optimization_info['final_quality'] = quality
            
            optimized_data = buffer.getvalue()
            optimized_size = len(optimized_data)
            
            optimization_info.update({
                'optimized': True,
                'optimized_size_mb': round(optimized_size / (1024 * 1024), 2),
                'compression_ratio': round(original_size / optimized_size, 2),
                'size_reduction_percent': round((1 - optimized_size / original_size) * 100, 1)
            })
            
            logger.info(f"Image optimization completed: "
                       f"{optimization_info['original_size_mb']}MB → {optimization_info['optimized_size_mb']}MB "
                       f"({optimization_info['size_reduction_percent']}% reduction)")
            
            return optimized_data, optimization_info
            
        except Exception as e:
            logger.warning(f"Image optimization failed, using original: {e}")
            optimization_info['optimization_error'] = str(e)
            optimization_info['optimized_size_mb'] = optimization_info['original_size_mb']
            return image_data, optimization_info
    
    def _create_vision_prompt(self, language: str, detail_level: str) -> str:
        """Create optimized prompts cho different scenarios"""
        
        prompts = {
            'vietnamese': {
                'brief': (
                    "Hãy mô tả ngắn gọn hình ảnh này bằng tiếng Việt trong 1-2 câu. "
                    "Tập trung vào những yếu tố quan trọng nhất."
                ),
                'detailed': (
                    "Hãy mô tả chi tiết hình ảnh này bằng tiếng Việt. Bao gồm:\n"
                    "- Các đối tượng chính và con người trong ảnh\n"
                    "- Màu sắc, ánh sáng và không gian\n"
                    "- Hoạt động hoặc tình huống đang diễn ra\n"
                    "- Cảm xúc và không khí tổng thể của ảnh\n"
                    "Viết mô tả tự nhiên và dễ hiểu."
                ),
                'extensive': (
                    "Hãy phân tích toàn diện hình ảnh này bằng tiếng Việt với mức độ chi tiết cao:\n"
                    "1. MÔ TẢ TỔNG QUAN: Khung cảnh chính và bối cảnh\n"
                    "2. ĐỐI TƯỢNG: Mô tả từng người, vật thể, động vật (nếu có)\n"
                    "3. MÀU SẮC VÀ ÁNH SÁNG: Bảng màu, nguồn sáng, bóng râm\n"
                    "4. COMPOSITION: Cách bố trí, góc chụp, perspective\n"
                    "5. HOẠT ĐỘNG: Hành động, cử chỉ, biểu cảm\n"
                    "6. KHÔNG KHÍ: Cảm xúc, mood, thông điệp\n"
                    "7. CHI TIẾT KỸ THUẬT: Phong cách chụp, kỹ thuật (nếu nhận ra được)\n"
                    "Viết mô tả rõ ràng và có cấu trúc."
                )
            },
            'english': {
                'brief': (
                    "Provide a brief description of this image in 1-2 sentences in English. "
                    "Focus on the most important elements."
                ),
                'detailed': (
                    "Describe this image in detail in English. Include:\n"
                    "- Main objects and people in the image\n"
                    "- Colors, lighting, and spatial arrangement\n"
                    "- Activities or situations taking place\n"
                    "- Overall mood and atmosphere\n"
                    "Write in natural, clear language."
                ),
                'extensive': (
                    "Provide a comprehensive analysis of this image in English with high detail:\n"
                    "1. OVERVIEW: Main scene and context\n"
                    "2. SUBJECTS: Describe each person, object, animal (if present)\n"
                    "3. COLORS & LIGHTING: Color palette, light sources, shadows\n"
                    "4. COMPOSITION: Layout, camera angle, perspective\n"
                    "5. ACTIONS: Activities, gestures, expressions\n"
                    "6. ATMOSPHERE: Emotions, mood, message\n"
                    "7. TECHNICAL DETAILS: Photography style, technique (if recognizable)\n"
                    "Write clearly and with good structure."
                )
            }
        }
        
        lang = language.lower()
        if lang not in prompts:
            lang = 'english'  # fallback
        
        if detail_level not in prompts[lang]:
            detail_level = 'detailed'  # fallback
        
        return prompts[lang][detail_level]
    
    def _call_vision_api_with_retry(self,
                                   base64_image: str,
                                   prompt: str,
                                   detail_level: str,
                                   filename: str,
                                   system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Call OpenAI Vision API với robust retry logic
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Vision API call attempt {attempt + 1}/{self.max_retries} for '{filename}'")
                
                # Determine optimal settings based on detail level
                max_tokens = {
                    'brief': 150,
                    'detailed': 800,
                    'extensive': 1500
                }.get(detail_level, 800)
                
                image_detail = "high" if detail_level == "extensive" else "auto"
                
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": image_detail
                            }
                        }
                    ]
                })

                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Best vision model as of 2025
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,  # Lower temperature for more consistent descriptions
                    timeout=self.timeout
                )
                
                description = response.choices[0].message.content.strip()
                
                if not description:
                    raise ValueError("Empty response from OpenAI Vision API")
                
                # Calculate cost estimate
                usage = response.usage
                cost_estimate = self._calculate_cost(usage) if usage else 0
                
                return {
                    'success': True,
                    'description': description,
                    'usage': {
                        'prompt_tokens': usage.prompt_tokens if usage else 0,
                        'completion_tokens': usage.completion_tokens if usage else 0,
                        'total_tokens': usage.total_tokens if usage else 0,
                        'cost_estimate': cost_estimate,
                        'model': response.model
                    }
                }
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Vision API attempt {attempt + 1} failed for '{filename}': {error_msg}")
                
                # Check if this is a retryable error
                if attempt < self.max_retries - 1:
                    if any(retryable in error_msg.lower() for retryable in 
                          ['rate limit', 'timeout', 'network', 'connection', 'server error', '429', '500', '502', '503']):
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.info(f"Retrying after {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                # Final attempt or non-retryable error
                return {
                    'success': False,
                    'error': f"Vision API error after {attempt + 1} attempts: {error_msg}"
                }
        
        return {
            'success': False,
            'error': f"Vision API failed after {self.max_retries} attempts"
        }
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on current OpenAI pricing"""
        if not usage:
            return 0.0
        
        # GPT-4o pricing (as of 2025)
        input_cost_per_token = 5.00 / 1_000_000   # $5 per 1M input tokens
        output_cost_per_token = 15.00 / 1_000_000  # $15 per 1M output tokens
        
        input_cost = usage.prompt_tokens * input_cost_per_token
        output_cost = usage.completion_tokens * output_cost_per_token

        return round(input_cost + output_cost, 6)

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Attempt to extract a JSON object from text."""
        try:
            if '```' in text:
                for part in text.split('```'):
                    part = part.strip()
                    if part.startswith('{') and part.endswith('}'):
                        return json.loads(part)
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except Exception as e:
            logger.debug(f"JSON parse failed: {e}")
        return None
    
    def _extract_comprehensive_metadata(self, image_data: bytes) -> Dict[str, Any]:
        """Extract comprehensive image metadata"""
        try:
            img = Image.open(io.BytesIO(image_data))
            
            metadata = {
                'format': img.format,
                'size': img.size,
                'mode': img.mode,
                'file_size_bytes': len(image_data),
                'file_size_mb': round(len(image_data) / (1024 * 1024), 2),
                'aspect_ratio': round(img.size[0] / img.size[1], 3) if img.size[1] > 0 else 0,
                'total_pixels': img.size[0] * img.size[1],
                'megapixels': round((img.size[0] * img.size[1]) / 1_000_000, 2)
            }
            
            # Extract EXIF if available
            try:
                exif = img._getexif()
                if exif:
                    metadata['has_exif'] = True
                    metadata['exif_data'] = {k: str(v) for k, v in exif.items() if k in [
                        'DateTime', 'Make', 'Model', 'Software', 'Artist'
                    ]}
                else:
                    metadata['has_exif'] = False
            except:
                metadata['has_exif'] = False
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {e}")
            return {
                'error': str(e),
                'file_size_bytes': len(image_data),
                'file_size_mb': round(len(image_data) / (1024 * 1024), 2)
            }
    
    def _create_error_response(self, error_message: str, filename: str, processing_time: float = 0) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            'success': False,
            'error': error_message,
            'description': None,
            'filename': filename,
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for monitoring"""
        return {
            'total_requests': self.total_requests,
            'total_tokens_used': self.total_tokens_used,
            'estimated_total_cost': round(self.total_tokens_used * 10.00 / 1_000_000, 4)  # Rough estimate
        }


# ===== STEP 2: Integration Functions =====
# Add to existing BE/core/document_api.py or create new file

def describe_image_with_openai_vision(image_data: bytes,
                                     filename: str,
                                     language: str = "vietnamese",
                                     detail_level: str = "detailed",
                                     custom_prompt: Optional[str] = None) -> Dict[str, Any]:
    """
    Wrapper function để integrate vision functionality vào existing system
    """
    try:
        from config.settings import CHATGPT_API_KEY
        
        if not CHATGPT_API_KEY or not CHATGPT_API_KEY.strip():
            return {
                'success': False,
                'error': 'OpenAI API key not configured in settings',
                'description': None
            }
        
        # Initialize vision client
        vision_client = OpenAIVisionClient(CHATGPT_API_KEY)
        
        # Perform analysis
        result = vision_client.describe_image(
            image_data=image_data,
            filename=filename,
            language=language,
            detail_level=detail_level,
            custom_prompt=custom_prompt
        )
        
        # Log usage for monitoring
        if result['success']:
            logger.info(f"Vision analysis successful: {filename} - "
                       f"{result['text_metrics']['word_count']} words, "
                       f"${result['api_usage']['cost_estimate']:.4f}")
        
        return result
        
    except Exception as e:
        logger.error(f"Vision analysis wrapper error for '{filename}': {e}", exc_info=True)
        return {
            'success': False,
            'error': f'System error in vision wrapper: {str(e)}',
            'description': None,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }


def detect_ai_image_with_openai_vision(image_data: bytes, filename: str) -> Dict[str, Any]:
    """Wrapper for AI-generated image detection."""
    try:
        from config.settings import CHATGPT_API_KEY

        if not CHATGPT_API_KEY or not CHATGPT_API_KEY.strip():
            return {
                'success': False,
                'error': 'OpenAI API key not configured in settings',
                'detection': None
            }

        vision_client = OpenAIVisionClient(CHATGPT_API_KEY)
        result = vision_client.detect_ai_generated_image(image_data=image_data, filename=filename)
        return result
    except Exception as e:
        logger.error(f"AI detection wrapper error for '{filename}': {e}", exc_info=True)
        return {
            'success': False,
            'error': f'System error in AI detection wrapper: {str(e)}',
            'detection': None,
            'filename': filename,
            'timestamp': datetime.now().isoformat()
        }


# ===== STEP 3: Update requirements.txt =====
# Add these dependencies to BE/requirements.txt:
"""
openai>=1.10.0
Pillow>=10.0.0
"""


# ===== STEP 4: Configuration Updates =====
# Add to BE/config/settings.py (if not already there):
"""
# OpenAI API Configuration  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
VISION_MAX_IMAGE_SIZE = int(os.getenv("VISION_MAX_IMAGE_SIZE", 20 * 1024 * 1024))  # 20MB
VISION_TIMEOUT = int(os.getenv("VISION_TIMEOUT", 60))  # seconds
"""