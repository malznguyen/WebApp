import os
import io
import base64
import logging
from PIL import Image
from config.constants import MAX_IMAGE_SIZE_MB, IMAGE_RESIZE_RATIO, IMAGE_QUALITY, PREVIEW_MAX_SIZE

logger = logging.getLogger('ImageSearchApp')


def optimize_image(image_path):
    file_size = os.path.getsize(image_path) / (1024 * 1024)
    
    if file_size > MAX_IMAGE_SIZE_MB:
        logger.info(f"Image larger than {MAX_IMAGE_SIZE_MB}MB ({file_size:.2f}MB), compressing...")
        img = Image.open(image_path)
        
        new_width = int(img.width * IMAGE_RESIZE_RATIO)
        new_height = int(img.height * IMAGE_RESIZE_RATIO)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        img_format = 'JPEG' if img.mode == 'RGB' else 'PNG'
        img.save(buffer, format=img_format, quality=IMAGE_QUALITY)
        binary_data = buffer.getvalue()
        logger.info(f"Compressed image, new size: {len(binary_data) / (1024 * 1024):.2f} MB")
        return binary_data
    else:
        with open(image_path, "rb") as image_file:
            binary_data = image_file.read()
        logger.info(f"Read image, size: {len(binary_data) / (1024 * 1024):.2f} MB")
        return binary_data


def optimize_image_bytes(image_bytes, max_size_mb=None):
    if max_size_mb is None:
        max_size_mb = MAX_IMAGE_SIZE_MB
    
    current_size = len(image_bytes) / (1024 * 1024)
    
    if current_size <= max_size_mb:
        logger.info(f"Image size OK: {current_size:.2f}MB")
        return image_bytes
    
    logger.info(f"Optimizing image: {current_size:.2f}MB > {max_size_mb}MB")
    
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        new_width = int(img.width * IMAGE_RESIZE_RATIO)
        new_height = int(img.height * IMAGE_RESIZE_RATIO)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        img_format = 'JPEG' if img.mode == 'RGB' else 'PNG'
        img.save(buffer, format=img_format, quality=IMAGE_QUALITY)
        
        optimized_bytes = buffer.getvalue()
        new_size = len(optimized_bytes) / (1024 * 1024)
        logger.info(f"Optimized image: {current_size:.2f}MB → {new_size:.2f}MB")
        
        return optimized_bytes
        
    except Exception as e:
        logger.error(f"Error optimizing image bytes: {e}", exc_info=True)
        return image_bytes


def create_preview(image_path):
    try:
        img = Image.open(image_path)
        logger.debug(f"Original image size: {img.size}")
        
        width, height = img.size
        max_size = PREVIEW_MAX_SIZE
        
        ratio = min(max_size / width, max_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        logger.debug(f"Resizing image to: {new_width}x{new_height}")
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return buffer.getvalue()
            
    except Exception as e:
        logger.error(f"Error creating preview: {str(e)}", exc_info=True)
        return None


def create_preview_from_bytes(image_bytes, max_size=None):
    if max_size is None:
        max_size = PREVIEW_MAX_SIZE
        
    try:
        img = Image.open(io.BytesIO(image_bytes))
        logger.debug(f"Original image size: {img.size}")
        
        width, height = img.size
        ratio = min(max_size / width, max_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        logger.debug(f"Creating preview: {new_width}x{new_height}")
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating preview from bytes: {e}", exc_info=True)
        return None


def create_preview_base64(image_path, max_size=None):
    preview_bytes = create_preview(image_path)
    if preview_bytes:
        b64_data = base64.b64encode(preview_bytes).decode('utf-8')
        return f"data:image/png;base64,{b64_data}"
    return None


def create_preview_base64_from_bytes(image_bytes, max_size=None):
    """Create base64 preview from bytes for web display"""
    preview_bytes = create_preview_from_bytes(image_bytes, max_size)
    if preview_bytes:
        b64_data = base64.b64encode(preview_bytes).decode('utf-8')
        return f"data:image/png;base64,{b64_data}"
    return None


def verify_image(image_path):
    try:
        with open(image_path, "rb") as f:
            img_test = Image.open(f)
            img_test.verify()
            logger.debug(f"Verified valid image: {image_path}")
            return True
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}", exc_info=True)
        return False


def verify_image_bytes(image_bytes):
    try:
        img_test = Image.open(io.BytesIO(image_bytes))
        img_test.verify()
        logger.debug("Verified valid image from bytes")
        return True
    except Exception as e:
        logger.error(f"Invalid image bytes: {e}", exc_info=True)
        return False


def get_image_info(image_path):
    try:
        with Image.open(image_path) as img:
            file_size = os.path.getsize(image_path)
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": file_size,
                "size_mb": file_size / (1024 * 1024),
                "aspect_ratio": img.width / img.height if img.height > 0 else 1
            }
    except Exception as e:
        logger.error(f"Error getting image info: {e}")
        return None


def get_image_info_from_bytes(image_bytes):
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": len(image_bytes),
                "size_mb": len(image_bytes) / (1024 * 1024),
                "aspect_ratio": img.width / img.height if img.height > 0 else 1
            }
    except Exception as e:
        logger.error(f"Error getting image info from bytes: {e}")
        return None


def save_uploaded_image(image_bytes, filename, upload_dir="temp"):
    try:
        os.makedirs(upload_dir, exist_ok=True)
        
        # Sanitize filename
        import re
        safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        file_path = os.path.join(upload_dir, safe_filename)
        
        # Add timestamp if file exists
        if os.path.exists(file_path):
            import time
            name, ext = os.path.splitext(safe_filename)
            timestamp = int(time.time())
            file_path = os.path.join(upload_dir, f"{name}_{timestamp}{ext}")
        
        with open(file_path, "wb") as f:
            f.write(image_bytes)
        
        logger.info(f"Saved uploaded image: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Error saving uploaded image: {e}", exc_info=True)
        return None


def process_web_upload(image_data, filename):
    """Complete processing pipeline for web uploads"""
    try:
        # Decode base64
        if isinstance(image_data, str) and image_data.startswith('data:'):
            header, data = image_data.split(',', 1)
            image_bytes = base64.b64decode(data)
        else:
            image_bytes = image_data
        
        # Verify image
        if not verify_image_bytes(image_bytes):
            return {"error": "Invalid image format"}
        
        # Get image info
        img_info = get_image_info_from_bytes(image_bytes)
        if not img_info:
            return {"error": "Cannot read image information"}
        
        # Optimize
        optimized_bytes = optimize_image_bytes(image_bytes)
        
        # Create preview
        preview_b64 = create_preview_base64_from_bytes(optimized_bytes)
        
        # Save to temp file
        temp_path = save_uploaded_image(optimized_bytes, filename)
        if not temp_path:
            return {"error": "Cannot save uploaded image"}
        
        return {
            "success": True,
            "temp_path": temp_path,
            "image_info": img_info,
            "preview_base64": preview_b64,
            "optimized_size_mb": len(optimized_bytes) / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Error processing web upload: {e}", exc_info=True)
        return {"error": str(e)}


def cleanup_temp_files(temp_dir="temp", max_age_hours=24):
    """Clean up old temporary files"""
    try:
        if not os.path.exists(temp_dir):
            return
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        cleaned_count = 0
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        os.unlink(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Could not delete temp file {file_path}: {e}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")


def validate_image_upload(image_bytes, max_size_mb=None, allowed_formats=None):
    """Validate uploaded image"""
    if max_size_mb is None:
        max_size_mb = MAX_IMAGE_SIZE_MB * 2  # More lenient for web uploads
    
    if allowed_formats is None:
        allowed_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']
    
    try:
        # Check size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > max_size_mb:
            return {"valid": False, "error": f"Image too large: {size_mb:.1f}MB > {max_size_mb}MB"}
        
        # Check format
        img = Image.open(io.BytesIO(image_bytes))
        if img.format not in allowed_formats:
            return {"valid": False, "error": f"Unsupported format: {img.format}. Allowed: {', '.join(allowed_formats)}"}
        
        # Check dimensions
        if img.width < 50 or img.height < 50:
            return {"valid": False, "error": "Image too small (minimum 50x50 pixels)"}
        
        if img.width > 10000 or img.height > 10000:
            return {"valid": False, "error": "Image too large (maximum 10000x10000 pixels)"}
        
        return {"valid": True, "format": img.format, "size": (img.width, img.height), "size_mb": size_mb}
        
    except Exception as e:
        return {"valid": False, "error": f"Cannot process image: {str(e)}"}