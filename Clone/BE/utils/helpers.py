import os
import datetime
import logging

logger = logging.getLogger('ImageSearchApp')

def ensure_dir_exists(directory):
    """tạo thư mục nếu chưa có"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"tạo thư mục: {directory}")
    return directory

def generate_timestamp():
    """tạo timestamp cho tên file"""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def format_file_size(size_bytes):
    """format kích thước file"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.2f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"