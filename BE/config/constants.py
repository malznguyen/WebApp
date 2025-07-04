# Log levels
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Giới hạn hiển thị kết quả
MAX_RESULTS_TO_DISPLAY = 100  # Hiển thị tối đa 100 kết quả

# Timeout 
SEARCH_TIMEOUT_MS = 100000 
API_TIMEOUT_SEC = 180

# Document summary
MAX_CHUNK_SIZE = 100000  # Kích thước tối đa của một chunk văn bản
SUPPORTED_DOC_FORMATS = ['.pdf', '.docx', '.txt', '.md']

# Kích thước ảnh xem trước
PREVIEW_MAX_SIZE = 150

# Kích thước ảnh tối đa (MB) trước khi nén
MAX_IMAGE_SIZE_MB = 5

# Tỷ lệ nén ảnh
IMAGE_RESIZE_RATIO = 0.7
IMAGE_QUALITY = 85

# Maximum size (in bytes) allowed for decoded base64 payloads
MAX_BASE64_DECODE_SIZE = 50 * 1024 * 1024  # 50MB safety limit

# Danh sách các domain mạng xã hội nổi tiếng cần lọc
SOCIAL_MEDIA_DOMAINS = [
    "facebook.com", 
    "instagram.com",
    "twitter.com",  
    "x.com",        
    "tiktok.com",
    "youtube.com",
    "linkedin.com",
    "pinterest.com",
    "reddit.com",
    "tumblr.com",
    "snapchat.com",
    "threads.net",
    "vk.com",
    "weibo.com",
    "flickr.com",
    "medium.com",
    "quora.com",
    "zalo.me",
    "imgur.com",
    "vimeo.com",
    "dailymotion.com"
]

# Chọn chỉ lọc kết quả mạng xã hội mặc định
SOCIAL_MEDIA_FILTER_DEFAULT = True

# Download limits
DOWNLOAD_MAX_SIZE_MB = 50
