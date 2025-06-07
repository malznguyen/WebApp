import os
import logging
import datetime
import time

class DebounceTimer:
    def __init__(self, min_interval=1.0):
        self.last_time = 0
        self.min_interval = min_interval
        
    def should_run(self):
        current_time = time.time()
        if current_time - self.last_time >= self.min_interval:
            self.last_time = current_time
            return True
        return False

def setup_logger():
    # tạo thư mục logs
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # tên file log với timestamp
    log_filename = f"logs/app_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # tạo logger
    logger = logging.getLogger('ImageSearchApp')
    logger.setLevel(logging.DEBUG)
    
    # bỏ millisecond cho đẹp
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s', 
                              '%Y-%m-%d %H:%M:%S')
    
    # log ra file
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # log ra console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # thêm handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger