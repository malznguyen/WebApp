import json
import csv
import io
import zipfile
import tempfile
import os
from typing import Dict, Any

from BE.config.constants import DOWNLOAD_MAX_SIZE_MB

class DownloadGenerator:
    """Utility to generate download data securely."""
    @staticmethod
    def sanitize_filename(name: str) -> str:
        safe = ''.join(c for c in name if c.isalnum() or c in ('_', '-', '.'))
        return safe or 'download'

    @staticmethod
    def generate_json(data: Dict[str, Any]) -> bytes:
        return json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8')

    @staticmethod
    def generate_txt(text: str) -> bytes:
        return text.encode('utf-8')

    @staticmethod
    def generate_csv(rows: Dict[str, Any]) -> bytes:
        output = io.StringIO()
        if isinstance(rows, list):
            if rows and isinstance(rows[0], dict):
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        output_str = output.getvalue()
        return output_str.encode('utf-8')

    @staticmethod
    def generate_zip(files: Dict[str, bytes]) -> str:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for fname, data in files.items():
                zipf.writestr(fname, data)
        temp.close()
        if os.path.getsize(temp.name) > DOWNLOAD_MAX_SIZE_MB * 1024 * 1024:
            os.unlink(temp.name)
            raise ValueError('Generated zip exceeds size limit')
        return temp.name
