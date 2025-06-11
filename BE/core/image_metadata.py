import io
import logging
import hashlib
from typing import Any, Dict, Optional

from PIL import Image

try:
    import piexif  # type: ignore
except Exception:  # piexif is optional
    piexif = None

try:
    import imagehash  # type: ignore
except Exception:  # imagehash is optional but required by requirements
    imagehash = None

logger = logging.getLogger('ImageSearchApp')


class ImageMetadataExtractor:
    """Extracts image metadata with privacy safeguards."""

    def __init__(self, privacy_safe: bool = True) -> None:
        self.privacy_safe = privacy_safe
        self.piexif_available = piexif is not None
        self.imagehash_available = imagehash is not None

    def extract_metadata(self, image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
        """Extract basic info, EXIF data and fingerprints."""
        result: Dict[str, Any] = {
            "success": False,
            "basic": {},
            "exif": {},
            "fingerprints": {},
            "warnings": [],
        }

        if not image_bytes:
            result["error"] = "Empty image data"
            return result

        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()
            img = Image.open(io.BytesIO(image_bytes))

            result["basic"] = {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": len(image_bytes),
                "size_mb": round(len(image_bytes) / (1024 * 1024), 2),
            }

            exif_data, warnings = self._parse_exif(image_bytes, img)
            result["exif"] = exif_data
            result["warnings"] = warnings
            result["fingerprints"] = self._generate_fingerprints(image_bytes, img)
            result["success"] = True
            return result
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Error extracting metadata for '{filename}': {e}", exc_info=True)
            result["error"] = str(e)
            return result

    def _parse_exif(self, image_bytes: bytes, img: Image.Image) -> (Dict[str, Any], list):
        warnings = []
        metadata: Dict[str, Any] = {}

        if self.piexif_available:
            try:
                exif_dict = piexif.load(image_bytes)
                for ifd_name, ifd_data in exif_dict.items():
                    if ifd_name == "thumbnail" or not isinstance(ifd_data, dict):
                        continue
                    if self.privacy_safe and ifd_name == "GPS":
                        warnings.append("GPS data omitted for privacy")
                        continue
                    for tag, value in ifd_data.items():
                        tag_name = piexif.TAGS[ifd_name][tag]["name"]
                        if self.privacy_safe and tag_name.lower() in {
                            "makernote",
                            "bodyserialnumber",
                            "serialnumber",
                            "lensserialnumber",
                            "cameraserialnumber",
                            "usercomment",
                        }:
                            warnings.append(f"{tag_name} omitted for privacy")
                            continue
                        if isinstance(value, bytes):
                            try:
                                value = value.decode("utf-8", errors="replace")
                            except Exception:
                                value = str(value)
                        metadata[tag_name] = value
            except Exception as e:  # pragma: no cover - don't crash on exif
                logger.warning(f"piexif failed to parse EXIF: {e}")
        else:
            try:
                raw_exif = img._getexif()
                if raw_exif:
                    from PIL.ExifTags import TAGS
                    for tag_id, value in raw_exif.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        if self.privacy_safe and tag_name == "GPSInfo":
                            warnings.append("GPS data omitted for privacy")
                            continue
                        if self.privacy_safe and tag_name in {
                            "MakerNote",
                            "BodySerialNumber",
                            "SerialNumber",
                            "LensSerialNumber",
                            "CameraSerialNumber",
                            "UserComment",
                        }:
                            warnings.append(f"{tag_name} omitted for privacy")
                            continue
                        metadata[tag_name] = value
            except Exception as e:  # pragma: no cover
                logger.warning(f"Pillow EXIF parsing failed: {e}")

        return metadata, warnings

    def _generate_fingerprints(self, image_bytes: bytes, img: Image.Image) -> Dict[str, str]:
        fingerprints: Dict[str, str] = {}
        try:
            fingerprints["md5"] = hashlib.md5(image_bytes).hexdigest()
            fingerprints["sha256"] = hashlib.sha256(image_bytes).hexdigest()
        except Exception as e:  # pragma: no cover
            logger.error(f"Error hashing image data: {e}")

        if self.imagehash_available:
            try:
                fingerprints["perceptual"] = str(imagehash.phash(img))
            except Exception as e:  # pragma: no cover
                logger.error(f"Error generating perceptual hash: {e}")
        return fingerprints

