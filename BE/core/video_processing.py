import os
import json
import subprocess
import logging
from typing import Dict, Any, List, Optional

from openai import OpenAI
from BE.utils.helpers import ensure_dir_exists

logger = logging.getLogger('ImageSearchApp')

def _run_ffprobe(video_path: str) -> Dict[str, Any]:
    """Run ffprobe and return parsed JSON metadata."""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except FileNotFoundError:
        logger.error("ffprobe not found. Please install ffmpeg.")
        return {}
    except Exception as e:
        logger.error(f"ffprobe failed for {video_path}: {e}")
        return {}


def extract_comprehensive_metadata(video_path: str) -> Dict[str, Any]:
    """Extract basic and technical metadata, including GPS if available."""
    metadata = {
        'success': False,
        'basic': {},
        'technical': {},
        'gps': {},
        'exif': {},
    }
    if not os.path.isfile(video_path):
        metadata['error'] = 'File does not exist'
        return metadata
    data = _run_ffprobe(video_path)
    if not data:
        metadata['error'] = 'Could not parse metadata'
        return metadata
    try:
        format_info = data.get('format', {})
        streams = data.get('streams', [])
        video_stream = next((s for s in streams if s.get('codec_type') == 'video'), {})
        audio_stream = next((s for s in streams if s.get('codec_type') == 'audio'), {})

        metadata['basic'] = {
            'filename': os.path.basename(video_path),
            'duration': float(format_info.get('duration', 0)),
            'size': int(format_info.get('size', 0)),
            'bit_rate': int(format_info.get('bit_rate', 0)),
            'format': format_info.get('format_long_name'),
            'resolution': f"{video_stream.get('width')}x{video_stream.get('height')}" if video_stream else '',
        }
        metadata['technical'] = {
            'video_codec': video_stream.get('codec_name'),
            'audio_codec': audio_stream.get('codec_name'),
            'frame_rate': eval(video_stream.get('r_frame_rate', '0')) if video_stream else 0,
            'color_space': video_stream.get('color_space'),
        }
        # GPS and EXIF-like data
        tags = format_info.get('tags', {})
        if tags:
            gps_keys = [k for k in tags.keys() if 'location' in k.lower() or 'gps' in k.lower()]
            metadata['gps'] = {k: tags[k] for k in gps_keys}
            metadata['exif'] = tags
        metadata['success'] = True
    except Exception as e:  # pragma: no cover - defensive parsing
        logger.error(f"Metadata extraction failed for {video_path}: {e}")
        metadata['error'] = str(e)
    return metadata


def transcribe_with_timestamps(video_path: str, language: str = 'en', api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Transcribe audio with word-level timestamps using Whisper."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHATGPT_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not configured")
    try:
        client = OpenAI(api_key=api_key)
        with open(video_path, 'rb') as f:
            resp = client.audio.transcriptions.create(
                model='whisper-1',
                file=f,
                response_format='verbose_json',
                timestamp_granularities=['word'],
                language=language
            )
        return resp.get('segments', [])
    except Exception as e:
        logger.error(f"Transcription failed for {video_path}: {e}")
        raise


def export_transcript_multiple_formats(transcript: List[Dict[str, Any]], output_dir: str,
                                       base_filename: str) -> Dict[str, str]:
    """Export transcript to multiple formats."""
    ensure_dir_exists(output_dir)
    file_paths: Dict[str, str] = {}
    text_lines = []
    for seg in transcript:
        start = seg.get('start', 0)
        end = seg.get('end', 0)
        text = seg.get('text', '')
        speaker = seg.get('speaker', '')
        line = f"[{start:.2f}-{end:.2f}] {speaker} {text}".strip()
        text_lines.append(line)
    txt_path = os.path.join(output_dir, base_filename + '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_lines))
    file_paths['txt'] = txt_path

    # JSON
    json_path = os.path.join(output_dir, base_filename + '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
    file_paths['json'] = json_path

    # CSV
    import csv
    csv_path = os.path.join(output_dir, base_filename + '.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start', 'end', 'speaker', 'text'])
        for seg in transcript:
            writer.writerow([
                seg.get('start', 0),
                seg.get('end', 0),
                seg.get('speaker', ''),
                seg.get('text', '')
            ])
    file_paths['csv'] = csv_path

    # SRT
    srt_path = os.path.join(output_dir, base_filename + '.srt')
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(transcript, 1):
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            f.write(str(i) + '\n')
            f.write(_format_srt_time(start) + ' --> ' + _format_srt_time(end) + '\n')
            f.write(seg.get('text', '') + '\n\n')
    file_paths['srt'] = srt_path

    # DOCX
    try:
        from docx import Document
        doc = Document()
        for seg in transcript:
            ts = f"[{_format_srt_time(seg.get('start',0))}]"
            doc.add_paragraph(f"{ts} {seg.get('text','')}")
        docx_path = os.path.join(output_dir, base_filename + '.docx')
        doc.save(docx_path)
        file_paths['docx'] = docx_path
    except Exception as e:
        logger.warning(f"DOCX export failed: {e}")
    return file_paths


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def extract_audio_to_mp3(video_path: str, output_path: str) -> str:
    """Extract audio track to an MP3 file using ffmpeg."""
    try:
        cmd = [
            'ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'libmp3lame', output_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        logger.error(f"Audio extraction failed for {video_path}: {e}")
        raise


def extract_thumbnails(video_path: str, output_dir: str, interval: int = 10) -> List[str]:
    """Extract thumbnails every `interval` seconds using ffmpeg."""
    ensure_dir_exists(output_dir)
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-vf', f"fps=1/{interval}",
            os.path.join(output_dir, 'thumb_%04d.jpg')
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('thumb_')])
        return files
    except Exception as e:
        logger.error(f"Thumbnail extraction failed for {video_path}: {e}")
        return []


def generate_video_analysis_report(metadata: Dict[str, Any], transcript_files: Dict[str, str],
                                   thumbnails: List[str], output_path: str) -> str:
    """Generate a simple analysis report summarizing metadata and transcript."""
    try:
        from docx import Document
        doc = Document()
        doc.add_heading('Video Analysis Report', 0)
        doc.add_heading('Metadata', level=1)
        for k, v in metadata.get('basic', {}).items():
            doc.add_paragraph(f"{k}: {v}")
        doc.add_heading('Transcript', level=1)
        txt = transcript_files.get('txt')
        if txt and os.path.isfile(txt):
            with open(txt, 'r', encoding='utf-8') as f:
                for line in f:
                    doc.add_paragraph(line.strip())
        doc.add_heading('Thumbnails', level=1)
        for thumb in thumbnails:
            doc.add_picture(thumb, width=None)
        doc.save(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise

__all__ = [
    'extract_comprehensive_metadata',
    'transcribe_with_timestamps',
    'extract_audio_to_mp3',
    'export_transcript_multiple_formats',
    'extract_thumbnails',
    'generate_video_analysis_report',
]
