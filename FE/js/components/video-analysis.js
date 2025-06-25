// Video Analysis Component
// Provides utilities for video upload, metadata display and transcript export

const VideoAnalysis = {
    allowedExtensions: ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
    maxSize: 500 * 1024 * 1024, // 500MB

    validateVideoFile(file) {
        const ext = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        if (!this.allowedExtensions.includes(ext)) {
            return { valid: false, error: 'Unsupported format' };
        }
        if (file.size > this.maxSize) {
            return { valid: false, error: 'File too large (max 500MB)' };
        }
        return { valid: true };
    },

    formatMetadata(data) {
        if (!data || !data.success) return null;
        return {
            basic: data.basic || {},
            technical: data.technical || {},
            gps: data.gps || {},
            exif: data.exif || {}
        };
    },

    secondsToTime(sec) {
        const h = Math.floor(sec / 3600).toString().padStart(2, '0');
        const m = Math.floor((sec % 3600) / 60).toString().padStart(2, '0');
        const s = Math.floor(sec % 60).toString().padStart(2, '0');
        return `${h}:${m}:${s}`;
    },

    transcriptText(transcript) {
        if (!transcript || !transcript.segments) return '';
        return transcript.segments.map(seg => seg.text).join('\n');
    }
};

window.VideoAnalysis = VideoAnalysis;
